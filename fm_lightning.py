from typing import Optional

import torch
import torch.optim as optim

import pytorch_lightning as pl

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler

from utils.general_utils import (
    create_dataloader,
    load_and_prepare_data,
    resolve_allowed_ids,
    resolve_mask_conditioning_config,
)
from utils.utils_fm import (
    build_model,
    compute_fm_mse_loss,
    save_validation_outputs,
)


class FlowMatchingDataModule(pl.LightningDataModule):
    """Lightning DataModule wrapping the existing data helpers."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.train_data: Optional[dict] = None
        self.val_data: Optional[dict] = None

    def setup(self, stage: Optional[str] = None) -> None:
        resolve_mask_conditioning_config(self.config)
        data_config = self.config["data_args"]
        allowed_ids = resolve_allowed_ids(data_config)

        def _load(split: str) -> dict:
            return load_and_prepare_data(
                pickle_path=data_config["pickle_path"],
                split=split,
                convert_classes_to_onehot=True,
                mask_num_classes=data_config.get("mask_num_classes"),
                allowed_ids=allowed_ids,
            )

        if stage in (None, "fit"):
            if self.train_data is None:
                self.train_data = _load(data_config["split_train"])
            if self.val_data is None:
                self.val_data = _load(data_config["split_val"])
        elif stage == "validate":
            if self.val_data is None:
                self.val_data = _load(data_config["split_val"])

    def _make_loader(self, data: dict, *, shuffle: bool, drop_last: bool) -> torch.utils.data.DataLoader:
        tr = self.config["train_args"]
        return create_dataloader(
            Images=data["images"],
            Masks=data["masks"],
            classes=data.get("classes"),
            batch_size=tr["batch_size"],
            shuffle=shuffle,
            num_workers=int(tr.get("num_workers", 0)),
            pin_memory=tr.get("pin_memory", None),
            persistent_workers=tr.get("persistent_workers", None),
            drop_last=drop_last,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self._make_loader(
            self.train_data,
            shuffle=True,
            drop_last=bool(self.config["train_args"].get("drop_last", False)),
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self._make_loader(self.val_data, shuffle=False, drop_last=False)


class FlowMatchingLightningModule(pl.LightningModule):
    """Lightning Module for the flow matching model."""

    def __init__(self, config: dict):
        super().__init__()
        resolve_mask_conditioning_config(config)
        self.save_hyperparameters(config)
        self.model = build_model(config["model_args"])
        self.mask_conditioning = config["model_args"]["mask_conditioning"]
        self.class_conditioning = config["model_args"]["with_conditioning"]
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self._did_batch_sanity_check = False

    def _rank_zero_print(self, msg: str) -> None:
        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        if getattr(trainer, "is_global_zero", True):
            print(msg)

    def _maybe_sanity_check_batch(self, batch: dict) -> None:
        if self._did_batch_sanity_check:
            return
        self._did_batch_sanity_check = True

        imgs = batch.get("images")
        if not torch.is_tensor(imgs):
            raise TypeError(f"Expected batch['images'] to be a Tensor, got {type(imgs)}")
        if imgs.dim() != 4:
            raise ValueError(f"Expected images of shape [B,C,H,W], got {tuple(imgs.shape)}")
        if torch.isnan(imgs).any() or torch.isinf(imgs).any():
            raise ValueError("Found NaN/Inf in input images.")

        img_min, img_max = float(imgs.min().item()), float(imgs.max().item())
        if img_min < -1.0 - 1e-3 or img_max > 1.0 + 1e-3:
            self._rank_zero_print(
                f"[WARN] Input images out of [-1,1] range: "
                f"min={img_min:.4f}, max={img_max:.4f}. Check image scaling."
            )

        if self.mask_conditioning:
            masks = batch.get("masks")
            if masks is None:
                raise ValueError("mask_conditioning=True but batch['masks'] is missing.")
            if not torch.is_tensor(masks):
                raise TypeError(f"Expected batch['masks'] to be a Tensor, got {type(masks)}")
            if masks.dim() != 4:
                raise ValueError(f"Expected masks of shape [B,C,H,W], got {tuple(masks.shape)}")
            if masks.shape[0] != imgs.shape[0] or masks.shape[2:] != imgs.shape[2:]:
                raise ValueError(
                    f"Mask/image shape mismatch: images={tuple(imgs.shape)}, masks={tuple(masks.shape)}"
                )
            if torch.isnan(masks).any() or torch.isinf(masks).any():
                raise ValueError("Found NaN/Inf in conditioning masks.")
            mask_min, mask_max = float(masks.min().item()), float(masks.max().item())
            if mask_min < -1e-3 or mask_max > 1.0 + 1e-3:
                self._rank_zero_print(
                    f"[WARN] Conditioning masks out of [0,1] range: min={mask_min:.4f}, max={mask_max:.4f}. "
                    "Check mask values."
                )

        if self.class_conditioning:
            classes = batch.get("classes")
            if classes is None:
                raise ValueError("with_conditioning=True but batch['classes'] is missing.")
            if not torch.is_tensor(classes):
                raise TypeError(f"Expected batch['classes'] to be a Tensor, got {type(classes)}")
            if classes.dim() not in (1, 2):
                raise ValueError(
                    f"Expected classes of shape [B] or [B,C], got {tuple(classes.shape)}"
                )

            expected = self.hparams.get("model_args", {}).get("cross_attention_dim")
            if (expected is not None) and (classes.dim() == 2) and (classes.shape[1] != expected):
                self._rank_zero_print(
                    f"[WARN] Class conditioning dim mismatch: batch has {classes.shape[1]}, "
                    f"config cross_attention_dim={expected}."
                )
            if classes.dim() == 2:
                row_sums = classes.to(torch.float32).sum(dim=1)
                if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=0.0):
                    self._rank_zero_print(
                        "[WARN] Class conditioning does not look one-hot (row sums != 1). "
                        "If you intended integer class labels, convert to one-hot before training."
                    )

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        self._maybe_sanity_check_batch(batch)
        return compute_fm_mse_loss(
            self.model, self.path, batch, self.mask_conditioning, self.class_conditioning
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self._compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss = self._compute_loss(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.model.parameters(), lr=self.hparams["train_args"]["lr"])

    def on_validation_epoch_end(self) -> None:
        save_validation_outputs(
            self,
            mask_conditioning=self.mask_conditioning,
            class_conditioning=self.class_conditioning,
        )
