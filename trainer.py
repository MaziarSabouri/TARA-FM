import argparse
import gc
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

from utils.general_utils import load_config, resolve_mask_conditioning_config
from fm_lightning import FlowMatchingDataModule, FlowMatchingLightningModule
from utils.checkpoint_utils import resolve_resume_checkpoint


def _should_use_ddp(accelerator, devices) -> bool:
    if not torch.cuda.is_available():
        return False

    if isinstance(accelerator, str) and accelerator not in {"auto", "gpu", "cuda"}:
        return False

    if isinstance(devices, int):
        if devices == -1:
            return torch.cuda.device_count() > 1
        return devices > 1
    if isinstance(devices, (list, tuple)):
        return len(devices) > 1
    if isinstance(devices, str):
        devices = devices.strip().lower()
        if devices in {"auto", "-1"}:
            return torch.cuda.device_count() > 1
        if "," in devices:
            return len([d for d in devices.split(",") if d.strip()]) > 1
        try:
            return int(devices) > 1
        except ValueError:
            pass
        return False
    return False


def _is_oom_error(err: BaseException) -> bool:
    msg = str(err).lower()
    return (
        "out of memory" in msg
        or "cuda error: out of memory" in msg
        or "cublas_status_alloc_failed" in msg
    )


def _find_max_batch_size(config: dict, datamodule: FlowMatchingDataModule, start_bsz: int) -> int:
    tr = config.get("train_args", {})
    start_bsz = max(1, int(start_bsz))

    if not torch.cuda.is_available():
        print("[auto_batch_size] CUDA not available; keeping configured batch_size.")
        return start_bsz

    datamodule.setup("fit")
    max_bsz = None
    try:
        imgs = datamodule.train_data["images"]  # type: ignore[index]
        max_bsz = int(imgs.shape[0])
    except Exception:
        max_bsz = None

    if max_bsz is not None:
        start_bsz = min(start_bsz, max_bsz)
        hard_cap = max_bsz
    else:
        hard_cap = 4096

    device = torch.device("cuda")
    lr = float(tr.get("lr", 1e-5))
    model = FlowMatchingLightningModule(config).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.model.parameters(), lr=lr)

    cache = {}

    def can_fit(batch_size: int) -> bool:
        batch_size = int(batch_size)
        if batch_size < 1:
            return False
        if max_bsz is not None and batch_size > max_bsz:
            return False
        if batch_size in cache:
            return cache[batch_size]

        config.setdefault("train_args", {})["batch_size"] = batch_size
        ok = False
        try:
            loader = datamodule.train_dataloader()
            batch = next(iter(loader))
            batch = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            loss = model._compute_loss(batch)
            loss.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ok = True
            return True
        except RuntimeError as e:
            if _is_oom_error(e):
                return False
            raise
        finally:
            cache[batch_size] = ok
            optimizer.zero_grad(set_to_none=True)
            try:
                del batch  # type: ignore[possibly-undefined]
            except Exception:
                pass
            try:
                del loss  # type: ignore[possibly-undefined]
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print(f"[auto_batch_size] Searching max batch size (start={start_bsz})...")

    # Find a lower bound that fits.
    low = start_bsz
    while low > 1 and not can_fit(low):
        low //= 2
    if not can_fit(low):
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError("[auto_batch_size] batch_size=1 still OOM; reduce model size/precision.")

    # Find an upper bound that fails (or hit cap).
    high = min(hard_cap, low * 2)
    while low < hard_cap and high <= hard_cap and can_fit(high):
        low = high
        if low >= hard_cap:
            break
        high = min(hard_cap, low * 2)

    if low >= hard_cap:
        found = low
    else:
        # Ensure high is a failing bound.
        if can_fit(high):
            found = high
        else:
            left, right = low, high
            while right - left > 1:
                mid = (left + right) // 2
                if can_fit(mid):
                    left = mid
                else:
                    right = mid
            found = left

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"[auto_batch_size] Using batch_size={found}")
    return int(found)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the flow matching model with Lightning.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/fm_mask_class.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--auto_batch_size",
        action="store_true",
        help="Find the max batch size that fits (single forward/backward/step) and override config.",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    config["train_args"] = config["fm_train_args"]
    resolve_mask_conditioning_config(config)
    run_name = os.path.splitext(os.path.basename(args.config_path))[0]
    tr = config["train_args"]
    root_ckpt_dir = os.path.abspath(os.path.expanduser(tr["checkpoint_dir"]))

    # Data and model modules
    datamodule = FlowMatchingDataModule(config)
    if bool(tr.get("auto_batch_size", False) or args.auto_batch_size):
        tr["batch_size"] = _find_max_batch_size(config, datamodule, tr.get("batch_size", 1))
    model = FlowMatchingLightningModule(config)

    # Logging and callbacks
    logger = TensorBoardLogger(save_dir=root_ckpt_dir, name=run_name)
    ckpt_every = max(1, int(tr.get("val_freq", 5)))
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(root_ckpt_dir, run_name),
        filename="epoch{epoch:03d}-valloss{val/loss:.6f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
        every_n_epochs=ckpt_every,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    cbs = [ckpt_cb, lr_cb]

    resume_ckpt = resolve_resume_checkpoint(root_ckpt_dir, run_name)
    if resume_ckpt:
        print(f"Resuming training from checkpoint: {resume_ckpt}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    bf16_supported = (
        torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    default_precision = (
        "bf16-mixed" if bf16_supported else ("16-mixed" if torch.cuda.is_available() else "32-true")
    )
    precision = tr.get("precision", default_precision)

    trainer = pl.Trainer(
        default_root_dir=root_ckpt_dir,
        max_epochs=tr["num_epochs"],
        precision=precision,
        accumulate_grad_batches=tr.get("gradient_accumulation_steps", 8),
        gradient_clip_val=tr.get("grad_clip_norm", 0.0) or None,
        check_val_every_n_epoch=ckpt_every,
        enable_progress_bar=True,
        logger=logger,
        callbacks=cbs,
        # Distributed/accelerator knobs
        accelerator=tr.get("accelerator", "auto"),
        devices=tr.get("devices", "auto"),
        strategy=(
            DDPStrategy(find_unused_parameters=True)
            if _should_use_ddp(tr.get("accelerator", "auto"), tr.get("devices", "auto"))
            else tr.get("strategy", "auto")
        ),
        deterministic=tr.get("deterministic", False),
        log_every_n_steps=tr.get("log_every_n_steps", 50),
        num_sanity_val_steps=tr.get("num_sanity_val_steps", 0),
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
