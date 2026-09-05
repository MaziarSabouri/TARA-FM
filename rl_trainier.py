import argparse
import copy
import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import optim

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler

from utils.general_utils import (
    image_to_display_range,
    load_config,
    resolve_mask_conditioning_config,
)
from utils.utils_fm import (
    autocast_for,
    build_model,
    compute_fm_mse_loss,
    save_validation_outputs,
    sample_with_solver,
)
from utils.checkpoint_utils import (
    resolve_checkpoint_path,
    load_model_from_checkpoint,
    resolve_resume_checkpoint,
)
from utils.solver_utils import build_solver_config
from fm_lightning import FlowMatchingDataModule
from rewards import compute_rewards


def _repeat_batch(batch: Dict[str, torch.Tensor], repeats: int) -> Dict[str, torch.Tensor]:
    if repeats == 1:
        return batch
    return {k: v.repeat_interleave(repeats, dim=0) for k, v in batch.items() if torch.is_tensor(v)}


def _group_softmax_weights(
    rewards: torch.Tensor,
    group_size: int,
    temperature: float,
    adv_alpha: Optional[float] = None,
    weight_clip: Optional[float] = None,
) -> torch.Tensor:
    """Convert rewards to per-group advantages -> exp(alpha*A) weights with mean=1 per group.

    Optionally clip weights to [1/clip, clip] and re-normalize.
    """
    # Standardize rewards within each GRPO group (z-score) -> advantage A.
    r = rewards.view(-1, group_size)
    A = (r - r.mean(dim=1, keepdim=True)) / r.std(
        dim=1, keepdim=True, unbiased=False
    ).clamp_min(1e-6)
    # Soft selection: w_i ∝ exp(alpha * A_i), renormalized so the per-group mean is 1.
    alpha = adv_alpha if (adv_alpha is not None) else (1.0 / max(1e-6, float(temperature)))
    w = torch.exp(alpha * A)
    w = w / (w.mean(dim=1, keepdim=True) + 1e-8)
    if weight_clip is not None and weight_clip > 1.0:
        w = w.clamp(min=1.0 / float(weight_clip), max=float(weight_clip))
        w = w / (w.mean(dim=1, keepdim=True) + 1e-8)
    return w.view(-1).detach()


def _group_topk_weights(
    rewards: torch.Tensor,
    group_size: int,
    top_k: int,
    reward_floor: Optional[float] = None,
) -> torch.Tensor:
    """Hard best-of-N selection: keep the top `top_k` candidates per group, drop the rest.

    Lower variance than the exponential weights when the group is small (G=4 gives
    a 4-sample estimate of the advantage z-score), and it is the standard
    rejection-sampling / RAFT weighting. `reward_floor` additionally discards
    accepted candidates whose absolute reward is too low, so a group in which
    every rollout is bad contributes no gradient instead of distilling its least
    bad member. Groups with nothing kept get all-zero weights.
    """
    r = rewards.view(-1, group_size)
    k_eff = max(1, min(int(top_k), group_size))
    keep = torch.zeros_like(r)
    keep.scatter_(1, r.topk(k_eff, dim=1).indices, 1.0)
    if reward_floor is not None:
        keep = keep * (r >= float(reward_floor)).to(keep.dtype)
    n_keep = keep.sum(dim=1, keepdim=True)
    w = torch.where(n_keep > 0, keep * (group_size / n_keep.clamp_min(1.0)), keep)
    return w.view(-1).detach()


def _adv_weighted_flow_matching_loss(
    model: torch.nn.Module,
    path: AffineProbPath,
    batch_rep: Dict[str, torch.Tensor],
    x0s: torch.Tensor,
    weights: torch.Tensor,
    class_conditioning: bool,
    mask_conditioning: bool,
    ref_model: Optional[torch.nn.Module] = None,
    lam_ref: float = 0.0,
    x1_target: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Advantage-weighted FM loss + optional teacher-proximity (KL-proxy) regularizer.

    `x1_target` is the endpoint the probability path is regressed towards. Passing
    the model's own high-reward rollouts turns this into reward-weighted
    self-distillation; leaving it None regresses towards the real image, in which
    case the reward can only reweight noise/image pairs (see `objective` in
    RLFlowMatchingLightningModule).
    """
    imgs = x1_target if x1_target is not None else batch_rep["images"]
    t = torch.rand(imgs.shape[0], device=imgs.device)
    info = path.sample(t=t, x_0=x0s, x_1=imgs)

    cond = (
        batch_rep["classes"].unsqueeze(1)
        if (class_conditioning and ("classes" in batch_rep))
        else None
    )
    masks = batch_rep["masks"] if (mask_conditioning and ("masks" in batch_rep)) else None

    v_pred = model(x=info.x_t, t=info.t, cond=cond, masks=masks)
    # Reward-weighted FM loss: high-reward samples contribute more to the gradient.
    per_sample = F.mse_loss(v_pred, info.dx_t, reduction="none").flatten(1).mean(dim=1)
    fm_loss = (weights.detach() * per_sample).mean()

    # KL-proxy regularizer: keeps fine-tuned velocity field close to the frozen base model.
    if (ref_model is not None) and (lam_ref > 0.0):
        with torch.no_grad():
            v_ref = ref_model(x=info.x_t, t=info.t, cond=cond, masks=masks)
        return fm_loss + lam_ref * F.mse_loss(v_pred, v_ref, reduction="mean")
    return fm_loss


OBJECTIVES = ("reweight_real", "self_distill")


class RLFlowMatchingLightningModule(pl.LightningModule):
    """GRPO-style reward-weighted Flow Matching update.

    Reuses the model build, solver, and validation utilities of the base trainer.

    Two objectives are available:

    `reweight_real` (the original conference/v2 behaviour) regresses towards the
    *real* image and uses the reward only to reweight which (noise, real image)
    pairs receive gradient. The optimum of that objective is still the data
    distribution, so the reward cannot actually move the sample distribution --
    it mostly injects heteroscedastic noise into ordinary FM training, which is
    why the v2 run degraded both FID and downstream accuracy.

    `self_distill` regresses towards the model's own high-reward rollouts
    (reward-weighted self-training / best-of-N distillation), so the reward
    shapes what the generator produces. `real_anchor_weight` mixes in a plain FM
    loss on real images and `ref_reg_weight` anchors the velocity field to the
    frozen base model; both guard against the collapse that pure self-training
    can cause.
    """

    def __init__(
        self,
        config: Dict,
        solver_config: Dict,
        group_size: int = 4,
        reward_temperature: float = 0.5,
        adv_alpha: Optional[float] = None,
        weight_clip: Optional[float] = 10.0,
        ref_reg_weight: float = 0.05,
        model: Optional[torch.nn.Module] = None,
        objective: str = "reweight_real",
        reward_topk: int = 0,
        reward_floor: Optional[float] = None,
        real_anchor_weight: float = 0.0,
    ) -> None:
        super().__init__()
        resolve_mask_conditioning_config(config)

        if objective not in OBJECTIVES:
            raise ValueError(f"Unknown objective={objective!r}. Choose from {OBJECTIVES}.")

        self.save_hyperparameters(
            {
                "train_args": config.get("train_args", {}),
                "model_args": config.get("model_args", {}),
                "data_args": config.get("data_args", {}),
                "solver_args": solver_config,
                "group_size": group_size,
                "reward_temperature": reward_temperature,
                "adv_alpha": adv_alpha,
                "weight_clip": weight_clip,
                "ref_reg_weight": ref_reg_weight,
                "objective": objective,
                "reward_topk": reward_topk,
                "reward_floor": reward_floor,
                "real_anchor_weight": real_anchor_weight,
            }
        )

        self.model = model if model is not None else build_model(config["model_args"])
        self.mask_conditioning = bool(config["model_args"].get("mask_conditioning", False))
        self.class_conditioning = bool(config["model_args"].get("with_conditioning", False))
        self.path = AffineProbPath(scheduler=CondOTScheduler())

        # Frozen teacher copy used as the KL-proxy anchor in _adv_weighted_flow_matching_loss.
        self.ref_model = None
        if ref_reg_weight and ref_reg_weight > 0.0:
            self.ref_model = copy.deepcopy(self.model)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

        self._solver_config = solver_config

    def configure_optimizers(self):
        return optim.Adam(
            self.model.parameters(), lr=self.hparams.get("train_args", {}).get("lr", 1e-4)
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        # GRPO step: replicate each input G times, sample candidates, score them,
        # then take a reward-weighted FM step.
        G = int(self.hparams["group_size"]) or 1
        batch_rep = _repeat_batch(batch, repeats=G)

        x0s = torch.randn_like(batch_rep["images"])
        cond = (
            batch_rep["classes"].unsqueeze(1)
            if (self.class_conditioning and ("classes" in batch_rep))
            else None
        )
        masks = batch_rep["masks"] if (self.mask_conditioning and ("masks" in batch_rep)) else None

        # Sample candidates without grad. Use trainer-precision-aware autocast on CUDA.
        # Match both "16-mixed" and "bf16-mixed" (the previous "16" in prec check missed bf16).
        prec = str(getattr(self.trainer, "precision", ""))
        use_amp = (("16" in prec) or ("bf16" in prec)) and batch_rep["images"].is_cuda
        with torch.no_grad(), autocast_for(batch_rep["images"], force=use_amp):
            final_imgs = sample_with_solver(
                model=self.model,
                x_init=x0s,
                solver_config=self._solver_config,
                cond=cond,
                masks=masks,
                return_intermediates=False,
            )

        reward_imgs = image_to_display_range(final_imgs)
        rewards, reward_metrics = compute_rewards(
            images=reward_imgs,
            masks=masks,
            classes=batch_rep.get("classes"),
            batch=batch_rep,
            group_size=G,
        )
        rewards = rewards.to(device=batch_rep["images"].device, dtype=torch.float32).view(-1)
        if rewards.numel() != batch_rep["images"].shape[0]:
            raise ValueError(
                "compute_rewards must return one scalar per generated candidate: "
                f"expected {batch_rep['images'].shape[0]}, got {rewards.numel()}."
            )
        objective = str(self.hparams.get("objective", "reweight_real"))
        # Clamped to the model's image range so self-distillation cannot teach the
        # generator to produce values the real data never contains.
        x1_target = final_imgs.detach().clamp(-1.0, 1.0) if objective == "self_distill" else None
        del final_imgs, reward_imgs
        if batch_rep["images"].is_cuda:
            torch.cuda.empty_cache()

        top_k = int(self.hparams.get("reward_topk", 0) or 0)
        if top_k > 0:
            weights = _group_topk_weights(
                rewards,
                group_size=G,
                top_k=top_k,
                reward_floor=self.hparams.get("reward_floor"),
            )
        else:
            weights = _group_softmax_weights(
                rewards,
                group_size=G,
                temperature=float(self.hparams["reward_temperature"]),
                adv_alpha=self.hparams.get("adv_alpha"),
                weight_clip=self.hparams.get("weight_clip"),
            )

        lam_ref = float(self.hparams.get("ref_reg_weight", 0.0) or 0.0)
        loss = _adv_weighted_flow_matching_loss(
            model=self.model,
            path=self.path,
            batch_rep=batch_rep,
            x0s=x0s,
            weights=weights,
            class_conditioning=self.class_conditioning,
            mask_conditioning=self.mask_conditioning,
            ref_model=self.ref_model,
            lam_ref=lam_ref,
            x1_target=x1_target,
        )

        anchor_weight = float(self.hparams.get("real_anchor_weight", 0.0) or 0.0)
        if x1_target is not None and anchor_weight > 0.0:
            # Unweighted FM loss on the real images: keeps the policy tied to the
            # data distribution while the reward pulls it towards high-reward modes.
            # Computed on the *un-replicated* batch: the anchor only needs the real
            # data distribution, and replicating it G times would hold a second
            # activation graph of the same size as the main loss (which OOMs a 24 GB
            # card at G=8 on this 496M-parameter model).
            anchor_loss = _adv_weighted_flow_matching_loss(
                model=self.model,
                path=self.path,
                batch_rep=batch,
                x0s=torch.randn_like(batch["images"]),
                weights=torch.ones(batch["images"].shape[0], device=batch["images"].device),
                class_conditioning=self.class_conditioning,
                mask_conditioning=self.mask_conditioning,
            )
            loss = (1.0 - anchor_weight) * loss + anchor_weight * anchor_loss
            self.log("train/anchor_loss", anchor_loss, on_step=True, on_epoch=True)

        B = batch["images"].shape[0]
        r = rewards.view(B, G)
        w = weights.view(B, G)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/reward_mean", r.mean(dim=1).mean(), on_step=True, on_epoch=True)
        self.log("train/reward_max", r.max(dim=1).values.mean(), on_step=True, on_epoch=True)
        self.log("train/weight_mean", w.mean(dim=1).mean(), on_step=True, on_epoch=True)
        self.log("train/weight_max", w.max(dim=1).values.mean(), on_step=True, on_epoch=True)
        if reward_metrics:
            for key, value in reward_metrics.items():
                if torch.is_tensor(value):
                    self.log(f"train/reward_{key}", value.mean(), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss = compute_fm_mse_loss(
            self.model, self.path, batch, self.mask_conditioning, self.class_conditioning
        )
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        save_validation_outputs(
            self,
            mask_conditioning=self.mask_conditioning,
            class_conditioning=self.class_conditioning,
            label="RL",
        )


def train_grpo_like(
    config_path: str,
    model_path: Optional[str] = None,
    num_inference_steps: Optional[int] = None,
    group_size: Optional[int] = None,
    reward_temperature: Optional[float] = None,
    max_updates: Optional[int] = None,
    lr: Optional[float] = None,
    adv_alpha: Optional[float] = None,
    weight_clip: Optional[float] = None,
    ref_reg_weight: Optional[float] = None,
    grad_clip_norm: Optional[float] = None,
    batch_size: Optional[int] = None,
    ckpt_every_n_steps: Optional[int] = None,
    objective: Optional[str] = None,
    reward_topk: Optional[int] = None,
    reward_floor: Optional[float] = None,
    real_anchor_weight: Optional[float] = None,
    checkpoint_dir: Optional[str] = None,
) -> None:
    """GRPO-style Flow Matching training with PyTorch Lightning."""
    config = load_config(config_path)
    base_config = dict(config)
    base_config["train_args"] = dict(config["fm_train_args"])
    config["train_args"] = dict(config["rl_train_args"])
    resolve_mask_conditioning_config(config)
    tr_args = config["train_args"]

    overrides = {
        "num_inference_steps": num_inference_steps,
        "batch_size": batch_size,
        "group_size": group_size,
        "reward_temperature": reward_temperature,
        "max_updates": max_updates,
        "lr": lr,
        "adv_alpha": adv_alpha,
        "weight_clip": weight_clip,
        "ref_reg_weight": ref_reg_weight,
        "grad_clip_norm": grad_clip_norm,
        "ckpt_every_n_steps": ckpt_every_n_steps,
        "objective": objective,
        "reward_topk": reward_topk,
        "reward_floor": reward_floor,
        "real_anchor_weight": real_anchor_weight,
        "checkpoint_dir": checkpoint_dir,
    }
    tr_args.update({k: v for k, v in overrides.items() if v is not None})

    run_name = os.path.splitext(os.path.basename(config_path))[0] + "-rl"
    root_ckpt_dir = os.path.abspath(
        os.path.expanduser(tr_args.get("checkpoint_dir", "checkpoints"))
    )
    resume_ckpt = resolve_resume_checkpoint(root_ckpt_dir, run_name)

    if resume_ckpt:
        init_model = None
        print(f"Resuming RL training from checkpoint: {resume_ckpt}")
    else:
        ckpt_path, _ = resolve_checkpoint_path(model_path, base_config, config_path)
        init_model, _ = load_model_from_checkpoint(
            ckpt_path, base_config, device=torch.device("cpu")
        )
        init_model.train()
        print(f"Initialized RL module from checkpoint: {ckpt_path}")

    solver_config = build_solver_config(config, tr_args.get("num_inference_steps"))
    print(
        f"Solver for sampling: method={solver_config['method']} "
        f"step_size={solver_config['step_size']} time_points={solver_config['time_points']}"
    )

    datamodule = FlowMatchingDataModule(config)
    rl_module = RLFlowMatchingLightningModule(
        config=config,
        solver_config=solver_config,
        group_size=tr_args.get("group_size", 8),
        reward_temperature=tr_args.get("reward_temperature", 0.5),
        adv_alpha=tr_args.get("adv_alpha"),
        weight_clip=tr_args.get("weight_clip"),
        ref_reg_weight=tr_args.get("ref_reg_weight", 0.05),
        model=init_model,
        objective=tr_args.get("objective", "reweight_real"),
        reward_topk=tr_args.get("reward_topk", 0),
        reward_floor=tr_args.get("reward_floor"),
        real_anchor_weight=tr_args.get("real_anchor_weight", 0.0),
    )
    print(
        f"RL objective={tr_args.get('objective', 'reweight_real')} "
        f"group_size={tr_args.get('group_size')} reward_topk={tr_args.get('reward_topk', 0)} "
        f"real_anchor_weight={tr_args.get('real_anchor_weight', 0.0)} "
        f"ref_reg_weight={tr_args.get('ref_reg_weight')}"
    )

    logger = TensorBoardLogger(save_dir=root_ckpt_dir, name=run_name)
    step_interval = int(tr_args.get("ckpt_every_n_steps") or 0)
    ckpt_every = max(1, int(tr_args.get("val_freq", 5)))
    if step_interval > 0:
        ckpt_cb = ModelCheckpoint(
            dirpath=os.path.join(root_ckpt_dir, run_name),
            filename="step{step:06d}",
            save_top_k=-1,
            save_last=True,
            auto_insert_metric_name=False,
            every_n_train_steps=step_interval,
        )
        print(f"Saving a checkpoint every {step_interval} optimizer steps (plus last.ckpt).")
    else:
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

    bf16_supported = (
        torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    default_precision = (
        "bf16-mixed" if bf16_supported else ("16-mixed" if torch.cuda.is_available() else "32-true")
    )
    precision = tr_args.get("precision", default_precision)

    trainer = pl.Trainer(
        default_root_dir=root_ckpt_dir,
        max_epochs=tr_args.get("num_epochs", 100),
        max_steps=tr_args.get("max_updates"),
        precision=precision,
        accumulate_grad_batches=tr_args.get("gradient_accumulation_steps", 8),
        gradient_clip_val=tr_args.get("grad_clip_norm"),
        check_val_every_n_epoch=ckpt_every,
        enable_progress_bar=True,
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        accelerator=tr_args.get("accelerator", "auto"),
        devices=tr_args.get("devices", "auto"),
        strategy=DDPStrategy(find_unused_parameters=True),
        deterministic=tr_args.get("deterministic", False),
        log_every_n_steps=tr_args.get("log_every_n_steps", 50),
        num_sanity_val_steps=tr_args.get("num_sanity_val_steps", 0),
    )

    trainer.fit(rl_module, datamodule=datamodule, ckpt_path=resume_ckpt)


def main():
    parser = argparse.ArgumentParser(
        description="GRPO-style training for Flow Matching with configurable rewards."
    )
    parser.add_argument("--config_path", type=str, default="configs/mask_conditioning.yaml")
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to Lightning .ckpt file or dir"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=None, help="Override solver time_points"
    )
    parser.add_argument(
        "--group_size", type=int, default=None, help="Candidates per input (GRPO group size)"
    )
    parser.add_argument(
        "--reward_temperature",
        type=float,
        default=None,
        help="Temperature; alpha=1/T if --adv_alpha unset",
    )
    parser.add_argument("--max_updates", type=int, default=None, help="Number of optimizer updates")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override training batch size for this RL run"
    )
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--adv_alpha",
        type=float,
        default=None,
        help="Advantage scale (overrides 1/temperature if set)",
    )
    parser.add_argument(
        "--weight_clip",
        type=float,
        default=None,
        help="Clip weights to [1/clip, clip] (then re-normalize)",
    )
    parser.add_argument(
        "--ref_reg_weight",
        type=float,
        default=None,
        help="Teacher proximity (KL-proxy) weight; 0 disables",
    )
    parser.add_argument(
        "--grad_clip_norm", type=float, default=None, help="Gradient clipping (global norm)"
    )
    parser.add_argument(
        "--ckpt_every_n_steps",
        type=int,
        default=None,
        help="Save a checkpoint every N optimizer steps (in addition to last.ckpt).",
    )
    parser.add_argument(
        "--objective",
        choices=OBJECTIVES,
        default=None,
        help="reweight_real = reward only reweights real-image FM targets (v1/v2 behaviour); "
             "self_distill = regress towards the model's own high-reward rollouts.",
    )
    parser.add_argument(
        "--reward_topk",
        type=int,
        default=None,
        help="Keep only the top-K candidates per group (hard best-of-N). 0 = soft exp weights.",
    )
    parser.add_argument(
        "--reward_floor",
        type=float,
        default=None,
        help="Discard accepted candidates whose absolute reward is below this value.",
    )
    parser.add_argument(
        "--real_anchor_weight",
        type=float,
        default=None,
        help="Mix an unweighted FM loss on real images into the self-distillation loss.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Override rl_train_args.checkpoint_dir (keeps throwaway runs out of the real dir).",
    )
    args = parser.parse_args()
    train_grpo_like(**vars(args))


if __name__ == "__main__":
    main()
