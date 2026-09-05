import contextlib
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from generative.networks.nets import DiffusionModelUNet, ControlNet
from flow_matching.solver import ODESolver

from .general_utils import (
    image_to_display_range,
    mask_to_display_range,
    save_image,
    save_image_3d,
)
from tqdm import tqdm


def autocast_for(reference: torch.Tensor, *, force: bool = True):
    """Return a torch.autocast context for `reference`'s device, else nullcontext."""
    if not force or not reference.is_cuda:
        return contextlib.nullcontext()
    bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16 if bf16_ok else torch.float16)


###############################################################################
# Model Building
###############################################################################
class MergedModel(nn.Module):
    """UNet wrapper with optional ControlNet for mask conditioning."""

    def __init__(self, unet: DiffusionModelUNet, controlnet: ControlNet = None, max_timestep=1000):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.max_timestep = max_timestep
        self.has_controlnet = controlnet is not None

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor = None,
        masks: torch.Tensor = None,
    ):
        # Continuous t in [0,1] -> discrete timestep index expected by the UNet embedding.
        t = (t * (self.max_timestep - 1)).floor().long()
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        if cond is not None and cond.dim() == 2:
            cond = cond.unsqueeze(1)

        if self.has_controlnet:
            # ControlNet runs in parallel and injects mask-derived residuals into the UNet skip path.
            down_res, mid_res = self.controlnet(
                x=x, timesteps=t, controlnet_cond=masks, context=cond
            )
            return self.unet(
                x=x,
                timesteps=t,
                context=cond,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
            )
        return self.unet(x=x, timesteps=t, context=cond)


def build_model(model_config: dict, device: torch.device = None) -> MergedModel:
    """Build a UNet (+ optional ControlNet) from `model_config`."""
    mc = model_config.copy()
    mask_conditioning = mc.pop("mask_conditioning", False)
    max_timestep = mc.pop("max_timestep", 1000)
    cond_embed_channels = mc.pop("conditioning_embedding_num_channels", None)
    cond_embed_in_channels = mc.pop("conditioning_embedding_in_channels", None)

    unet = DiffusionModelUNet(**mc)

    controlnet = None
    if mask_conditioning:
        mc.pop("out_channels", None)
        if cond_embed_channels is None:
            cond_embed_channels = (16, 32, 96, 256)
        cn_kwargs = {"conditioning_embedding_num_channels": cond_embed_channels}
        if cond_embed_in_channels is not None:
            cn_kwargs["conditioning_embedding_in_channels"] = cond_embed_in_channels
        controlnet = ControlNet(**mc, **cn_kwargs)
        # Warm-start ControlNet with UNet weights so training starts near a sensible velocity field.
        controlnet.load_state_dict(unet.state_dict(), strict=False)

    model = MergedModel(unet=unet, controlnet=controlnet, max_timestep=max_timestep)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters ({num_params * 4 / (1024**2):.2f} MB).")

    return model.to(device) if device is not None else model


def sample_with_solver(
    model,
    x_init,
    solver_config,
    cond=None,
    masks=None,
    return_intermediates: bool = False,
):
    """Run the ODE solver from `x_init` to t=1 and return either trajectory or final state."""
    solver = ODESolver(velocity_model=model)
    time_points = solver_config.get("time_points", 10)
    T = torch.linspace(0, 1, time_points, device=x_init.device)
    return solver.sample(
        time_grid=T,
        x_init=x_init,
        method=solver_config.get("method", "midpoint"),
        step_size=solver_config.get("step_size", 0.02),
        return_intermediates=bool(solver_config.get("return_intermediates", return_intermediates)),
        cond=cond,
        masks=masks,
    )


def compute_fm_mse_loss(
    model: torch.nn.Module,
    path,
    batch: dict,
    mask_conditioning: bool,
    class_conditioning: bool,
) -> torch.Tensor:
    """Standard flow-matching MSE loss against a random t along an AffineProbPath."""
    imgs = batch["images"]
    masks = batch.get("masks") if mask_conditioning else None
    classes = batch.get("classes") if class_conditioning else None

    # x_0: noise, x_1: real image. info.x_t is the interpolated point at time t,
    # info.dx_t is the target velocity the model must learn to predict.
    x_0 = torch.randn_like(imgs)
    t = torch.rand(imgs.shape[0], device=imgs.device)
    info = path.sample(t=t, x_0=x_0, x_1=imgs)
    v_pred = model(x=info.x_t, t=info.t, masks=masks, cond=classes)
    return F.mse_loss(v_pred, info.dx_t)


def resolve_log_dir(trainer) -> str:
    """Trainer logger log_dir, falling back to default_root_dir."""
    logger = getattr(trainer, "logger", None)
    if logger is not None and hasattr(logger, "log_dir") and logger.log_dir:
        return logger.log_dir
    return trainer.default_root_dir


def plot_solver_steps(sol, im_batch, mask_batch, outdir, max_plot=4):
    if sol.dim() != 5:  # No intermediates to plot
        return
    n_samples = min(sol.shape[1], max_plot)
    n_steps = sol.shape[0]
    n_cols = n_steps + (2 if mask_batch is not None else 1)
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(20, 8))
    if n_samples == 1:
        axes = [axes]
    for i in range(n_samples):
        for t in range(n_steps):
            step_img = image_to_display_range(sol[t, i].detach().float())
            axes[i][t].imshow(step_img.cpu().numpy().squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
            axes[i][t].axis("off")
            if i == 0:
                axes[i][t].set_title(f"Step {t}")
        col = n_steps
        if mask_batch is not None:
            mask_img = mask_to_display_range(mask_batch[i].detach().float())
            axes[i][col].imshow(mask_img.cpu().numpy().squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
            axes[i][col].axis("off")
            if i == 0:
                axes[i][col].set_title("Mask")
            col += 1
        real_img = image_to_display_range(im_batch[i].detach().float())
        axes[i][col].imshow(real_img.cpu().numpy().squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[i][col].axis("off")
        if i == 0:
            axes[i][col].set_title("Real")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "solver_steps.png"), bbox_inches="tight", pad_inches=0)
    plt.close()


def validate_and_save_samples(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    checkpoint_dir: str,
    epoch: int,
    solver_config: dict,
    max_samples=16,
    mask_conditioning=True,
    class_conditioning=False,
):
    model.eval()
    outdir = os.path.join(checkpoint_dir, f"val_samples_epoch_{epoch}")
    os.makedirs(outdir, exist_ok=True)
    count, step_plot_done = 0, False
    for batch in tqdm(val_loader, desc="Validating"):
        imgs = batch["images"].to(device)
        is_3d = imgs.dim() == 5
        cond = batch["classes"].to(device).unsqueeze(1) if class_conditioning else None
        masks = batch["masks"].to(device) if mask_conditioning else None

        x_init = torch.randn_like(imgs)
        sol = sample_with_solver(
            model, x_init, solver_config, cond=cond, masks=masks, return_intermediates=True
        )
        final_imgs = sol[-1] if sol.dim() == imgs.dim() + 1 else sol

        for i in range(final_imgs.size(0)):
            if count >= max_samples:
                break
            gen_img = image_to_display_range(final_imgs[i].detach().float())
            real_img = image_to_display_range(imgs[i].detach().float())
            sdir = os.path.join(outdir, f"sample_{count+1:03d}")
            os.makedirs(sdir, exist_ok=True)
            saver = save_image_3d if is_3d else save_image
            ext = "" if is_3d else ".png"
            saver(gen_img, os.path.join(sdir, f"gen{ext}"), vmin=0.0, vmax=1.0)
            saver(real_img, os.path.join(sdir, f"real{ext}"), vmin=0.0, vmax=1.0)
            if masks is not None:
                cnd_img = mask_to_display_range(masks[i].detach().float())
                saver(cnd_img, os.path.join(sdir, f"mask{ext}"), vmin=0.0, vmax=1.0)
            count += 1
        if not step_plot_done:
            plot_solver_steps(sol, imgs, masks, outdir)
            step_plot_done = True
        if count >= max_samples:
            break
    print(f"Validation samples saved in: {outdir}")


def save_validation_outputs(
    module,
    *,
    mask_conditioning: bool,
    class_conditioning: bool,
    label: str = "val",
) -> None:
    if hasattr(module.trainer, "is_global_zero") and not module.trainer.is_global_zero:
        return

    tr = module.hparams.get("train_args", {})
    solver_args = module.hparams.get("solver_args", {})
    val_loader = module.trainer.datamodule.val_dataloader()

    validate_and_save_samples(
        model=module.model,
        val_loader=val_loader,
        device=module.device,
        checkpoint_dir=resolve_log_dir(module.trainer),
        epoch=module.current_epoch,
        solver_config=solver_args,
        max_samples=tr.get("num_val_samples", 16),
        mask_conditioning=mask_conditioning,
        class_conditioning=class_conditioning,
    )

    from utils.fid_utils import log_fid_metrics

    log_fid_metrics(
        module,
        val_loader,
        solver_args,
        tr,
        mask_conditioning=mask_conditioning,
        class_conditioning=class_conditioning,
        label=label,
    )


@torch.no_grad()
def sample_batch(
    model: torch.nn.Module,
    solver_config: dict,
    batch: torch.Tensor,
    device: torch.device,
    class_conditioning: bool = False,
    mask_conditioning: bool = False,
):
    model.eval()
    imgs = batch["images"].to(device)
    cond = batch["classes"].to(device).unsqueeze(1) if class_conditioning else None
    masks = batch["masks"].to(device) if mask_conditioning else None

    x_init = torch.randn_like(imgs)
    return sample_with_solver(
        model=model,
        solver_config=solver_config,
        x_init=x_init,
        cond=cond,
        masks=masks,
        return_intermediates=False,
    )
