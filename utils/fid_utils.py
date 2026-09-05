import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.general_utils import image_to_display_range

# Name of the cached InceptionV3 weights used for FID (matches torchvision default)
_FID_INCEPTION_FILENAME = "inception_v3_google-0cc3c7bd.pth"
_FID_ENV_VAR = "FID_INCEPTION_WEIGHTS"

# Process-wide cache so we allocate the (large) InceptionV3 feature extractor on the
# GPU once and reuse it, instead of re-instantiating it every validation epoch. Repeatedly
# allocating/dropping a full model on top of the training tensors fragments the CUDA
# caching allocator over long runs and can eventually lead to OOM/driver instability
# (this is what was causing crashes during `mask_and_class_conditioning` training).
_INCEPTION_CACHE: Dict[tuple, torch.nn.Module] = {}


def _default_inception_cache_path() -> Path:
    """
    Resolve the default cache path where torchvision stores downloaded checkpoints.
    """
    try:
        hub_dir = Path(torch.hub.get_dir())
    except Exception:
        hub_dir = Path.home() / ".cache" / "torch"
    return hub_dir / "checkpoints" / _FID_INCEPTION_FILENAME


def _load_inception_state_dict(weights_path: Optional[str] = None) -> Optional[dict]:
    """
    Try to load a state dict for InceptionV3 from a provided path, environment
    variable, or the default torch hub cache. Returns None if nothing is found.
    """
    if weights_path:
        candidate = Path(weights_path).expanduser()
    elif os.environ.get(_FID_ENV_VAR):
        candidate = Path(os.environ[_FID_ENV_VAR]).expanduser()
    else:
        candidate = _default_inception_cache_path()

    if candidate.is_file():
        return torch.load(candidate, map_location="cpu")
    return None


def _prepare_for_inception(x: torch.Tensor) -> torch.Tensor:
    """
    Prepare images for InceptionV3 feature extraction.
    - Clamp to [0,1]
    - Convert to 3 channels if grayscale by replicating channels
    - Resize to 299x299 using bilinear
    """
    x = x.clamp(0.0, 1.0)
    if x.dim() != 4:
        raise ValueError(f"Expected images of shape [N,C,H,W], got {tuple(x.shape)}")
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    return x


def _get_inception_feature_extractor(
    device: torch.device, weights_path: Optional[str] = None
) -> torch.nn.Module:
    """
    Returns a (cached) InceptionV3 feature extractor that outputs 2048-d pool features.

    Cached per (device, weights_path) so repeated calls (e.g. once or twice per
    validation epoch) reuse the same GPU-resident model instead of allocating a new
    one every time.
    """
    cache_key = (str(device), weights_path or "")
    cached = _INCEPTION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Delayed import to avoid requiring torchvision at module import time in some environments
    from torchvision.models import inception_v3
    state_dict = _load_inception_state_dict(weights_path)

    if state_dict is not None:
        # Use cached weights to avoid network calls on offline nodes
        model = inception_v3(weights=None, aux_logits=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[WARN] Loaded InceptionV3 with missing keys={missing}, unexpected={unexpected}")
    else:
        # Fallback to torchvision-managed weights (may download if cache is empty)
        try:
            from torchvision.models import Inception_V3_Weights

            weights = Inception_V3_Weights.IMAGENET1K_V1
            # Some torchvision versions require aux_logits=True when loading weights
            model = inception_v3(weights=weights, aux_logits=True)
        except Exception:
            # Fallback if weights enum not available
            model = inception_v3(pretrained=True, aux_logits=True)
    # Replace the classification head with identity to expose the pooled features
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    _INCEPTION_CACHE[cache_key] = model
    return model


@torch.no_grad()
def _inception_features(model: torch.nn.Module, x: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
    feats = []
    N = x.shape[0]
    for i in range(0, N, batch_size):
        xb = x[i : i + batch_size]
        fb = model(xb)
        if fb.dim() > 2:
            fb = torch.flatten(fb, 1)
        feats.append(fb.cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def inception_features_chunked(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """Pool features for images that may not fit resized in memory.

    `_prepare_for_inception` upsamples 128px images to 299px, a 5.5x blow-up: a
    4000-image pool becomes 4.3 GB in one tensor, which is enough to exhaust the
    GPU (and, on a 23 GB host, the machine) when a whole selection pool is scored
    at once. Resizing chunk by chunk keeps the peak at `batch_size` images.
    """
    feats = []
    for start in range(0, images.shape[0], batch_size):
        chunk = _prepare_for_inception(images[start : start + batch_size].to(device))
        out = model(chunk)
        if out.dim() > 2:
            out = torch.flatten(out, 1)
        feats.append(out.float().cpu())
        del chunk, out
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(feats, dim=0)


def _symmetric_matrix_sqrt(m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the symmetric square root of a positive semi-definite matrix via eigh.
    """
    m = m.clone()
    # Stabilize
    m = 0.5 * (m + m.t())
    eigvals, eigvecs = torch.linalg.eigh(m)
    eigvals = torch.clamp(eigvals, min=eps)
    return (eigvecs * eigvals.sqrt().unsqueeze(0)) @ eigvecs.t()


def _frechet_distance(mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor) -> float:
    """
    Compute Frechet distance between two Gaussians parameterized by means and covariances.
    Uses numerically stable symmetric sqrt: Tr(S1 + S2 - 2*sqrt(S1^{1/2} S2 S1^{1/2}))
    """
    mu1 = mu1.double()
    mu2 = mu2.double()
    sigma1 = sigma1.double()
    sigma2 = sigma2.double()

    diff = (mu1 - mu2)
    diff_sq = diff.dot(diff)

    s1_sqrt = _symmetric_matrix_sqrt(sigma1)
    middle = s1_sqrt @ sigma2 @ s1_sqrt
    middle_sqrt = _symmetric_matrix_sqrt(middle)

    trace = torch.trace(sigma1 + sigma2 - 2.0 * middle_sqrt)
    fid = (diff_sq + trace).item()
    # Guard against small negative due to numerical error
    return float(max(fid, 0.0))


def _polynomial_mmd2(x: torch.Tensor, y: torch.Tensor, degree: int = 3, gamma: Optional[float] = None, coef0: float = 1.0) -> float:
    """Unbiased MMD^2 estimate with a polynomial kernel, i.e. the Kernel
    Inception Distance (KID) as defined in Binkowski et al. 2018 (using
    Inception pool features for `x`/`y` instead of raw pixels).

    `x`: [Nx, D] real features, `y`: [Ny, D] generated features.
    """
    x = x.double()
    y = y.double()
    d = x.shape[1]
    if gamma is None:
        gamma = 1.0 / d

    def kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (gamma * (a @ b.t()) + coef0) ** degree

    kxx = kernel(x, x)
    kyy = kernel(y, y)
    kxy = kernel(x, y)

    m = x.shape[0]
    n = y.shape[0]
    # Unbiased estimator: exclude the diagonal (self-similarity) terms.
    sum_kxx = (kxx.sum() - kxx.diagonal().sum()) / (m * (m - 1))
    sum_kyy = (kyy.sum() - kyy.diagonal().sum()) / (n * (n - 1))
    sum_kxy = kxy.sum() / (m * n)

    mmd2 = sum_kxx + sum_kyy - 2.0 * sum_kxy
    return float(mmd2.item())


def compute_kid_from_features(
    real_feats: torch.Tensor,
    gen_feats: torch.Tensor,
    n_subsets: int = 10,
    subset_size: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, float]:
    """Kernel Inception Distance between two feature sets, averaged over
    `n_subsets` random subsets (the standard KID protocol, since the
    unbiased MMD^2 estimator has high variance on a single subset).

    Returns {"kid_mean": ..., "kid_std": ...}. Both sets must have >= 2
    samples (and `subset_size` <= min(Nx, Ny) if given).
    """
    n = min(real_feats.shape[0], gen_feats.shape[0])
    if subset_size is None:
        subset_size = min(n, 1000)
    subset_size = max(2, min(subset_size, n))

    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_subsets):
        idx_r = rng.choice(real_feats.shape[0], size=subset_size, replace=False)
        idx_g = rng.choice(gen_feats.shape[0], size=subset_size, replace=False)
        vals.append(_polynomial_mmd2(real_feats[idx_r], gen_feats[idx_g]))
    vals = np.asarray(vals, dtype=np.float64)
    return {"kid_mean": float(vals.mean()), "kid_std": float(vals.std())}


def compute_fid_from_features(real_feats: torch.Tensor, gen_feats: torch.Tensor) -> float:
    """FID between two feature sets (Frechet distance between fitted Gaussians)."""
    real_feats = real_feats.double()
    gen_feats = gen_feats.double()
    mu_r, mu_g = real_feats.mean(dim=0), gen_feats.mean(dim=0)
    cr, cg = torch.cov(real_feats.T), torch.cov(gen_feats.T)
    return _frechet_distance(mu_r, cr, mu_g, cg)


@torch.no_grad()
def compute_fid_kid_between_image_sets(
    real_images: torch.Tensor,
    gen_images: torch.Tensor,
    device: torch.device,
    inception_weights_path: Optional[str] = None,
    kid_n_subsets: int = 10,
    kid_subset_size: Optional[int] = None,
    seed: int = 0,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Offline FID + KID between two raw image sets (e.g. pickled real vs.
    generated samples), independent of any live generative model/solver.

    `real_images`/`gen_images`: [N,1,H,W] or [N,3,H,W] tensors, values in
    [0,1] (min-max normalized, matching this project's pickle format).
    """
    inc = _get_inception_feature_extractor(device, weights_path=inception_weights_path)
    real_feats = inception_features_chunked(inc, real_images, device, batch_size)
    gen_feats = inception_features_chunked(inc, gen_images, device, batch_size)

    fid = compute_fid_from_features(real_feats, gen_feats)
    kid = compute_kid_from_features(
        real_feats, gen_feats, n_subsets=kid_n_subsets, subset_size=kid_subset_size, seed=seed
    )
    return {"fid": fid, "n_real": int(real_feats.shape[0]), "n_gen": int(gen_feats.shape[0]), **kid}


def _extract_class_indices(classes: torch.Tensor) -> torch.Tensor:
    """
    Convert one-hot or integer class labels to a flat tensor of class indices.
    """
    if classes.dim() == 1:
        return classes.long()
    if classes.dim() >= 2:
        return classes.argmax(dim=1).long()
    raise ValueError(f"Unsupported class tensor shape: {tuple(classes.shape)}")


def _prepare_class_condition(
    cond: torch.Tensor, num_classes: Optional[int] = None
) -> torch.Tensor:
    """
    Ensure conditioning vectors are float and have shape [B, 1, C] for the model.
    If `cond` is [B], convert to one-hot with an inferred or provided class count.
    """
    cond = cond.to(torch.float32)
    if cond.dim() == 1:
        n_cls = num_classes or (int(cond.max().item()) + 1)
        cond = torch.nn.functional.one_hot(cond.long(), num_classes=n_cls).to(torch.float32)
    if cond.dim() == 2:
        cond = cond.unsqueeze(1)
    return cond


@torch.no_grad()
def compute_fid_small_sample(
    gen_model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    solver_config: dict,
    max_samples: int = 16,
    mask_conditioning: bool = True,
    class_conditioning: bool = False,
    inception_weights_path: Optional[str] = None,
):
    """
    Compute a small-sample FID between real images in the validation loader
    and images generated by `gen_model` using the provided solver settings.

    Optionally supply `inception_weights_path` (or set env FID_INCEPTION_WEIGHTS)
    to force using a locally cached InceptionV3 checkpoint and avoid downloads.

    Returns a float FID (lower is better) or None if not enough samples.
    """
    # Delayed import to avoid circulars
    from utils.utils_fm import sample_with_solver

    inc = _get_inception_feature_extractor(device, weights_path=inception_weights_path)

    real_feats = []
    gen_feats = []
    processed = 0

    for batch in val_loader:
        if processed >= max_samples:
            break
        imgs = batch["images"].to(device)
        if imgs.dim() == 5:
            # Skip 3D volumes for FID
            continue
        B = imgs.size(0)
        take = min(B, max_samples - processed)
        imgs = imgs[:take]

        # Real features
        real_in = _prepare_for_inception(image_to_display_range(imgs))
        rf = _inception_features(inc, real_in)
        real_feats.append(rf)

        # Generate
        cond = batch.get("classes") if class_conditioning and ("classes" in batch) else None
        masks = batch.get("masks") if mask_conditioning and ("masks" in batch) else None
        if cond is not None:
            cond = cond[:take].to(device).unsqueeze(1)
        if masks is not None:
            masks = masks[:take].to(device)

        x_init = torch.randn_like(imgs[:take])
        gen = sample_with_solver(
            model=gen_model,
            x_init=x_init,
            solver_config=solver_config,
            cond=cond,
            masks=masks,
            return_intermediates=False,
        )
        gen = image_to_display_range(gen)

        gen_in = _prepare_for_inception(gen)
        gf = _inception_features(inc, gen_in)
        gen_feats.append(gf)

        processed += take

    if processed < 2:
        return None

    real_feats = torch.cat(real_feats, dim=0).double()
    gen_feats = torch.cat(gen_feats, dim=0).double()

    mu_r = real_feats.mean(dim=0)
    mu_g = gen_feats.mean(dim=0)
    # Use unbiased estimator (N-1), safe-guard for N=1 handled above
    cr = torch.cov(real_feats.T)
    cg = torch.cov(gen_feats.T)

    fid = _frechet_distance(mu_r, cr, mu_g, cg)
    return fid


def log_fid_metrics(
    module,
    val_loader: DataLoader,
    solver_args: dict,
    tr: dict,
    *,
    mask_conditioning: bool,
    class_conditioning: bool,
    label: str = "val",
) -> None:
    """Compute and log small-sample FID (and per-class FID if class-conditioned).

    Logs are silently skipped on failure (e.g. missing torchvision weights).
    """
    try:
        fid_n = int(tr.get("fid_num_samples", tr.get("num_val_samples", 16)))
        fid_weights = tr.get("fid_inception_weights_path") or os.environ.get(_FID_ENV_VAR)
        fid_val = compute_fid_small_sample(
            gen_model=module.model,
            val_loader=val_loader,
            device=module.device,
            solver_config=solver_args,
            max_samples=fid_n,
            mask_conditioning=mask_conditioning,
            class_conditioning=class_conditioning,
            inception_weights_path=fid_weights,
        )
        if fid_val is not None:
            module.log("val/FID", float(fid_val), prog_bar=True, on_epoch=True)
        if class_conditioning:
            fid_per_class = compute_fid_per_class_small_sample(
                gen_model=module.model,
                val_loader=val_loader,
                device=module.device,
                solver_config=solver_args,
                max_samples_per_class=int(tr.get("fid_num_samples_per_class", fid_n)),
                mask_conditioning=mask_conditioning,
                class_conditioning=class_conditioning,
                inception_weights_path=fid_weights,
            )
            for cls_idx, fid_cls in fid_per_class.items():
                module.log(f"val/FID_class_{cls_idx}", float(fid_cls), on_epoch=True)
    except Exception as e:
        suffix = f" ({label})" if label and label != "val" else ""
        print(f"[WARN] FID computation failed{suffix}: {e}")


@torch.no_grad()
def compute_fid_per_class_small_sample(
    gen_model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    solver_config: dict,
    max_samples_per_class: int = 16,
    mask_conditioning: bool = True,
    class_conditioning: bool = True,
    inception_weights_path: Optional[str] = None,
) -> Dict[int, float]:
    """
    Compute per-class small-sample FID using the provided validation loader.
    Returns a dict mapping class index -> FID. Classes with <2 samples are skipped.
    """
    from utils.utils_fm import sample_with_solver

    inc = _get_inception_feature_extractor(device, weights_path=inception_weights_path)

    real_feats = defaultdict(list)
    gen_feats = defaultdict(list)
    processed = defaultdict(int)
    num_classes_hint: Optional[int] = None

    for batch in val_loader:
        if "classes" not in batch:
            break

        imgs = batch["images"].to(device)
        if imgs.dim() == 5:
            continue

        classes = batch["classes"]
        if not torch.is_tensor(classes):
            classes = torch.as_tensor(classes)
        classes = classes.to(device)

        class_idx = _extract_class_indices(classes)
        current_max = (int(class_idx.max().item()) + 1) if class_idx.numel() > 0 else 0
        if num_classes_hint is None:
            if classes.dim() >= 2 and classes.shape[-1] > 1:
                num_classes_hint = int(classes.shape[-1])
            else:
                num_classes_hint = current_max
        num_classes_hint = max(num_classes_hint or 0, current_max)
        unique_classes = class_idx.unique()

        for cls in unique_classes.tolist():
            remaining = max_samples_per_class - processed[cls]
            if remaining <= 0:
                continue
            mask = (class_idx == cls)
            imgs_cls = imgs[mask][:remaining]
            if imgs_cls.numel() == 0:
                continue

            real_in = _prepare_for_inception(image_to_display_range(imgs_cls))
            real_feats[cls].append(_inception_features(inc, real_in))

            cond = None
            if class_conditioning:
                cond = _prepare_class_condition(classes[mask][:remaining], num_classes=num_classes_hint)
            masks = batch.get("masks") if (mask_conditioning and ("masks" in batch)) else None
            if masks is not None:
                masks = masks.to(device)[mask][:remaining]

            x_init = torch.randn_like(imgs_cls)
            gen = sample_with_solver(
                model=gen_model,
                x_init=x_init,
                solver_config=solver_config,
                cond=cond,
                masks=masks,
                return_intermediates=False,
            )
            gen = image_to_display_range(gen)

            gen_in = _prepare_for_inception(gen)
            gen_feats[cls].append(_inception_features(inc, gen_in))
            processed[cls] += imgs_cls.size(0)

    fid_per_class: Dict[int, float] = {}
    for cls, rf_list in real_feats.items():
        if processed[cls] < 2 or (cls not in gen_feats):
            continue
        rf = torch.cat(rf_list, dim=0).double()
        gf = torch.cat(gen_feats[cls], dim=0).double()
        if rf.shape[0] < 2 or gf.shape[0] < 2:
            continue
        mu_r = rf.mean(dim=0)
        mu_g = gf.mean(dim=0)
        cr = torch.cov(rf.T)
        cg = torch.cov(gf.T)
        fid_per_class[cls] = _frechet_distance(mu_r, cr, mu_g, cg)

    return fid_per_class
