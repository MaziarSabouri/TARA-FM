"""Shared building blocks for the downstream classifiers used to:

  - train the real-data-only classifier that also serves as the frozen RL reward,
  - train every augmentation scenario (real / synthetic / mixed training pools),
  - run the multi-architecture generalization and TSTR studies.

Kept independent of the Flow Matching / Lightning stack (no `flow_matching`,
`monai_generative`, or `pytorch_lightning` imports) so it runs in the
torch + torchvision + scikit-learn environment used for data preparation.

Journal-version additions over the conference code:
  - any backbone from `utils.backbones` (not just ResNet18),
  - per-backbone input resizing (128 px for the modified-stem ResNet18,
    224 px for stock ImageNet stems),
  - mixed-precision training,
  - calibration (ECE / Brier) and balanced-accuracy / MCC / kappa metrics,
  - deterministic seeding helpers for the multi-seed robustness study.
"""

import os
import pickle
import random
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from utils.backbones import (  # noqa: F401  (re-exported for callers)
    IMAGENET_MEAN,
    IMAGENET_STD,
    SIMPLE_MEAN,
    SIMPLE_STD,
    build_backbone,
    get_backbone_spec,
    normalization_for,
    resolve_backbone_name,
    resolve_input_size,
)

CLASS_NAMES_DEFAULT = ["Normal", "Diffuse Goiter", "Nodular Goiter", "Thyroiditis"]


def sample_id(sample: Dict) -> str:
    """Stable patient/scan id used by the nested reward vs. downstream split."""
    for key in ("name", "folder_id", "folder"):
        value = sample.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return f"anon_{id(sample)}"


def filter_by_ids(samples: List[Dict], allowed_ids) -> List[Dict]:
    allowed = set(allowed_ids)
    return [s for s in samples if sample_id(s) in allowed]


###############################################################################
# Reproducibility
###############################################################################
def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python/numpy/torch. `deterministic` also pins cuDNN algorithm choice,
    which makes multi-seed runs exactly reproducible at some speed cost."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def _worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


###############################################################################
# Data loading
###############################################################################
def load_pickle_samples(pickle_path: str, split: Optional[str] = None) -> List[Dict]:
    """Load a list of sample dicts from a prepare_dataset.py-style pickle.

    If `split` is given, returns `data[split]`. Otherwise, if the pickle
    contains exactly one list-valued split (besides "class_names"), that
    split is used (this matches inferer.py's `{"train": [...]}` output for
    generated/synthetic samples).
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        raise TypeError(f"Unsupported pickle contents in {pickle_path}: {type(data)}")

    if split is not None:
        if split not in data:
            raise KeyError(f"Split '{split}' not found in {pickle_path}. Keys: {list(data.keys())}")
        return list(data[split])

    list_splits = [k for k, v in data.items() if isinstance(v, list)]
    if len(list_splits) != 1:
        raise ValueError(
            f"Ambiguous pickle {pickle_path}: pass --split explicitly. "
            f"Available list-valued keys: {list_splits}"
        )
    return list(data[list_splits[0]])


def get_class_names(pickle_path: str) -> List[str]:
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "class_names" in data:
        return list(data["class_names"])
    return list(CLASS_NAMES_DEFAULT)


def stratified_split(
    samples: List[Dict], val_frac: float, seed: int
) -> Tuple[List[Dict], List[Dict]]:
    """Per-class shuffled split so both splits keep the original class balance."""
    by_class: Dict[int, List[Dict]] = {}
    for s in samples:
        by_class.setdefault(int(s["class"]), []).append(s)

    rng = random.Random(seed)
    train, valid = [], []
    for _, group in sorted(by_class.items()):
        group = group[:]
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_frac)) if len(group) > 1 else 0
        valid.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def class_distribution(samples: List[Dict], class_names: Sequence[str]) -> str:
    counts = Counter(int(s["class"]) for s in samples)
    return ", ".join(f"{name}={counts.get(i, 0)}" for i, name in enumerate(class_names))


def class_counts(samples: List[Dict], num_classes: int) -> List[int]:
    counts = Counter(int(s["class"]) for s in samples)
    return [int(counts.get(i, 0)) for i in range(num_classes)]


###############################################################################
# Dataset
###############################################################################
def _as_chw_float(image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    return torch.from_numpy(arr).clamp(0.0, 1.0)


def _apply_conventional_augment(
    image: torch.Tensor,
    *,
    rotation_degrees: float = 15.0,
    translate_frac: float = 0.10,
    scale_range: Tuple[float, float] = (0.8, 1.2),
    noise_sigma_range: Tuple[float, float] = (0.001, 0.01),
) -> torch.Tensor:
    """Randomized transform pipeline: hflip, rotation (+-15 deg), translation
    (+-10%), scaling (0.8-1.2x), and additive Gaussian noise (sigma in
    [0.001, 0.01]) -- the "conventional augmentation" (CA) protocol.
    """
    if random.random() < 0.5:
        image = TF.hflip(image)

    _, h, w = image.shape
    angle = random.uniform(-rotation_degrees, rotation_degrees)
    max_dx = translate_frac * w
    max_dy = translate_frac * h
    translate = (random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy))
    scale = random.uniform(scale_range[0], scale_range[1])
    image = TF.affine(image, angle=angle, translate=translate, scale=scale, shear=0.0, fill=0.0)

    sigma = random.uniform(noise_sigma_range[0], noise_sigma_range[1])
    if sigma > 0:
        image = image + torch.randn_like(image) * sigma
        image = image.clamp(0.0, 1.0)
    return image


def resize_to(image: torch.Tensor, size: Optional[int]) -> torch.Tensor:
    """Resize a [C,H,W] tensor to size x size (no-op when already correct).

    Antialiasing is enabled for downsampling; upsampling 128 -> 224 for the
    stock-stem backbones uses plain bilinear interpolation.
    """
    if size is None or (image.shape[-1] == size and image.shape[-2] == size):
        return image
    antialias = image.shape[-1] > size
    return F.interpolate(
        image.unsqueeze(0), size=(size, size), mode="bilinear",
        align_corners=False, antialias=antialias,
    ).squeeze(0)


class ThyroidClassificationDataset(Dataset):
    """Wraps a list of {"image": [C,H,W] in [0,1], "class": int} sample dicts.

    Images are repeated to 3 channels (grayscale -> pseudo-RGB) so standard
    torchvision backbones can be used with ImageNet weights.

    `augment`:
      - "none":         no augmentation.
      - "conventional": hflip + rotation + translation + scaling + Gaussian noise.

    A per-sample `apply_augment=False` flag opts individual samples out of CA,
    which is how the "CA on real images only" scenarios are implemented.
    """

    def __init__(
        self,
        samples: List[Dict],
        augment: str = "none",
        normalize_mean: Sequence[float] = SIMPLE_MEAN,
        normalize_std: Sequence[float] = SIMPLE_STD,
        input_size: Optional[int] = None,
    ) -> None:
        if augment not in ("none", "conventional"):
            raise ValueError(f"Unknown augment mode: {augment!r}. Use 'none' or 'conventional'.")
        self.samples = samples
        self.augment = augment
        self.input_size = input_size
        self.mean = torch.tensor(normalize_mean).view(3, 1, 1)
        self.std = torch.tensor(normalize_std).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        image = _as_chw_float(sample["image"])  # [1,H,W] in [0,1]

        # Augment at native resolution, then resize: keeps the geometric
        # transform identical across backbones with different input sizes.
        if self.augment == "conventional" and sample.get("apply_augment", True):
            image = _apply_conventional_augment(image)

        image = resize_to(image, self.input_size)
        image = image.repeat(3, 1, 1)  # pseudo-RGB
        image = (image - self.mean) / self.std
        return image, int(sample["class"])


def make_dataloader(
    samples: List[Dict],
    batch_size: int,
    augment,
    shuffle: bool,
    num_workers: int = 4,
    normalize_mean: Sequence[float] = SIMPLE_MEAN,
    normalize_std: Sequence[float] = SIMPLE_STD,
    input_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    """`augment`: "none" | "conventional" (bool accepted for backward
    compatibility: True -> "conventional", False -> "none")."""
    if isinstance(augment, bool):
        augment = "conventional" if augment else "none"
    dataset = ThyroidClassificationDataset(
        samples,
        augment=augment,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        input_size=input_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=_worker_init_fn,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


###############################################################################
# Model
###############################################################################
def build_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Conference-paper ResNet18 (3x3/stride-1 stem, Dropout(0.2) head).

    Thin wrapper kept so existing call sites keep working; new code should call
    `build_backbone(name, ...)` directly.
    """
    return build_backbone("resnet18", num_classes=num_classes, pretrained=pretrained)


def save_classifier_checkpoint(
    path: str,
    model: nn.Module,
    class_names: Sequence[str],
    pretrained_norm: bool,
    backbone: str = "resnet18",
    input_size: Optional[int] = None,
    extra: Optional[Dict] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": list(class_names),
            "pretrained_norm": bool(pretrained_norm),
            "backbone": resolve_backbone_name(backbone),
            "input_size": int(input_size) if input_size else get_backbone_spec(backbone).input_size,
            "extra": dict(extra or {}),
        },
        path,
    )


def load_classifier_bundle(path: str, device: torch.device) -> Dict:
    """Load a checkpoint into a dict with the model plus everything needed to
    preprocess inputs the same way training did."""
    ckpt = torch.load(path, map_location=device, weights_only=False)  # locally-produced checkpoint
    class_names = list(ckpt.get("class_names", CLASS_NAMES_DEFAULT))
    pretrained_norm = bool(ckpt.get("pretrained_norm", False))
    backbone = resolve_backbone_name(ckpt.get("backbone", "resnet18"))
    input_size = int(ckpt.get("input_size") or get_backbone_spec(backbone).input_size)

    model = build_backbone(backbone, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    mean, std = normalization_for(pretrained_norm)
    return {
        "model": model,
        "class_names": class_names,
        "pretrained_norm": pretrained_norm,
        "backbone": backbone,
        "spec": get_backbone_spec(backbone),
        "input_size": input_size,
        "mean": mean,
        "std": std,
        "extra": ckpt.get("extra", {}),
        "path": path,
    }


def load_classifier_checkpoint(path: str, device: torch.device) -> Tuple[nn.Module, List[str], bool]:
    """Backward-compatible 3-tuple loader used by the reward hook and Grad-CAM."""
    bundle = load_classifier_bundle(path, device)
    return bundle["model"], bundle["class_names"], bundle["pretrained_norm"]


def to_model_input(
    image,
    mean: Sequence[float],
    std: Sequence[float],
    input_size: Optional[int] = None,
) -> torch.Tensor:
    """Single-image preprocessing mirroring ThyroidClassificationDataset (no augmentation)."""
    t = _as_chw_float(image)
    t = resize_to(t, input_size)
    t = t.repeat(3, 1, 1)
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    return (t - mean_t) / std_t


###############################################################################
# Train / evaluate
###############################################################################
@torch.no_grad()
def _compute_val_loss(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = F.cross_entropy(logits.float(), labels, reduction="sum")
        total_loss += loss.item()
        n += images.size(0)
    return total_loss / max(1, n)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    checkpoint_path: str,
    class_names: Sequence[str],
    pretrained_norm: bool,
    weight_decay: float = 1e-5,
    patience: Optional[int] = None,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.8,
    min_lr: float = 2e-5,
    amp: bool = False,
    backbone: str = "resnet18",
    input_size: Optional[int] = None,
    class_weights: Optional[torch.Tensor] = None,
    grad_clip: Optional[float] = None,
    log_every: int = 1,
) -> Dict:
    """Train with Adam + ReduceLROnPlateau on validation loss.

    Best checkpoint is selected by lowest validation loss. Optional early
    stopping (`patience`) is off by default so a full `epochs` run completes.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience, min_lr=min_lr
    )
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    weight_t = class_weights.to(device) if class_weights is not None else None

    best_val_loss = float("inf")
    best_val_f1 = -1.0
    best_epoch = -1
    epochs_since_improve = 0
    history = []

    for epoch in range(epochs):
        model.train()
        running_loss, running_correct, running_n = 0.0, 0, 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = F.cross_entropy(logits.float(), labels, weight=weight_t)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_n += images.size(0)

        train_loss = running_loss / max(1, running_n)
        train_acc = running_correct / max(1, running_n)

        val_loss = _compute_val_loss(model, val_loader, device, use_amp)
        val_metrics = evaluate_classifier(
            model, val_loader, device, class_names, compute_auroc=False, amp=use_amp
        )
        val_f1 = val_metrics["macro_f1"]
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if log_every and ((epoch + 1) % log_every == 0 or epoch == 0):
            print(
                f"[epoch {epoch + 1}/{epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_macro_f1={val_f1:.4f} lr={current_lr:.2e}",
                flush=True,
            )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "lr": current_lr,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_f1,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            epochs_since_improve = 0
            save_classifier_checkpoint(
                checkpoint_path, model, class_names, pretrained_norm,
                backbone=backbone, input_size=input_size,
                extra={"best_epoch": best_epoch, "best_val_loss": best_val_loss},
            )
        else:
            epochs_since_improve += 1
            if patience is not None and epochs_since_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1} (no val_loss improvement for {patience} epochs).")
                break

    print(
        f"Best val_loss={best_val_loss:.4f} (val_macro_f1={best_val_f1:.4f}) "
        f"at epoch {best_epoch}. Saved to {checkpoint_path}"
    )
    return {
        "best_val_loss": best_val_loss,
        "best_val_macro_f1": best_val_f1,
        "best_epoch": best_epoch,
        "history": history,
    }


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> Tuple[float, List[Dict]]:
    """Top-label ECE plus the per-bin data needed for a reliability diagram."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_rows = []
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        count = int(in_bin.sum())
        if count == 0:
            bin_rows.append({"bin_lower": float(lo), "bin_upper": float(hi), "count": 0,
                             "confidence": None, "accuracy": None})
            continue
        avg_conf = float(confidences[in_bin].mean())
        avg_acc = float(accuracies[in_bin].mean())
        ece += (count / n) * abs(avg_acc - avg_conf)
        bin_rows.append({"bin_lower": float(lo), "bin_upper": float(hi), "count": count,
                         "confidence": avg_conf, "accuracy": avg_acc})
    return float(ece), bin_rows


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
    compute_auroc: bool = True,
    return_raw: bool = False,
    amp: bool = False,
) -> Dict:
    """Run inference and compute the full metric suite:

    accuracy, balanced accuracy, MCC, Cohen's kappa, per-class
    precision/recall/specificity/F1 (and one-vs-rest AUROC / average
    precision), micro/macro/weighted averages, confusion matrix, and
    calibration (ECE, Brier).
    """
    model.eval()
    use_amp = bool(amp and device.type == "cuda")
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=use_amp):
            logits = model(images)
        probs = F.softmax(logits.float(), dim=1)
        all_probs.append(probs.cpu())
        all_labels.append(labels)

    probs = torch.cat(all_probs, dim=0).numpy().astype(np.float64)
    labels = torch.cat(all_labels, dim=0).numpy()
    metrics = metrics_from_probs(probs, labels, class_names, compute_auroc=compute_auroc)
    if return_raw:
        metrics["_raw_probs"] = probs
        metrics["_raw_labels"] = labels
    return metrics


def apply_logit_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """p'_c ∝ p_c * exp(b_c). Equivalent to adding `bias` to the logits."""
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    shifted = logp + np.asarray(bias, dtype=np.float64).reshape(1, -1)
    shifted -= shifted.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def inverse_prior_bias(train_counts: Sequence[float]) -> np.ndarray:
    """Closed-form bias that maps the training prior onto a uniform decision prior."""
    freq = np.asarray(train_counts, dtype=np.float64)
    freq = freq / max(freq.sum(), 1.0)
    uniform = 1.0 / max(len(freq), 1)
    return np.log(uniform) - np.log(np.clip(freq, 1e-8, 1.0))


def _macro_f1(labels: np.ndarray, preds: np.ndarray, n_classes: int) -> float:
    from sklearn.metrics import f1_score

    return float(
        f1_score(labels, preds, labels=list(range(n_classes)), average="macro", zero_division=0)
    )


def tune_logit_bias(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    n_classes: int,
    seed: int = 0,
    n_restarts: int = 8,
) -> Tuple[np.ndarray, float]:
    """Maximize validation macro-F1 over an additive, sum-zero logit bias.

    Macro-F1 is not differentiable through argmax, so this is a derivative-free
    search (Nelder-Mead, several random restarts). The last class is determined
    by the sum-zero constraint so the search is (K-1)-dimensional.
    """
    from scipy.optimize import minimize

    rng = np.random.RandomState(seed)

    def pack(x: np.ndarray) -> np.ndarray:
        return np.concatenate([x, [-float(x.sum())]])

    def objective(x: np.ndarray) -> float:
        preds = apply_logit_bias(val_probs, pack(x)).argmax(axis=1)
        return -_macro_f1(val_labels, preds, n_classes)

    best = np.zeros(n_classes, dtype=np.float64)
    best_f1 = _macro_f1(val_labels, val_probs.argmax(axis=1), n_classes)
    for _ in range(n_restarts):
        x0 = rng.normal(0.0, 0.4, size=n_classes - 1)
        result = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 250, "xatol": 1e-3, "fatol": 1e-4})
        bias = pack(np.asarray(result.x, dtype=np.float64))
        f1 = _macro_f1(val_labels, apply_logit_bias(val_probs, bias).argmax(axis=1), n_classes)
        if f1 > best_f1:
            best_f1, best = f1, bias
    return best, best_f1


def metrics_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    compute_auroc: bool = True,
    preds: Optional[np.ndarray] = None,
) -> Dict:
    """Full metric suite from softmax probabilities and integer labels."""
    preds = probs.argmax(axis=1) if preds is None else np.asarray(preds)
    n_classes = len(class_names)
    class_idx = list(range(n_classes))

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        matthews_corrcoef,
        precision_recall_fscore_support,
    )

    accuracy = float(accuracy_score(labels, preds))
    cm = confusion_matrix(labels, preds, labels=class_idx)

    p_pc, r_pc, f1_pc, support_pc = precision_recall_fscore_support(
        labels, preds, labels=class_idx, average=None, zero_division=0
    )

    # Specificity (true-negative rate), one-vs-rest -- reported alongside recall
    # because it is the clinically meaningful complement for rare classes.
    specificity_pc = []
    total = cm.sum()
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        specificity_pc.append(float(tn / max(1, tn + fp)))

    def _avg(average: str):
        p, r, f1, _ = precision_recall_fscore_support(
            labels, preds, labels=class_idx, average=average, zero_division=0
        )
        return float(p), float(r), float(f1)

    p_micro, r_micro, f1_micro = _avg("micro")
    p_macro, r_macro, f1_macro = _avg("macro")
    p_weighted, r_weighted, f1_weighted = _avg("weighted")

    labels_onehot = np.eye(n_classes, dtype=np.float64)[labels]
    brier = float(np.mean(np.sum((probs - labels_onehot) ** 2, axis=1)))
    ece, ece_bins = expected_calibration_error(probs, labels)

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "cohen_kappa": float(cohen_kappa_score(labels, preds)),
        "brier_score": brier,
        "ece": ece,
        "ece_bins": ece_bins,
        "n_samples": int(len(labels)),
        "confusion_matrix": cm.tolist(),
        "precision_per_class": {class_names[i]: float(p_pc[i]) for i in range(n_classes)},
        "recall_per_class": {class_names[i]: float(r_pc[i]) for i in range(n_classes)},
        "specificity_per_class": {class_names[i]: specificity_pc[i] for i in range(n_classes)},
        "f1_per_class": {class_names[i]: float(f1_pc[i]) for i in range(n_classes)},
        "support_per_class": {class_names[i]: int(support_pc[i]) for i in range(n_classes)},
        "micro_precision": p_micro, "micro_recall": r_micro, "micro_f1": f1_micro,
        "macro_precision": p_macro, "macro_recall": r_macro, "macro_f1": f1_macro,
        "weighted_precision": p_weighted, "weighted_recall": r_weighted, "weighted_f1": f1_weighted,
    }

    if compute_auroc:
        from sklearn.metrics import average_precision_score, roc_auc_score

        auroc_per_class, auprc_per_class = {}, {}
        aurocs, weights = [], []
        for i in range(n_classes):
            y_true_i = labels_onehot[:, i]
            if y_true_i.sum() == 0 or y_true_i.sum() == len(y_true_i):
                continue  # AUROC undefined without both classes present
            auc = float(roc_auc_score(y_true_i, probs[:, i]))
            auroc_per_class[class_names[i]] = auc
            auprc_per_class[class_names[i]] = float(average_precision_score(y_true_i, probs[:, i]))
            aurocs.append(auc)
            weights.append(int(support_pc[i]))
        metrics["auroc_per_class"] = auroc_per_class
        metrics["auprc_per_class"] = auprc_per_class
        metrics["macro_auroc"] = float(np.mean(aurocs)) if aurocs else None
        metrics["weighted_auroc"] = float(np.average(aurocs, weights=weights)) if aurocs else None
        metrics["macro_auprc"] = (
            float(np.mean(list(auprc_per_class.values()))) if auprc_per_class else None
        )
        try:
            metrics["micro_auroc"] = float(roc_auc_score(labels_onehot.ravel(), probs.ravel()))
        except ValueError:
            metrics["micro_auroc"] = None

    return metrics


def print_metrics(title: str, metrics: Dict, class_names: Sequence[str]) -> None:
    print(f"\n=== {title} ===")
    print(f"n_samples: {metrics['n_samples']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}   Balanced acc: {metrics.get('balanced_accuracy', float('nan')):.4f}")
    print(f"MCC: {metrics.get('mcc', float('nan')):.4f}   Kappa: {metrics.get('cohen_kappa', float('nan')):.4f}   "
          f"ECE: {metrics.get('ece', float('nan')):.4f}   Brier: {metrics.get('brier_score', float('nan')):.4f}")
    print(f"Micro    P/R/F1: {metrics['micro_precision']:.4f}/{metrics['micro_recall']:.4f}/{metrics['micro_f1']:.4f}")
    print(f"Macro    P/R/F1: {metrics['macro_precision']:.4f}/{metrics['macro_recall']:.4f}/{metrics['macro_f1']:.4f}")
    print(f"Weighted P/R/F1: {metrics['weighted_precision']:.4f}/{metrics['weighted_recall']:.4f}/{metrics['weighted_f1']:.4f}")
    if metrics.get("macro_auroc") is not None:
        print(
            f"AUROC micro/macro/weighted: {metrics.get('micro_auroc'):.4f}/"
            f"{metrics['macro_auroc']:.4f}/{metrics.get('weighted_auroc'):.4f}"
        )
    print("Per-class P/R/Spec/F1(/AUROC):")
    for name in class_names:
        p = metrics["precision_per_class"].get(name)
        r = metrics["recall_per_class"].get(name)
        sp = metrics.get("specificity_per_class", {}).get(name)
        f1 = metrics["f1_per_class"].get(name)
        auc = metrics.get("auroc_per_class", {}).get(name)
        auc_str = f", AUROC={auc:.4f}" if auc is not None else ""
        sp_str = f", Spec={sp:.4f}" if sp is not None else ""
        print(f"  {name}: P={p:.4f}, R={r:.4f}{sp_str}, F1={f1:.4f}{auc_str}")
