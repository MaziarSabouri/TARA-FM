import json
import os
import yaml
import torch
import pickle
from typing import Dict, Optional, Sequence, Union

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset


###############################################################################
# Config Handling
###############################################################################
def load_config(config_path: str = "config.yaml"):
    """
    Loads a YAML config file from the given path.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


###############################################################################
# Data Loading & Preparation
###############################################################################
class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def __len__(self):
        return len(next(iter(self.data_dict.values())))

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.data_dict.items()}


def _scale_image_to_model_range(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0).mul(2.0).sub(1.0)


def image_to_display_range(tensor: torch.Tensor) -> torch.Tensor:
    """Convert normalized model images back to [0, 1] for saving, FID, or reward models."""
    return tensor.add(1.0).mul(0.5).clamp(0.0, 1.0)


def mask_to_display_range(mask: torch.Tensor) -> torch.Tensor:
    """Convert scalar or one-hot masks to a single [0, 1] image for visualization."""
    if mask.dim() >= 3 and mask.shape[-3] > 1:
        labels = mask.argmax(dim=-3, keepdim=True).to(torch.float32)
        return labels.div(float(mask.shape[-3] - 1)).clamp(0.0, 1.0)
    return mask.clamp(0.0, 1.0)


def _as_chw_tensor(value, key: str, sample_idx: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    if tensor.dim() != 3:
        raise ValueError(
            f"Sample {sample_idx} '{key}' must have shape [H,W] or [C,H,W]; "
            f"got {tuple(tensor.shape)}."
        )
    return tensor


def _as_label_hw(value, key: str, sample_idx: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 2:
        raise ValueError(
            f"Sample {sample_idx} '{key}' must be a 2D class-index map for one-hot encoding; "
            f"got {tuple(tensor.shape)}."
        )
    return tensor


def _onehot_chw(label_hw: torch.Tensor, num_classes: int, sample_idx: int) -> torch.Tensor:
    lo, hi = int(label_hw.min()), int(label_hw.max())
    if lo < 0 or hi >= num_classes:
        raise ValueError(
            f"Sample {sample_idx} 'mask' has label values outside [0, {num_classes-1}]: "
            f"min={lo}, max={hi}."
        )
    return (
        torch.nn.functional.one_hot(label_hw, num_classes=num_classes)
        .permute(2, 0, 1)
        .to(torch.float32)
    )


def _is_auto_mask_num_classes(value) -> bool:
    return isinstance(value, str) and value.lower() in {"auto", "infer", "inferred"}


def _sample_id(sample: dict) -> str:
    for key in ("name", "folder_id", "folder"):
        value = sample.get(key)
        if value is not None and str(value) != "":
            return str(value)
    raise KeyError(
        "Sample has no name/folder_id/folder key; cannot apply a nested split filter."
    )


def resolve_allowed_ids(data_config: dict) -> Optional[Sequence[str]]:
    """Return patient ids for ``data_args.split_role`` from a nested-split manifest.

    Used so the Flow Matching generator (and RL rollouts / sampling masks) never
    see the reward-pool patients. ``None`` means "use the full pickle split".
    """
    manifest = data_config.get("split_manifest")
    role = data_config.get("split_role")
    if not manifest:
        return None
    if not role:
        raise ValueError("data_args.split_role is required when split_manifest is set")
    with open(manifest) as f:
        payload = json.load(f)
    key = f"{role}_ids"
    if key not in payload:
        raise KeyError(f"Manifest {manifest} has no '{key}'")
    ids = list(payload[key])
    print(f"[split] {manifest} role={role}: {len(ids)} patient ids")
    return ids


def _get_split_samples(data_dict: Dict, split: str):
    if split not in data_dict:
        raise ValueError(f"No data found for split '{split}' in the pickle file.")
    data_split = data_dict[split]
    if not isinstance(data_split, (list, tuple)):
        raise TypeError(f"Split '{split}' must be a list of sample dicts.")
    if not data_split:
        raise ValueError(f"No data found for split '{split}' in the pickle file.")
    return data_split


def _mask_num_classes_from_samples(data_split) -> int:
    max_label = None
    for idx, sample in enumerate(data_split):
        if not isinstance(sample, dict):
            raise TypeError(f"Sample {idx} must be a dict.")
        if "mask" not in sample:
            raise KeyError(f"Sample {idx} is missing key: 'mask'")

        label_hw = _as_label_hw(sample["mask"], "mask", idx)
        lo, hi = int(label_hw.min()), int(label_hw.max())
        if lo < 0:
            raise ValueError(f"Sample {idx} 'mask' has negative label values: min={lo}.")
        max_label = hi if max_label is None else max(max_label, hi)

    if max_label is None:
        raise ValueError("Cannot infer mask classes from an empty split.")
    return max_label + 1


def infer_mask_num_classes(
    pickle_path: str,
    splits: Union[str, Sequence[str]],
) -> int:
    """Infer mask label count as max class index + 1 across one or more splits."""
    if isinstance(splits, str):
        split_names = [splits]
    else:
        split_names = list(splits)
    split_names = list(dict.fromkeys(split for split in split_names if split))
    if not split_names:
        raise ValueError("At least one split is required to infer mask classes.")

    with open(pickle_path, "rb") as f:
        data_dict = pickle.load(f)
    if not isinstance(data_dict, dict):
        raise TypeError("Dataset pickle must contain a dict of splits.")

    max_classes = 0
    for split in split_names:
        split_classes = _mask_num_classes_from_samples(_get_split_samples(data_dict, split))
        max_classes = max(max_classes, split_classes)
    return max_classes


def resolve_mask_conditioning_config(config: Dict) -> Optional[int]:
    """
    Fill mask one-hot and ControlNet input-channel counts from dataset labels.

    This keeps YAML configs from hard-coding the mask class count while preserving
    consistency between data loading and ControlNet construction.
    """
    model_args = config.get("model_args") or {}
    if not bool(model_args.get("mask_conditioning", False)):
        return None

    data_args = config.get("data_args") or {}
    if not data_args.get("pickle_path"):
        raise ValueError("data_args.pickle_path is required to infer mask classes.")

    raw_num_classes = data_args.get("mask_num_classes")
    if raw_num_classes is None or _is_auto_mask_num_classes(raw_num_classes):
        split_names = [
            data_args.get("split_train"),
            data_args.get("split_val"),
        ]
        mask_num_classes = infer_mask_num_classes(data_args["pickle_path"], split_names)
        data_args["mask_num_classes"] = mask_num_classes
    else:
        mask_num_classes = int(raw_num_classes)
        data_args["mask_num_classes"] = mask_num_classes

    raw_in_channels = model_args.get("conditioning_embedding_in_channels")
    if raw_in_channels is None or _is_auto_mask_num_classes(raw_in_channels):
        model_args["conditioning_embedding_in_channels"] = mask_num_classes
    elif int(raw_in_channels) != mask_num_classes:
        raise ValueError(
            "model_args.conditioning_embedding_in_channels "
            f"({raw_in_channels}) does not match inferred mask_num_classes "
            f"({mask_num_classes})."
        )

    return mask_num_classes


def load_and_prepare_data(
    pickle_path: str,
    split: str = "train",
    convert_classes_to_onehot: bool = False,
    mask_num_classes: Optional[Union[int, str]] = None,
    allowed_ids: Optional[Sequence[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Loads a dataset split from a pickle file and prepares tensors for training.

    Expects the pickle to contain `data_dict[split]` -> list of sample dicts.
    Each sample needs "image" and "mask" arrays shaped [C,H,W].
    Optional "class" labels are one-hot encoded if requested.

    If `mask_num_classes` is set to an int or "auto", masks are treated as 2D
    integer class-index maps and one-hot encoded to [N, mask_num_classes, H, W].

    Returns a dict with:
      images: float tensor [N, C, H, W]
      masks: float tensor [N, C, H, W]
      classes (optional): one-hot float tensor [N, num_classes] if requested
      class_map (optional): index -> class label map
    """
    # Load the pickle file
    with open(pickle_path, "rb") as f:
        data_dict = pickle.load(f)
    if not isinstance(data_dict, dict):
        raise TypeError("Dataset pickle must contain a dict of splits.")

    # Extract data for the specified split
    data_split = list(_get_split_samples(data_dict, split))
    if allowed_ids is not None:
        allowed = set(allowed_ids)
        before = len(data_split)
        data_split = [s for s in data_split if _sample_id(s) in allowed]
        print(f"[split] pickle split '{split}': {before} -> {len(data_split)} images")
        if not data_split:
            raise ValueError(
                f"No samples left in split '{split}' after applying allowed_ids "
                f"({len(allowed)} ids)."
            )

    if _is_auto_mask_num_classes(mask_num_classes):
        mask_num_classes = _mask_num_classes_from_samples(data_split)
    if mask_num_classes is not None:
        mask_num_classes = int(mask_num_classes)
    onehot_masks = mask_num_classes is not None and mask_num_classes > 1

    # Assemble tensors while validating the serialized sample structure.
    imgs = []
    mks = []
    for idx, sample in enumerate(data_split):
        if not isinstance(sample, dict):
            raise TypeError(f"Sample {idx} in split '{split}' must be a dict.")
        missing = {"image", "mask"} - set(sample)
        if missing:
            raise KeyError(f"Sample {idx} in split '{split}' is missing keys: {sorted(missing)}")

        image = _as_chw_tensor(sample["image"], "image", idx)
        if onehot_masks:
            mask = _onehot_chw(_as_label_hw(sample["mask"], "mask", idx), mask_num_classes, idx)
        else:
            mask = _as_chw_tensor(sample["mask"], "mask", idx)
        if image.shape[1:] != mask.shape[1:]:
            raise ValueError(
                f"Sample {idx} image/mask spatial shape mismatch: "
                f"image={tuple(image.shape)}, mask={tuple(mask.shape)}."
            )
        if imgs and image.shape != imgs[0].shape:
            raise ValueError(
                f"Sample {idx} image shape {tuple(image.shape)} does not match "
                f"first image shape {tuple(imgs[0].shape)} in split '{split}'."
            )
        if mks and mask.shape != mks[0].shape:
            raise ValueError(
                f"Sample {idx} mask shape {tuple(mask.shape)} does not match "
                f"first mask shape {tuple(mks[0].shape)} in split '{split}'."
            )
        imgs.append(image)
        mks.append(mask)

    Images = torch.stack(imgs, dim=0)  # [N, C, H, W]
    Images = _scale_image_to_model_range(Images)

    Masks = torch.stack(mks, dim=0)  # [N, C, H, W]

    result = {"images": Images, "masks": Masks}

    class_flags = ["class" in e for e in data_split]
    if any(class_flags) and not all(class_flags):
        raise ValueError(f"Split '{split}' must either include 'class' for every sample or none.")
    has_class = all(class_flags)
    if has_class:
        class_list = [e["class"] for e in data_split]
        if convert_classes_to_onehot:
            all_classes = sorted(set(class_list))
            class_to_idx = {c: i for i, c in enumerate(all_classes)}
            idx_to_class = {i: c for i, c in enumerate(all_classes)}
            idxs = torch.tensor([class_to_idx[c] for c in class_list], dtype=torch.long)
            onehot = torch.nn.functional.one_hot(idxs, num_classes=len(all_classes)).float()
            result["classes"] = onehot
            result["class_map"] = idx_to_class
        else:
            result["classes"] = class_list

    return result


def create_dataloader(
    Images: torch.Tensor,
    Masks: Optional[torch.Tensor] = None,
    classes: Optional[torch.Tensor] = None,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    drop_last: bool = False,
):
    """
    Creates a performant DataLoader from tensors, optionally including masks/classes.

    Additional knobs:
      - num_workers: dataloader workers
      - pin_memory: defaults to True on CUDA machines
      - persistent_workers: defaults to True if num_workers > 0 and not shuffle-only epoch
    """
    data_dict = {"images": Images}
    if Masks is not None:
        data_dict["masks"] = Masks
    if classes is not None:
        data_dict["classes"] = classes

    dataset = CustomDataset(data_dict)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )


###############################################################################
# Image Saving
###############################################################################
def save_image(img_tensor, out_path, vmin=None, vmax=None):
    """
    Saves a single 2D image (assumed shape [1, H, W] or [H, W]) as PNG.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # remove batch/channel dims if present
    if img_tensor.dim() == 3 and img_tensor.shape[0] == 1:
        img_tensor = img_tensor.squeeze(0)
    plt.figure()
    plt.imshow(img_tensor.cpu().numpy(), cmap="gray", vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_image_3d(img_tensor, out_path, slice_idx=None, vmin=None, vmax=None):
    """
    Saves a single 3D image (assumed shape [1, D, H, W] or [D, H, W]) as a series of PNGs.
    If slice_idx is provided, saves only that slice.
    """
    os.makedirs(out_path, exist_ok=True)
    # remove batch/channel dims if present
    if img_tensor.dim() == 4 and img_tensor.shape[0] == 1:
        img_tensor = img_tensor.squeeze(0)
    if img_tensor.dim() != 3:
        raise ValueError("img_tensor must be 3D (D, H, W)")

    D = img_tensor.shape[0]
    slice_indices = [slice_idx] if slice_idx is not None else [D // 2]

    for i in slice_indices:
        plt.figure()
        plt.imshow(img_tensor[i].cpu().numpy(), cmap="gray", vmin=vmin, vmax=vmax)
        plt.axis("off")
        plt.savefig(os.path.join(out_path, f"slice_{i:03d}.png"), bbox_inches="tight", pad_inches=0)
        plt.close()
