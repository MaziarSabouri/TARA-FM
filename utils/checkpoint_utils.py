import os
from typing import Dict, Optional, Tuple

import torch


def select_checkpoint_file(ckpt_dir: str) -> Optional[str]:
    """Pick `last.ckpt` if present, else most recent `.ckpt` in `ckpt_dir`."""
    if not os.path.isdir(ckpt_dir):
        return None

    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.isfile(last_ckpt):
        return last_ckpt

    candidates = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def resolve_checkpoint_path(
    model_path: Optional[str], config: Dict, config_path: str
) -> Tuple[str, str]:
    """Resolve user-provided checkpoint path or infer from config/run name."""
    run_name = os.path.splitext(os.path.basename(config_path))[0]
    root_dir = os.path.abspath(os.path.expanduser(config["train_args"]["checkpoint_dir"]))
    candidate = (
        os.path.abspath(os.path.expanduser(model_path)) if model_path
        else os.path.join(root_dir, run_name)
    )

    if os.path.isfile(candidate) and candidate.endswith(".ckpt"):
        return candidate, os.path.dirname(candidate)
    if os.path.isdir(candidate):
        resolved = select_checkpoint_file(candidate)
        if resolved:
            return resolved, candidate

    raise FileNotFoundError(f"Unable to locate a checkpoint at: {candidate}")


def _state_dict_for_fm_module(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Keep only FlowMatchingLightningModule weights from an FM or RL checkpoint.

    RL checkpoints also store a frozen `ref_model.*` copy used as a KL anchor during
    fine-tuning. Inference only needs the trainable policy under `model.*`.
    """
    # Drop the frozen reference copy if present (RL ckpts).
    filtered = {k: v for k, v in state_dict.items() if not k.startswith("ref_model.")}
    dropped = len(state_dict) - len(filtered)
    if dropped:
        print(f"[INFO] Dropped {dropped} ref_model.* tensors from RL checkpoint (not needed for inference).")
    return filtered


def load_model_from_checkpoint(
    checkpoint_path: str, config: Dict, device: torch.device
) -> Tuple[torch.nn.Module, Dict]:
    """Load a Lightning checkpoint and return the underlying model + metadata."""
    # Lazy import to avoid pulling Lightning module deps for callers that only need path resolution.
    from fm_lightning import FlowMatchingLightningModule

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    lightning_module = FlowMatchingLightningModule(config)
    state_dict = _state_dict_for_fm_module(checkpoint["state_dict"])
    missing, unexpected = lightning_module.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys when loading {checkpoint_path}: {unexpected[:10]}"
            + (" ..." if len(unexpected) > 10 else "")
        )
    if missing:
        print(f"[WARN] Missing keys when loading checkpoint ({len(missing)}): {missing[:5]}...")
    model = lightning_module.model.to(device)
    model.eval()
    return model, {
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "checkpoint_name": os.path.splitext(os.path.basename(checkpoint_path))[0],
    }


def resolve_resume_checkpoint(root_ckpt_dir: str, run_name: str) -> Optional[str]:
    """Return the run's last checkpoint path to resume from, if any."""
    root_ckpt_dir = os.path.abspath(os.path.expanduser(root_ckpt_dir))
    last_ckpt = os.path.join(root_ckpt_dir, run_name, "last.ckpt")
    return last_ckpt if os.path.isfile(last_ckpt) else None
