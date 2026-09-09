"""Classifier rewards used by the paper's RL fine-tuning experiments.

The frozen judge is an ensemble of ResNet-18 classifiers. Each member produces
softmax probabilities and the probabilities are averaged before applying one
of the three reported rewards:

    soft:    p_y
    focal:   (1 - p_y)^gamma * p_y
    utility: focal when argmax(p) == y, otherwise -eta

Set ``CLASSIFIER_REWARD_CKPT`` to a comma-separated list of judge checkpoints
and ``REWARD_TYPE`` to ``prob``, ``focal``, or ``utility``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from utils.reward_model import load_reward_classifier


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CHECKPOINT = os.path.join(
    _THIS_DIR, "checkpoints", "reward", "reward_resnet18_prob.pth"
)
REWARD_TYPES = ("prob", "focal", "utility")
_REWARD_MODEL = None


class ClassifierReward:
    """Frozen ResNet-18 judge with checkpoint-matched preprocessing."""

    def __init__(self, checkpoint_path: str, device: torch.device):
        bundle = load_reward_classifier(checkpoint_path, device)
        self.checkpoint_path = checkpoint_path
        self.model = bundle["model"]
        self.class_names = bundle["class_names"]
        self.input_size = bundle["input_size"]
        self.device = device
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.mean = torch.tensor(bundle["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(bundle["std"], device=device).view(1, 3, 1, 1)

    def to(self, device: torch.device) -> "ClassifierReward":
        if device != self.device:
            self.model.to(device)
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
            self.device = device
        return self

    @torch.no_grad()
    def predict_probs(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(device=self.device, dtype=torch.float32)
        if images.shape[1] != 3:
            images = images[:, :1].repeat(1, 3, 1, 1)
        if images.shape[-2:] != (self.input_size, self.input_size):
            downsampling = (
                images.shape[-2] > self.input_size
                or images.shape[-1] > self.input_size
            )
            images = F.interpolate(
                images,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
                antialias=downsampling,
            )
        images = (images - self.mean) / self.std
        return F.softmax(self.model(images).float(), dim=1)


class EnsembleClassifierReward:
    """Average softmax probabilities from the three frozen paper judges."""

    def __init__(self, checkpoint_paths: List[str], device: torch.device):
        self.checkpoint_path = ",".join(checkpoint_paths)
        self.members = [
            ClassifierReward(path, device=device) for path in checkpoint_paths
        ]
        class_names = self.members[0].class_names
        if any(member.class_names != class_names for member in self.members[1:]):
            raise ValueError("Reward ensemble members disagree on class ordering.")
        self.class_names = class_names
        self.input_size = self.members[0].input_size
        self.device = device

    def to(self, device: torch.device) -> "EnsembleClassifierReward":
        for member in self.members:
            member.to(device)
        self.device = device
        return self

    @torch.no_grad()
    def predict_probs(self, images: torch.Tensor) -> torch.Tensor:
        member_probs = [member.predict_probs(images) for member in self.members]
        return torch.stack(member_probs, dim=0).mean(dim=0)


def _reward_checkpoint_paths() -> List[str]:
    raw = os.environ.get("CLASSIFIER_REWARD_CKPT", _DEFAULT_CHECKPOINT)
    paths = [part.strip() for part in raw.split(",") if part.strip()]
    if not paths:
        raise ValueError("CLASSIFIER_REWARD_CKPT did not contain a checkpoint path.")
    return paths


def _get_reward_model(device: torch.device):
    global _REWARD_MODEL
    paths = _reward_checkpoint_paths()
    key = ",".join(paths)
    if _REWARD_MODEL is None or _REWARD_MODEL.checkpoint_path != key:
        missing = [path for path in paths if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                "Missing frozen reward checkpoint(s): " + ", ".join(missing)
            )
        _REWARD_MODEL = (
            ClassifierReward(paths[0], device)
            if len(paths) == 1
            else EnsembleClassifierReward(paths, device)
        )
        print(
            f"[reward] members={len(paths)} "
            f"type={os.environ.get('REWARD_TYPE', 'prob')}"
        )
    return _REWARD_MODEL.to(device)


def reward_from_probs(
    probs: torch.Tensor,
    target_idx: torch.Tensor,
    reward_type: str,
    *,
    gamma: float = 2.0,
    eta: float = 1.0,
) -> torch.Tensor:
    """Apply exactly the three reward equations reported in the manuscript."""
    if reward_type not in REWARD_TYPES:
        raise ValueError(
            f"Unknown REWARD_TYPE={reward_type!r}; choose from {REWARD_TYPES}."
        )
    p_true = probs.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    if reward_type == "prob":
        return p_true

    focal = ((1.0 - p_true) ** gamma) * p_true
    if reward_type == "focal":
        return focal

    correct = probs.argmax(dim=1) == target_idx
    return torch.where(correct, focal, torch.full_like(focal, -eta))


@torch.no_grad()
def compute_rewards(
    images: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
    classes: Optional[torch.Tensor] = None,
    batch: Optional[Dict[str, torch.Tensor]] = None,
    group_size: int = 1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Score generated candidates; masks/group_size are accepted by the RL hook."""
    del masks, group_size
    target_onehot = (
        classes if classes is not None else (batch.get("classes") if batch else None)
    )
    if target_onehot is None:
        raise ValueError("Reward scoring requires one-hot target classes.")
    target_idx = target_onehot.argmax(dim=1).to(images.device)

    probs = _get_reward_model(images.device).predict_probs(images)
    reward_type = os.environ.get("REWARD_TYPE", "prob")
    gamma = float(os.environ.get("REWARD_FOCAL_GAMMA", 2.0))
    eta = float(os.environ.get("REWARD_UTILITY_WRONG_PENALTY", 1.0))
    rewards = reward_from_probs(
        probs, target_idx, reward_type, gamma=gamma, eta=eta
    )

    p_true = probs.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    metrics = {
        "classifier_prob": p_true.detach(),
        "classifier_accuracy": (probs.argmax(dim=1) == target_idx).float().mean(),
        "reward_value": rewards.detach(),
    }
    return rewards.detach(), metrics
