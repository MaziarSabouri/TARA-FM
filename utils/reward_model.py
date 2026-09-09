"""Minimal loader for the frozen ResNet-18 reward judges used by RL."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torchvision import models


CLASS_NAMES = ["Normal", "Diffuse Goiter", "Nodular Goiter", "Thyroiditis"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIMPLE_MEAN = (0.5, 0.5, 0.5)
SIMPLE_STD = (0.5, 0.5, 0.5)


def build_reward_resnet18(num_classes: int) -> nn.Module:
    """Recreate the 128-pixel, modified-stem ResNet-18 judge architecture."""
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def load_reward_classifier(path: str, device: torch.device) -> Dict:
    """Load the model and preprocessing metadata saved with a reward checkpoint."""
    checkpoint = torch.load(
        path, map_location=device, weights_only=False
    )  # trusted, locally produced checkpoint
    class_names = list(checkpoint.get("class_names", CLASS_NAMES))
    backbone = str(checkpoint.get("backbone", "resnet18")).lower()
    if backbone not in {"resnet18", "resnet-18", "resnet_18"}:
        raise ValueError(
            f"The paper reward uses ResNet-18; checkpoint declares {backbone!r}."
        )

    model = build_reward_resnet18(len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    pretrained_norm = bool(checkpoint.get("pretrained_norm", False))
    mean = IMAGENET_MEAN if pretrained_norm else SIMPLE_MEAN
    std = IMAGENET_STD if pretrained_norm else SIMPLE_STD
    return {
        "model": model,
        "class_names": class_names,
        "input_size": int(checkpoint.get("input_size") or 128),
        "mean": mean,
        "std": std,
    }
