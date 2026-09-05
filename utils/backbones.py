"""Downstream classifier backbones for the multi-architecture generalization study.

The SPIE conference version used a single ResNet18 both as the RL reward model
and as the downstream classifier, which leaves open the objection that
RL-tuned synthetics simply exploit ResNet18-specific features ("reward
gaming"). This module provides a common factory so the *same* augmentation
pools can be evaluated with convolutional (ResNet, ConvNeXt, DenseNet,
EfficientNet) and attention-based (Swin) inductive biases.

Every backbone shares:
  - ImageNet-1k initialization (optional),
  - a `Dropout(p) -> Linear(num_classes)` head,
  - ImageNet channel normalization when pretrained, else (0.5, 0.5, 0.5).

`resnet18` additionally keeps the exact protocol of the conference paper: the
7x7/stride-2 stem is replaced by a 3x3/stride-1 convolution (center-cropped
from the pretrained kernel), so 128x128 scintigrams keep a 16x16 final feature
map. The other backbones use their stock stems and their native 224x224 input
resolution, since their pretrained weights are tightly coupled to the stem
geometry (patch embedding for Swin, 4x4 downsampling stem for ConvNeXt).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIMPLE_MEAN = (0.5, 0.5, 0.5)
SIMPLE_STD = (0.5, 0.5, 0.5)


@dataclass
class BackboneSpec:
    """Everything the training/interpretability code needs to know about a backbone."""

    name: str
    display_name: str
    family: str  # "cnn" | "transformer"
    input_size: int
    builder: Callable[..., nn.Module]
    # Returns the conv/stage module Grad-CAM should hook, or None if unsupported.
    gradcam_layer: Optional[Callable[[nn.Module], nn.Module]] = None
    # Swin's torchvision blocks emit [B, H, W, C] instead of [B, C, H, W].
    gradcam_channels_last: bool = False
    # Rough relative training cost, only used to print budget estimates.
    relative_cost: float = 1.0
    aliases: List[str] = field(default_factory=list)


def _replace_head(head_in_features: int, num_classes: int, dropout: float) -> nn.Module:
    return nn.Sequential(nn.Dropout(p=dropout), nn.Linear(head_in_features, num_classes))


###############################################################################
# Builders
###############################################################################
def _build_resnet(depth: int, small_stem: bool):
    def _builder(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
        from torchvision import models

        if depth == 18:
            ctor, weights_enum = models.resnet18, models.ResNet18_Weights
        elif depth == 50:
            ctor, weights_enum = models.resnet50, models.ResNet50_Weights
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")

        model = ctor(weights=weights_enum.IMAGENET1K_V1 if pretrained else None)

        if small_stem:
            # Paper protocol: 3x3/stride-1 stem so 128x128 inputs keep spatial
            # detail. When pretrained, seed it with the center 3x3 of the 7x7.
            old_w = model.conv1.weight.data
            new_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if pretrained:
                with torch.no_grad():
                    new_conv.weight.copy_(old_w[:, :, 2:5, 2:5])
            model.conv1 = new_conv

        model.fc = _replace_head(model.fc.in_features, num_classes, dropout)
        return model

    return _builder


def _build_convnext_tiny(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    from torchvision import models

    model = models.convnext_tiny(
        weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    )
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    # ConvNeXt's classifier is LayerNorm2d -> Flatten -> Linear; insert dropout
    # before the Linear rather than replacing the whole (norm-carrying) block.
    model.classifier = nn.Sequential(*list(model.classifier[:-1]), nn.Dropout(p=dropout), model.classifier[-1])
    return model


def _build_swin_t(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    from torchvision import models

    model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
    model.head = _replace_head(model.head.in_features, num_classes, dropout)
    return model


def _build_densenet121(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    from torchvision import models

    model = models.densenet121(
        weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    )
    model.classifier = _replace_head(model.classifier.in_features, num_classes, dropout)
    return model


def _build_efficientnet_b0(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    from torchvision import models

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    )
    model.classifier = _replace_head(model.classifier[-1].in_features, num_classes, dropout)
    return model


def _build_vit_b_16(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    from torchvision import models

    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
    model.heads = _replace_head(model.heads.head.in_features, num_classes, dropout)
    return model


###############################################################################
# Registry
###############################################################################
BACKBONES: Dict[str, BackboneSpec] = {
    "resnet18": BackboneSpec(
        name="resnet18",
        display_name="ResNet18",
        family="cnn",
        input_size=128,
        builder=_build_resnet(18, small_stem=True),
        gradcam_layer=lambda m: m.layer4[-1],
        relative_cost=1.0,
        aliases=["resnet-18", "r18"],
    ),
    "resnet50": BackboneSpec(
        name="resnet50",
        display_name="ResNet50",
        family="cnn",
        input_size=224,
        builder=_build_resnet(50, small_stem=False),
        gradcam_layer=lambda m: m.layer4[-1],
        relative_cost=2.0,
        aliases=["r50"],
    ),
    "convnext_tiny": BackboneSpec(
        name="convnext_tiny",
        display_name="ConvNeXt-T",
        family="cnn",
        input_size=224,
        builder=_build_convnext_tiny,
        gradcam_layer=lambda m: m.features[-1][-1],
        relative_cost=2.6,
        aliases=["convnext", "convnext-t"],
    ),
    "swin_t": BackboneSpec(
        name="swin_t",
        display_name="Swin-T",
        family="transformer",
        input_size=224,
        builder=_build_swin_t,
        gradcam_layer=lambda m: m.features[-1][-1],
        gradcam_channels_last=True,
        relative_cost=2.8,
        aliases=["swin", "swin-t"],
    ),
    "densenet121": BackboneSpec(
        name="densenet121",
        display_name="DenseNet121",
        family="cnn",
        input_size=224,
        builder=_build_densenet121,
        gradcam_layer=lambda m: m.features.denseblock4,
        relative_cost=2.4,
        aliases=["densenet"],
    ),
    "efficientnet_b0": BackboneSpec(
        name="efficientnet_b0",
        display_name="EfficientNet-B0",
        family="cnn",
        input_size=224,
        builder=_build_efficientnet_b0,
        gradcam_layer=lambda m: m.features[-1],
        relative_cost=1.6,
        aliases=["effnet", "efficientnet"],
    ),
    "vit_b_16": BackboneSpec(
        name="vit_b_16",
        display_name="ViT-B/16",
        family="transformer",
        input_size=224,
        builder=_build_vit_b_16,
        gradcam_layer=None,  # token-grid reshaping is model-specific; not wired up
        relative_cost=4.5,
        aliases=["vit"],
    ),
}

_ALIAS_TO_NAME = {a: spec.name for spec in BACKBONES.values() for a in [spec.name, *spec.aliases]}


def resolve_backbone_name(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "")
    if key in _ALIAS_TO_NAME:
        return _ALIAS_TO_NAME[key]
    raise ValueError(
        f"Unknown backbone {name!r}. Available: {sorted(BACKBONES)} "
        f"(aliases: {sorted(set(_ALIAS_TO_NAME) - set(BACKBONES))})"
    )


def get_backbone_spec(name: str) -> BackboneSpec:
    return BACKBONES[resolve_backbone_name(name)]


def build_backbone(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    spec = get_backbone_spec(name)
    return spec.builder(num_classes=num_classes, pretrained=pretrained, dropout=dropout)


def normalization_for(pretrained: bool) -> Sequence[Sequence[float]]:
    return (IMAGENET_MEAN, IMAGENET_STD) if pretrained else (SIMPLE_MEAN, SIMPLE_STD)


def resolve_input_size(name: str, override: Optional[int] = None) -> int:
    return int(override) if override else get_backbone_spec(name).input_size


###############################################################################
# Feature extraction (used by the embedding / distribution-shift analysis)
###############################################################################
def build_feature_extractor(name: str, pretrained: bool = True) -> nn.Module:
    """Return the backbone with its classification head replaced by Identity, so
    a forward pass yields penultimate-layer embeddings.

    Used to embed real vs. synthetic images in a feature space that is
    *independent* of both the reward model and InceptionV3 (which FID uses),
    to test whether the RL-induced distribution shift is Inception-specific.
    """
    spec = get_backbone_spec(name)
    model = spec.builder(num_classes=1, pretrained=pretrained, dropout=0.0)

    if spec.name.startswith("resnet"):
        model.fc = nn.Identity()
    elif spec.name == "convnext_tiny":
        model.classifier = nn.Sequential(*list(model.classifier[:-2]))  # drop Dropout+Linear
    elif spec.name == "swin_t":
        model.head = nn.Identity()
    elif spec.name == "densenet121":
        model.classifier = nn.Identity()
    elif spec.name == "efficientnet_b0":
        model.classifier = nn.Identity()
    elif spec.name == "vit_b_16":
        model.heads = nn.Identity()
    else:
        raise ValueError(f"No feature-extractor rule for {spec.name}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
