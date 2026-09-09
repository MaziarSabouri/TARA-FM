# Checkpoints

Trained weights are **not** included. After training they land here by default:

| path | contents |
|------|----------|
| `checkpoints/fm/` | Base mask+class Flow Matching Lightning checkpoints |
| `checkpoints/rl/` | RL fine-tuned checkpoints (one run per `REWARD_TYPE`) |
| `checkpoints/reward/` | Three frozen ResNet-18 judges supplied as RL inputs |

Typical usage:

```bash
# FM
bash scripts/train_fm.sh
# → checkpoints/fm/...

# RL (requires the three-member judge ensemble and a base FM .ckpt)
export CLASSIFIER_REWARD_CKPT=checkpoints/reward/fold0.pth,checkpoints/reward/fold1.pth,checkpoints/reward/fold2.pth
export FM_CKPT=checkpoints/fm/<your_fm>.ckpt
REWARD_TYPE=prob bash scripts/train_rl.sh
```

Sampling reads `--model_path` (FM or RL `.ckpt`); see `scripts/sample.sh`.

Each trusted, locally produced judge checkpoint is a PyTorch dictionary with:

```text
{
  "state_dict": <modified-stem ResNet-18 state dict>,
  "class_names": ["Normal", "Diffuse Goiter", "Nodular Goiter", "Thyroiditis"],
  "backbone": "resnet18",
  "input_size": 128,
  "pretrained_norm": <bool>
}
```

The reward loader reconstructs the paper architecture: a 3x3, stride-1 stem and
`Dropout(0.2) -> Linear(4)` classification head. The classifier training code
and patient-derived weights are intentionally not part of this release.
