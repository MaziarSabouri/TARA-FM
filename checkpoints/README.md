# Checkpoints

Trained weights are **not** included. After training they land here by default:

| path | contents |
|------|----------|
| `checkpoints/fm/` | Base mask+class Flow Matching Lightning checkpoints |
| `checkpoints/rl/` | RL fine-tuned checkpoints (one run per `REWARD_TYPE`) |
| `checkpoints/reward/` | Frozen reward classifier(s) you supply (`CLASSIFIER_REWARD_CKPT`) |

Typical usage:

```bash
# FM
bash scripts/train_fm.sh
# → checkpoints/fm/...

# RL (requires a reward .pth and a base FM .ckpt)
export CLASSIFIER_REWARD_CKPT=checkpoints/reward/reward_resnet18.pth
export FM_CKPT=checkpoints/fm/<your_fm>.ckpt
REWARD_TYPE=prob bash scripts/train_rl.sh
```

Sampling reads `--model_path` (FM or RL `.ckpt`); see `scripts/sample.sh`.
