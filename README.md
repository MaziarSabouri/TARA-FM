# Task-Aware Reinforcement Learning of Flow Matching

Minimal code companion to the paper
**“Task-Aware Reinforcement Learning of Flow Matching: Aligning Thyroid
Scintigraphy Augmentation With Downstream Classification.”**

This release contains only:

- preparation of the development dataset;
- mask- and class-conditioned Flow Matching (FM) training;
- RL fine-tuning with the paper’s Soft, Focal, and Utility rewards; and
- FM/RL image generation.

Patient data and trained weights are not included. They are available only
subject to institutional agreements.

## Installation

Python 3.10 and a CUDA-capable GPU are recommended.

```bash
conda create -n rlfm python=3.10 -y
conda activate rlfm
pip install -r requirements.txt
```



## Data

Configs expect `data/train_data_128.pkl`, containing development `train` and
`valid` splits. See `[data/README.md](data/README.md)` for the schema and raw-data
conversion command.

The external cohort is neither required nor accessed by this repository.

## 1. Train the base FM

```bash
bash scripts/train_fm.sh
```

The paper recipe is in `configs/fm_mask_class.yaml`: 200 epochs, Adam with learning rate (1e-4), effective batch size 256, and mask plus four-class conditioning.

## 2. Fine-tune the three RL variants

Supply the base FM checkpoint and the three frozen ResNet-18 judge checkpoints
as a comma-separated ensemble:

```bash
export FM_CKPT=checkpoints/fm/<base-fm>.ckpt
export CLASSIFIER_REWARD_CKPT=checkpoints/reward/fold0.pth,checkpoints/reward/fold1.pth,checkpoints/reward/fold2.pth

REWARD_TYPE=prob    bash scripts/train_rl.sh  # Soft-RL
REWARD_TYPE=focal   bash scripts/train_rl.sh  # Focal-RL
REWARD_TYPE=utility bash scripts/train_rl.sh  # Utility-RL
```

The reported RL recipe performs 4,000 Adam updates at learning rate 1e-5, with four rollouts per condition, hard top-2 selection, a 0.3 real-data anchor, a 0.05 frozen-base velocity regularizer, unit-norm gradient clipping, and ten Euler steps per rollout. Checkpoints are saved every 1,000 updates.

The reward checkpoint format is documented in
`[checkpoints/README.md](checkpoints/README.md)`.

## 3. Select and sample a checkpoint

RL checkpoints must be selected using development validation only—never the
external cohort. See `[docs/CHECKPOINT_SELECTION.md](docs/CHECKPOINT_SELECTION.md)`.

```bash
MODEL_PATH=checkpoints/rl/prob/<selected>.ckpt \
OUT=data/samples_rl_soft.pkl \
bash scripts/sample.sh
```

`PER_CLASS`, `STEPS`, `BATCH`, `SEED`, and `GEN_SPLIT` can be overridden.

## Method-to-code alignment

`[docs/METHOD_ALIGNMENT.md](docs/METHOD_ALIGNMENT.md)` maps each paper equation
to its implementation and records the fixed experiment constants. Formula-level
tests are in `tests/test_formulas.py`.

## Repository map


| Path                            | Purpose                                        |
| ------------------------------- | ---------------------------------------------- |
| `fm_lightning.py`, `trainer.py` | Conditional FM objective and training          |
| `rl_trainier.py`                | Top-k self-distillation and anchored RL loss   |
| `rewards.py`                    | Soft, Focal, and Utility reward equations      |
| `utils/reward_model.py`         | Frozen ResNet-18 ensemble loader               |
| `inferer.py`                    | Euler/ODE image generation                     |
| `configs/`                      | Exact paper FM and RL recipes                  |
| `scripts/`                      | Data preparation, FM/RL training, and sampling |




## Citation

```bibtex
@article{sabouri2026taskaware,
  title   = {Task-Aware Reinforcement Learning of Flow Matching: Aligning Thyroid Scintigraphy Augmentation With Downstream Classification},
  author  = {Sabouri, Maziar and Yazdani, Milad and Bayasi, Nourhan and others},
  journal = {TBD},
  year    = {2026},
  note    = {TBD}
}
```

This research release does not grant rights to patient data or model weights.
Contact the authors regarding reuse.