from pathlib import Path

import torch
import yaml

from rewards import reward_from_probs
from utils.rl_objectives import group_topk_weights


ROOT = Path(__file__).resolve().parents[1]


def test_reported_reward_equations() -> None:
    probs = torch.tensor(
        [
            [0.60, 0.20, 0.10, 0.10],
            [0.40, 0.50, 0.05, 0.05],
        ]
    )
    targets = torch.tensor([0, 0])
    expected_focal = torch.tensor([0.60 * 0.40**2, 0.40 * 0.60**2])

    assert torch.allclose(
        reward_from_probs(probs, targets, "prob"), torch.tensor([0.60, 0.40])
    )
    assert torch.allclose(
        reward_from_probs(probs, targets, "focal"), expected_focal
    )
    assert torch.allclose(
        reward_from_probs(probs, targets, "utility"),
        torch.tensor([expected_focal[0], -1.0]),
    )


def test_top_two_weights_sum_to_group_size() -> None:
    rewards = torch.tensor([0.1, 0.9, 0.4, 0.7, -1.0, 0.2, 0.1, 0.0])
    weights = group_topk_weights(rewards, group_size=4, top_k=2).view(2, 4)
    assert torch.equal((weights > 0).sum(dim=1), torch.tensor([2, 2]))
    assert torch.allclose(weights.sum(dim=1), torch.tensor([4.0, 4.0]))
    assert set(weights.flatten().tolist()) == {0.0, 2.0}


def test_paper_rl_config_constants() -> None:
    with (ROOT / "configs" / "rl_mask_class.yaml").open() as stream:
        config = yaml.safe_load(stream)
    fm_args = config["fm_train_args"]
    args = config["rl_train_args"]

    assert fm_args["num_epochs"] == 200
    assert fm_args["batch_size"] * fm_args["gradient_accumulation_steps"] == 256
    assert fm_args["lr"] == 1e-4
    assert args["max_updates"] == 4000
    assert args["lr"] == 1e-5
    assert args["group_size"] == 4
    assert args["reward_topk"] == 2
    assert args["real_anchor_weight"] == 0.3
    assert args["ref_reg_weight"] == 0.05
    assert args["grad_clip_norm"] == 1.0
    assert args["num_inference_steps"] == 10
    assert args["ckpt_every_n_steps"] == 1000
    assert config["solver_args"] == {
        "method": "euler",
        "step_size": 0.1,
        "time_points": 10,
    }
