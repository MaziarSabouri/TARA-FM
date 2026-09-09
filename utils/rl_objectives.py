"""Small, dependency-light helpers for the reported RL objective."""

from typing import Optional

import torch


def group_topk_weights(
    rewards: torch.Tensor,
    group_size: int,
    top_k: int,
    reward_floor: Optional[float] = None,
) -> torch.Tensor:
    """Assign equal normalized weight to each group's highest-reward rollouts."""
    grouped = rewards.view(-1, group_size)
    k_eff = max(1, min(int(top_k), group_size))
    keep = torch.zeros_like(grouped)
    keep.scatter_(1, grouped.topk(k_eff, dim=1).indices, 1.0)
    if reward_floor is not None:
        keep = keep * (grouped >= float(reward_floor)).to(keep.dtype)
    n_keep = keep.sum(dim=1, keepdim=True)
    weights = torch.where(
        n_keep > 0,
        keep * (group_size / n_keep.clamp_min(1.0)),
        keep,
    )
    return weights.view(-1).detach()
