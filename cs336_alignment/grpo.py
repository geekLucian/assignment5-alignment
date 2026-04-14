from __future__ import annotations

from typing import Callable, Literal

import torch


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute raw rewards and normalize them within each rollout group.

    Expected shapes:
    - len(rollout_responses) == len(repeated_ground_truths) == rollout_batch_size
    - rollout_batch_size is divisible by group_size

    Returns:
    - advantages: shape (rollout_batch_size,)
    - raw_rewards: shape (rollout_batch_size,)
    - metadata: any reward statistics you want to log
    """
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("rollout_responses and repeated_ground_truths must have the same length")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if len(rollout_responses) % group_size != 0:
        raise ValueError("rollout batch size must be divisible by group_size")

    raw_rewards = torch.tensor(
        [
            reward_fn(rollout_response, repeated_ground_truth)["reward"]
            for rollout_response, repeated_ground_truth in zip(
                rollout_responses, repeated_ground_truths
            )
        ],
        dtype=torch.float32,
    )

    grouped_rewards = raw_rewards.view(-1, group_size)

    group_means = grouped_rewards.mean(dim=-1, keepdim=True)
    advantages = grouped_rewards - group_means

    if normalize_by_std:
        group_stds = grouped_rewards.std(dim=-1, keepdim=True)
        advantages = advantages / (group_stds + advantage_eps)

    advantages = advantages.flatten()

    metadata = {
        "reward_mean": float(raw_rewards.mean()),
        "reward_std": float(raw_rewards.std()),
        "reward_min": float(raw_rewards.min()),
        "reward_max": float(raw_rewards.max()),
    }

    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute the per-token naive policy-gradient loss.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1).
            One scalar reward or advantage per rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length).
            Log-probability of each generated token under the current policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length) containing the
        per-token policy-gradient loss.
    """
    if raw_rewards_or_advantages.ndim != 2 or raw_rewards_or_advantages.shape[1] != 1:
        raise ValueError("raw_rewards_or_advantages must have shape (batch_size, 1)")
    if policy_log_probs.ndim != 2:
        raise ValueError("policy_log_probs must have shape (batch_size, sequence_length)")
    if raw_rewards_or_advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError("batch sizes must match")

    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the per-token GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1).
            One scalar advantage per rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length).
            Per-token log-probabilities under the current policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length).
            Per-token log-probabilities under the old policy.
        cliprange: float. PPO/GRPO clipping parameter epsilon.

    Returns:
        tuple containing:
        - loss: torch.Tensor of shape (batch_size, sequence_length)
        - metadata: dict of tensors you may want to inspect or log
    """
    if advantages.ndim != 2 or advantages.shape[1] != 1:
        raise ValueError("advantages must have shape (batch_size, 1)")
    if policy_log_probs.ndim != 2 or old_log_probs.ndim != 2:
        raise ValueError("policy_log_probs and old_log_probs must be rank-2 tensors")
    if policy_log_probs.shape != old_log_probs.shape:
        raise ValueError("policy_log_probs and old_log_probs must have the same shape")
    if advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError("batch sizes must match")

    ratio = torch.exp(policy_log_probs - old_log_probs)

    clipped_ratio = torch.clip(ratio, 1-cliprange, 1+cliprange)

    unclipped_objective = ratio * advantages
    clipped_objective = clipped_ratio * advantages

    loss = -torch.min(unclipped_objective, clipped_objective)

    metadata = {
        "ratio": ratio,
        "clipped_ratio": clipped_ratio,
        "was_clipped": clipped_objective <= unclipped_objective,
    }
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dispatch to the requested policy-gradient loss routine.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length).
        loss_type: Which loss routine to use.
        raw_rewards: Required for ``loss_type == "no_baseline"``.
        advantages: Required for ``loss_type in {"reinforce_with_baseline", "grpo_clip"}``.
        old_log_probs: Required for ``loss_type == "grpo_clip"``.
        cliprange: Required for ``loss_type == "grpo_clip"``.

    Returns:
        tuple containing:
        - loss: torch.Tensor of shape (batch_size, sequence_length)
        - metadata: dict of auxiliary tensors/statistics from the chosen routine
    """
    if policy_log_probs.ndim != 2:
        raise ValueError("policy_log_probs must have shape (batch_size, sequence_length)")

    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError('raw_rewards is required for loss_type == "no_baseline"')
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}
    elif loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError(
                'advantages is required for loss_type == "reinforce_with_baseline"'
            )
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}
    elif loss_type == "grpo_clip":
        if advantages is None:
            raise ValueError('advantages is required for loss_type == "grpo_clip"')
        if old_log_probs is None:
            raise ValueError('old_log_probs is required for loss_type == "grpo_clip"')
        if cliprange is None:
            raise ValueError('cliprange is required for loss_type == "grpo_clip"')
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    return loss, metadata


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Compute the mean of tensor over the masked elements."""
    if tensor.shape != mask.shape:
        raise ValueError("tensor and mask must have the same shape")

    masked_tensor = tensor * mask
    total = torch.sum(masked_tensor, dim=dim)
    count = torch.sum(mask, dim=dim)
    return total / count


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute a forward-and-backward pass on a GRPO microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length).
            Per-token log-probabilities from the policy being trained.
        response_mask: torch.Tensor of shape (batch_size, sequence_length).
            Boolean or 0/1 mask indicating which positions belong to the response.
        gradient_accumulation_steps: Number of microbatches accumulated before
            the optimizer step. The returned loss is typically scaled by this value
            before calling backward().
        loss_type: Which policy-gradient loss to use. Must be one of
            "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards: Required when ``loss_type == "no_baseline"``.
            Shape (batch_size, 1), one scalar reward per example.
        advantages: Required when ``loss_type != "no_baseline"``.
            Shape (batch_size, 1), one scalar advantage per example.
        old_log_probs: Required when ``loss_type == "grpo_clip"``.
            Shape (batch_size, sequence_length), containing log-probabilities
            under the reference / previous policy.
        cliprange: Required when ``loss_type == "grpo_clip"``.
            PPO/GRPO clipping parameter epsilon.

    Returns:
        tuple containing:
        - loss: Scalar microbatch loss tensor, adjusted for gradient accumulation.
        - metadata: Auxiliary tensors/statistics from the underlying loss routine
          and any additional values useful for logging.
    """
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
    )

    response_loss = masked_mean(loss, response_mask) / gradient_accumulation_steps
    response_loss.backward()

    return response_loss, metadata
