from __future__ import annotations

import torch
from .utils import masked_normalize

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the SFT loss and backpropagate for one microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length)
            containing per-token log-probabilities from the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length)
            containing 1 for response tokens and 0 for prompt/padding tokens.
        gradient_accumulation_steps: Number of microbatches per optimizer step.
        normalize_constant: Constant factor used to normalize the masked loss.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            The scaled microbatch loss and optional metadata for logging.
    """
    per_token_loss = -policy_log_probs
    per_batch_loss = masked_normalize(
        tensor=per_token_loss,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=-1,
    )
    scaled_loss = per_batch_loss.mean() / gradient_accumulation_steps
    scaled_loss.backward()

    return scaled_loss, {}
