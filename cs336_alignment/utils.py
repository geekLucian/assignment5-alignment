from __future__ import annotations

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, Tensor]:
    """Tokenize prompt/output separately, concatenate, and build response_mask.

    This is a teaching template for the assignment. Fill in the TODOs yourself.
    The expected return value is a dictionary with keys:
      - "input_ids"
      - "labels"
      - "response_mask"
    """
    assert len(prompt_strs) == len(output_strs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_batch = tokenizer(
        prompt_strs,
        add_special_tokens=False,
        padding=True,
        return_attention_mask=True,
    )

    output_batch = tokenizer(
        output_strs,
        add_special_tokens=False,
        padding=True,
        return_attention_mask=True,
    )

    input_id_rows: list[Tensor] = []
    label_rows: list[Tensor] = []
    response_mask_rows: list[Tensor] = []

    for prompt_ids, prompt_mask, output_ids, output_mask in zip(
        prompt_batch["input_ids"],
        prompt_batch["attention_mask"],
        output_batch["input_ids"],
        output_batch["attention_mask"],
    ):
        prompt_len = sum(prompt_mask)

        full_ids = prompt_ids[:prompt_len] + output_ids
        input_ids = full_ids[:-1]
        labels = full_ids[1:]
        response_mask = [0] * (prompt_len - 1) + output_mask

        input_id_rows.append(torch.tensor(input_ids, dtype=torch.long))
        label_rows.append(torch.tensor(labels, dtype=torch.long))
        response_mask_rows.append(torch.tensor(response_mask, dtype=torch.long))

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_id_rows,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        label_rows,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    response_mask = torch.nn.utils.rnn.pad_sequence(
        response_mask_rows,
        batch_first=True,
        padding_value=0,
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the next-token predictions.

    Args:
        logits: torch.Tensor of shape (batch_size, sequence_length, vocab_size)
            containing unnormalized logits.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length) containing the
        entropy of each next-token prediction.
    """
    log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    prob = torch.exp(log_prob)
    entropy = torch.sum(-prob * log_prob, dim=-1)
    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities from a causal LM.

    Args:
        model: Hugging Face causal LM used for scoring.
        input_ids: torch.Tensor of shape (batch_size, sequence_length)
            containing tokenized prompt + response tokens.
        labels: torch.Tensor of shape (batch_size, sequence_length)
            containing the shifted target tokens.
        return_token_entropy: If True, also return per-token entropy for the
            next-token distribution.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length)
                containing the conditional log-probability of each label token.
            "token_entropy": Optional[torch.Tensor] of shape
                (batch_size, sequence_length), included only when requested.
    """
    # (batch_size, sequence_length, vocab_size).
    logits = model(input_ids).logits

    logsumexp = torch.logsumexp(logits, dim=-1)
    label_logit = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = label_logit - logsumexp

    output = {"log_probs": token_log_probs}

    if return_token_entropy:
        output["token_entropy"] = compute_entropy(logits)

    return output


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum over tensor elements and normalize by a constant, respecting a mask.

    Args:
        tensor: torch.Tensor to sum and normalize.
        mask: torch.Tensor with the same shape as tensor. Positions with value 1
            are included; positions with value 0 do not contribute.
        normalize_constant: Constant factor to divide the masked sum by.
        dim: Dimension to sum along. If None, sum across all dimensions.

    Returns:
        torch.Tensor containing the masked, normalized sum.
    """
    masked_tensor = tensor * mask
    masked_sum = torch.sum(masked_tensor, dim=dim)
    return masked_sum / normalize_constant
