from __future__ import annotations

from typing import Any, Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .utils import compute_entropy

def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    generation_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate and summarize model responses for logging.

    Args:
        model: Hugging Face causal LM used for generation.
        tokenizer: Tokenizer paired with the model.
        prompts: Prompt strings to generate from.
        ground_truths: Ground-truth answers aligned with prompts.
        reward_fn: Callable that scores a generated response against a
            ground-truth answer. Expected to return reward information such as
            total reward, format reward, and answer reward.
        generation_kwargs: Optional kwargs forwarded to `model.generate(...)`.

    Returns:
        dict[str, Any] containing per-example logging information and aggregate
        summary statistics.
    """
    assert len(prompts) == len(ground_truths)

    generation_kwargs = generation_kwargs or {}

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    original_padding_side = tokenizer.padding_side
    if not model.config.is_encoder_decoder:
        tokenizer.padding_side = "left"

    prompt_batch = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(model.device)
    tokenizer.padding_side = original_padding_side


    generation_output = model.generate(
        **prompt_batch,
        **generation_kwargs,
        return_dict_in_generate=True,
        output_scores=True,
    )
    generated_ids = generation_output.sequences
    scores = generation_output.scores
    prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
    raw_response_ids = [
        generated_ids[i, prompt_length:]
        for i, prompt_length in enumerate(prompt_lengths)
    ]
    special_token_ids = set(tokenizer.all_special_ids)
    response_ids = [
        torch.tensor(
            [token_id for token_id in response_id.tolist() if token_id not in special_token_ids],
            device=response_id.device,
            dtype=response_id.dtype,
        )
        for response_id in raw_response_ids
    ]
    responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

    if scores:
        token_entropies = compute_entropy(torch.stack(scores, dim=1))
        avg_token_entropies = torch.tensor(
            [
                float(token_entropies[i, : len(response_id)].mean())
                if len(response_id) > 0
                else float("nan")
                for i, response_id in enumerate(response_ids)
            ],
            device=token_entropies.device,
        )
    else:
        avg_token_entropies = torch.full((len(prompts),), float("nan"))

    examples: list[dict[str, Any]] = []
    response_lengths: list[int] = []
    correct_response_lengths: list[int] = []
    incorrect_response_lengths: list[int] = []

    for prompt, response, response_id, ground_truth, avg_entropy in zip(
        prompts,
        responses,
        response_ids,
        ground_truths,
        avg_token_entropies,
    ):
        reward_info = reward_fn(response, ground_truth)
        response_length = len(response_id)

        is_correct = reward_info["answer_reward"] == 1.0

        response_lengths.append(response_length)
        if is_correct:
            correct_response_lengths.append(response_length)
        else:
            incorrect_response_lengths.append(response_length)

        examples.append(
            {
                "prompt": prompt,
                "response": response,
                "ground_truth": ground_truth,
                "reward": reward_info.get("reward"),
                "format_reward": reward_info.get("format_reward"),
                "answer_reward": reward_info.get("answer_reward"),
                "avg_token_entropy": float(avg_entropy),
                "response_length": response_length,
            }
        )

    def _safe_average(values: list[int | float]) -> float | None:
        return None if len(values) == 0 else float(sum(values) / len(values))

    return {
        "examples": examples,
        "avg_response_length": _safe_average(response_lengths),
        "avg_response_length_correct": _safe_average(correct_response_lengths),
        "avg_response_length_incorrect": _safe_average(incorrect_response_lengths),
    }
