from __future__ import annotations

import logging
import random
import re
import shutil
from pathlib import Path
from statistics import mean
from typing import Any, Literal

import torch
import typer
import wandb
from datasets import load_dataset
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.grpo import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
)
from cs336_alignment.utils import get_response_log_probs, tokenize_prompt_and_output

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
PROMPT_PATH = Path(__file__).resolve().parents[1] / "cs336_alignment" / "prompts" / "r1_zero.prompt"


def load_prompt_template(prompt_path: Path) -> str:
    return prompt_path.read_text().strip()


def ensure_vllm_tokenizer_compat() -> None:
    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: list(self.all_special_tokens)
        )


def format_question(example: dict[str, Any]) -> str:
    if "question" in example and example["question"]:
        return str(example["question"])
    if "problem" in example and example["problem"]:
        return str(example["problem"])
    if "prompt" in example and example["prompt"]:
        return str(example["prompt"])

    # TODO: extend this formatter if you switch to a different math dataset.
    raise KeyError(f"Could not infer question field from keys: {sorted(example.keys())}")


def get_ground_truth(example: dict[str, Any]) -> str:
    if "answer" in example and isinstance(example["answer"], str):
        match = re.search(r"####\s*(.+?)\s*$", example["answer"], re.DOTALL)
        if match is not None:
            return match.group(1).strip().replace("$", "").replace(",", "")
        return example["answer"].strip()

    for key in ("solution", "target"):
        if key in example and example[key] is not None:
            return str(example[key])

    # TODO: add any dataset-specific answer extraction you want to experiment with.
    raise KeyError(f"Could not infer ground-truth field from keys: {sorted(example.keys())}")


def build_prompt(question: str, prompt_template: str) -> str:
    return prompt_template.format(question=question)


def normalize_response_for_reward(response: str) -> str:
    normalized = response
    if not normalized.rstrip().endswith("</answer>") and "<answer>" in normalized:
        normalized = normalized + "</answer>"
    return normalized


def repeat_each(items: list[str], n: int) -> list[str]:
    return [item for item in items for _ in range(n)]


def iterate_minibatch_indices(
    total_size: int,
    batch_size: int,
    rng: random.Random,
) -> list[torch.Tensor]:
    permutation = list(range(total_size))
    rng.shuffle(permutation)
    return [
        torch.tensor(permutation[start : start + batch_size], dtype=torch.long)
        for start in range(0, total_size, batch_size)
    ]


def gather_examples(
    examples: list[dict[str, Any]],
    start_idx: int,
    count: int,
) -> tuple[list[dict[str, Any]], int]:
    if count <= 0:
        raise ValueError("count must be positive")

    gathered = []
    idx = start_idx
    for _ in range(count):
        gathered.append(examples[idx % len(examples)])
        idx += 1
    return gathered, idx


def score_generations(
    responses: list[str],
    ground_truths: list[str],
) -> dict[str, float]:
    rewards = [r1_zero_reward_fn(response, ground_truth) for response, ground_truth in zip(responses, ground_truths)]
    return {
        "reward_mean": mean(reward["reward"] for reward in rewards),
        "format_reward_mean": mean(reward["format_reward"] for reward in rewards),
        "answer_reward_mean": mean(reward["answer_reward"] for reward in rewards),
    }


def evaluate_validation_rewards(
    vllm_model: LLM,
    prompt_template: str,
    validation_examples: list[dict[str, Any]],
    sampling_params: SamplingParams,
    num_examples: int,
) -> dict[str, float]:
    eval_examples = validation_examples[:num_examples]
    prompts = [build_prompt(format_question(example), prompt_template) for example in eval_examples]
    ground_truths = [get_ground_truth(example) for example in eval_examples]
    raw_outputs = vllm_model.generate(prompts, sampling_params)
    responses = [
        normalize_response_for_reward(output.outputs[0].text)
        for output in raw_outputs
    ]
    return score_generations(responses, ground_truths)


def initialize_vllm_model(
    model_path: str,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> LLM:
    ensure_vllm_tokenizer_compat()
    return LLM(
        model=model_path,
        tensor_parallel_size=max(1, torch.cuda.device_count()),
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )


def save_training_checkpoint(
    checkpoint_dir: Path,
    policy: AutoModelForCausalLM,
    tokenizer_source_path: str,
    optimizer: torch.optim.Optimizer,
    step_idx: int,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(checkpoint_dir)
    tokenizer_source_dir = Path(tokenizer_source_path)
    if tokenizer_source_dir.exists():
        for filename in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
        ):
            source_file = tokenizer_source_dir / filename
            if source_file.exists():
                shutil.copy2(source_file, checkpoint_dir / filename)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "step_idx": step_idx,
        },
        checkpoint_dir / "trainer_state.pt",
    )


@app.command()
def main(
    model_name_or_path: str = typer.Option(..., help="HF model name or local path."),
    train_split: str = typer.Option("train", help="Dataset split for GRPO training."),
    validation_split: str = typer.Option("test", help="Dataset split for reward evaluation."),
    n_grpo_steps: int = typer.Option(200),
    learning_rate: float = typer.Option(1e-5),
    advantage_eps: float = typer.Option(1e-6),
    rollout_batch_size: int = typer.Option(256),
    group_size: int = typer.Option(8),
    sampling_temperature: float = typer.Option(1.0),
    sampling_min_tokens: int = typer.Option(4),
    sampling_max_tokens: int = typer.Option(1024),
    epochs_per_rollout_batch: int = typer.Option(1),
    train_batch_size: int = typer.Option(256),
    gradient_accumulation_steps: int = typer.Option(128),
    gpu_memory_utilization: float = typer.Option(0.85),
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = typer.Option(
        "reinforce_with_baseline"
    ),
    use_std_normalization: bool = typer.Option(True),
    validation_eval_interval: int = typer.Option(10),
    validation_num_examples: int = typer.Option(1024),
    seed: int = typer.Option(42),
    max_model_len: int = typer.Option(2048),
    output_dir: Path = typer.Option(Path("outputs/grpo")),
    checkpoint_interval: int = typer.Option(10),
    use_wandb: bool = typer.Option(False),
    wandb_project: str = typer.Option("cs336-grpo"),
    wandb_run_name: str | None = typer.Option(None),
) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    assert rollout_batch_size % train_batch_size == 0, (
        "rollout_batch_size must be divisible by train_batch_size"
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    assert n_microbatches_per_rollout_batch % gradient_accumulation_steps == 0, (
        "rollout batch must contain a whole number of gradient-accumulation windows"
    )
    if loss_type == "grpo_clip" and epochs_per_rollout_batch == 1:
        logger.warning(
            "GRPO-Clip is usually most useful in the off-policy setting with multiple epochs."
        )

    rng = random.Random(seed)
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = load_prompt_template(PROMPT_PATH)

    logger.info("Loading tokenizer and policy from %s", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.train()

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    logger.info("Loading %s/%s train=%s validation=%s", DATASET_NAME, DATASET_CONFIG, train_split, validation_split)
    train_dataset = [dict(example) for example in load_dataset(DATASET_NAME, DATASET_CONFIG, split=train_split)]
    validation_dataset = [dict(example) for example in load_dataset(DATASET_NAME, DATASET_CONFIG, split=validation_split)]

    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "model_name_or_path": model_name_or_path,
                "train_split": train_split,
                "validation_split": validation_split,
                "n_grpo_steps": n_grpo_steps,
                "learning_rate": learning_rate,
                "advantage_eps": advantage_eps,
                "rollout_batch_size": rollout_batch_size,
                "group_size": group_size,
                "sampling_temperature": sampling_temperature,
                "sampling_min_tokens": sampling_min_tokens,
                "sampling_max_tokens": sampling_max_tokens,
                "epochs_per_rollout_batch": epochs_per_rollout_batch,
                "train_batch_size": train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "gpu_memory_utilization": gpu_memory_utilization,
                "loss_type": loss_type,
                "use_std_normalization": use_std_normalization,
                "validation_eval_interval": validation_eval_interval,
                "validation_num_examples": validation_num_examples,
                "seed": seed,
                "max_model_len": max_model_len,
            },
        )

    latest_policy_dir = output_dir / "latest_policy"
    save_training_checkpoint(
        checkpoint_dir=latest_policy_dir,
        policy=policy,
        tokenizer_source_path=model_name_or_path,
        optimizer=optimizer,
        step_idx=0,
    )
    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    train_cursor = 0

    for step_idx in range(n_grpo_steps):
        logger.info("Step %d: initializing vLLM for rollout generation", step_idx)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        vllm_model = initialize_vllm_model(
            model_path=str(latest_policy_dir),
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )

        rollout_examples, train_cursor = gather_examples(
            train_dataset,
            start_idx=train_cursor,
            count=n_prompts_per_rollout_batch,
        )
        prompts = [build_prompt(format_question(example), prompt_template) for example in rollout_examples]
        ground_truths = [get_ground_truth(example) for example in rollout_examples]
        repeated_prompts = repeat_each(prompts, group_size)
        repeated_ground_truths = repeat_each(ground_truths, group_size)

        logger.info("Step %d: generating %d rollout responses", step_idx, rollout_batch_size)
        rollout_outputs = vllm_model.generate(repeated_prompts, sampling_params)
        rollout_responses = [
            normalize_response_for_reward(output.outputs[0].text)
            for output in rollout_outputs
        ]
        del vllm_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        reward_infos = [
            r1_zero_reward_fn(response, ground_truth)
            for response, ground_truth in zip(rollout_responses, repeated_ground_truths)
        ]
        advantages_1d, raw_rewards_1d, reward_metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )

        tokenized = tokenize_prompt_and_output(
            prompt_strs=repeated_prompts,
            output_strs=rollout_responses,
            tokenizer=tokenizer,
        )
        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        response_mask = tokenized["response_mask"].to(device)
        raw_rewards = raw_rewards_1d.unsqueeze(-1).to(device)
        advantages = advantages_1d.unsqueeze(-1).to(device)

        policy.to(device)
        with torch.no_grad():
            old_outputs = get_response_log_probs(
                model=policy,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True,
            )
            old_log_probs = old_outputs["log_probs"].detach()
            token_entropy = old_outputs["token_entropy"].detach()

        optimizer.zero_grad(set_to_none=True)
        update_losses: list[float] = []
        clip_fractions: list[float] = []

        for epoch_idx in range(epochs_per_rollout_batch):
            minibatch_indices = iterate_minibatch_indices(
                total_size=rollout_batch_size,
                batch_size=micro_train_batch_size,
                rng=rng,
            )

            for microbatch_idx, batch_indices in enumerate(minibatch_indices):
                batch_indices = batch_indices.to(device)
                batch_input_ids = input_ids[batch_indices]
                batch_labels = labels[batch_indices]
                batch_response_mask = response_mask[batch_indices]
                batch_raw_rewards = raw_rewards[batch_indices]
                batch_advantages = advantages[batch_indices]

                policy_outputs = get_response_log_probs(
                    model=policy,
                    input_ids=batch_input_ids,
                    labels=batch_labels,
                    return_token_entropy=False,
                )
                batch_old_log_probs = None
                if loss_type == "grpo_clip":
                    batch_old_log_probs = old_log_probs[batch_indices].detach()

                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs=policy_outputs["log_probs"],
                    response_mask=batch_response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    raw_rewards=batch_raw_rewards,
                    advantages=batch_advantages,
                    old_log_probs=batch_old_log_probs,
                    cliprange=0.2 if loss_type == "grpo_clip" else None,
                )
                update_losses.append(float(loss.detach().cpu()))

                if "was_clipped" in metadata:
                    clipped = metadata["was_clipped"] * batch_response_mask
                    clip_fraction = clipped.sum() / batch_response_mask.sum()
                    clip_fractions.append(float(clip_fraction.detach().cpu()))

                is_update_boundary = (microbatch_idx + 1) % gradient_accumulation_steps == 0
                if is_update_boundary:
                    grad_norm = clip_grad_norm_(policy.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    mean_update_loss = mean(update_losses[-gradient_accumulation_steps:])
                    mean_reward = mean(reward["reward"] for reward in reward_infos)
                    mean_format_reward = mean(reward["format_reward"] for reward in reward_infos)
                    mean_answer_reward = mean(reward["answer_reward"] for reward in reward_infos)
                    mean_entropy = float(
                        (token_entropy * response_mask).sum().detach().cpu()
                        / response_mask.sum().detach().cpu()
                    )
                    mean_clip_fraction = (
                        None
                        if not clip_fractions
                        else mean(clip_fractions[-gradient_accumulation_steps:])
                    )

                    logger.info(
                        "step=%d epoch=%d update_loss=%.6f grad_norm=%.6f reward=%.4f format=%.4f answer=%.4f entropy=%.4f clip_frac=%s",
                        step_idx,
                        epoch_idx,
                        mean_update_loss,
                        float(grad_norm.detach().cpu()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                        mean_reward,
                        mean_format_reward,
                        mean_answer_reward,
                        mean_entropy,
                        "n/a" if mean_clip_fraction is None else f"{mean_clip_fraction:.6f}",
                    )

                    if use_wandb:
                        wandb.log(
                            {
                                "train/step": step_idx,
                                "train/epoch": epoch_idx,
                                "train/loss": mean_update_loss,
                                "train/grad_norm": float(grad_norm.detach().cpu())
                                if isinstance(grad_norm, torch.Tensor)
                                else float(grad_norm),
                                "train/reward": mean_reward,
                                "train/format_reward": mean_format_reward,
                                "train/answer_reward": mean_answer_reward,
                                "train/token_entropy": mean_entropy,
                                "train/reward_mean_batch": reward_metadata["reward_mean"],
                                "train/reward_std_batch": reward_metadata["reward_std"],
                                **(
                                    {}
                                    if mean_clip_fraction is None
                                    else {"train/clip_fraction": mean_clip_fraction}
                                ),
                            }
                        )

        save_training_checkpoint(
            checkpoint_dir=latest_policy_dir,
            policy=policy,
            tokenizer_source_path=model_name_or_path,
            optimizer=optimizer,
            step_idx=step_idx + 1,
        )
        policy.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if checkpoint_interval > 0 and (step_idx + 1) % checkpoint_interval == 0:
            save_training_checkpoint(
                checkpoint_dir=output_dir / f"checkpoint_step_{step_idx + 1:04d}",
                policy=policy,
                tokenizer_source_path=model_name_or_path,
                optimizer=optimizer,
                step_idx=step_idx + 1,
            )

        if (step_idx + 1) % validation_eval_interval == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            vllm_model = initialize_vllm_model(
                model_path=str(latest_policy_dir),
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
            validation_summary = evaluate_validation_rewards(
                vllm_model=vllm_model,
                prompt_template=prompt_template,
                validation_examples=validation_dataset,
                sampling_params=sampling_params,
                num_examples=min(validation_num_examples, len(validation_dataset)),
            )
            del vllm_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(
                "validation step=%d reward=%.4f format=%.4f answer=%.4f",
                step_idx,
                validation_summary["reward_mean"],
                validation_summary["format_reward_mean"],
                validation_summary["answer_reward_mean"],
            )
            if use_wandb:
                wandb.log(
                    {
                        "validation/step": step_idx,
                        "validation/reward": validation_summary["reward_mean"],
                        "validation/format_reward": validation_summary["format_reward_mean"],
                        "validation/answer_reward": validation_summary["answer_reward_mean"],
                    }
                )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    app()
