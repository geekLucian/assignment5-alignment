from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import Adafactor, AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import tokenize_prompt_and_output


MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
PROMPT_PATH = Path("sft-cs336-assign5-datasets/sft-reason/r1_zero.prompt")
TRAIN_PATH = Path("sft-cs336-assign5-datasets/sft-reason/sft_gpt-oss-120b_filtered.jsonl")
VAL_PATH = Path("sft-cs336-assign5-datasets/sft-reason/val.jsonl")


@dataclass
class ExperimentConfig:
    model_name_or_path: str = MODEL_NAME
    train_path: Path = TRAIN_PATH
    val_path: Path = VAL_PATH
    prompt_path: Path = PROMPT_PATH
    output_dir: Path = Path("outputs/reason_sft_sweep")
    dataset_sizes: list[int] | None = None
    include_full: bool = False
    learning_rates: list[float] | None = None
    effective_batch_sizes: list[int] | None = None
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 8
    num_epochs: int = 1
    max_length: int = 768
    max_new_tokens: int = 256
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    seed: int = 42
    eval_max_examples: int | None = None
    num_trainable_layers: int = 4
    log_every: int = 100
    local_files_only: bool = True

    def __post_init__(self) -> None:
        if self.dataset_sizes is None:
            self.dataset_sizes = [128, 256, 512, 1024]
        if self.learning_rates is None:
            self.learning_rates = [2e-5]
        if self.effective_batch_sizes is None:
            self.effective_batch_sizes = [16]


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default=MODEL_NAME)
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=VAL_PATH)
    parser.add_argument("--prompt-path", type=Path, default=PROMPT_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reason_sft_sweep"))
    parser.add_argument("--dataset-sizes", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[2e-5])
    parser.add_argument("--effective-batch-sizes", type=int, nargs="+", default=[16])
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-max-examples", type=int, default=None)
    parser.add_argument("--num-trainable-layers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return ExperimentConfig(**vars(parser.parse_args()))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected {path} to contain a JSON array")
    return data


def build_prompt(question: str, prompt_template: str) -> str:
    return prompt_template.format(question=question)


def truncate_example(
    tokenizer: AutoTokenizer,
    prompt: str,
    response: str,
    max_length: int,
) -> tuple[str, str]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    max_response_len = max_length - len(prompt_ids)
    if max_response_len <= 0:
        return prompt, ""
    if len(response_ids) <= max_response_len:
        return prompt, response
    truncated_response = tokenizer.decode(
        response_ids[:max_response_len],
        skip_special_tokens=False,
    )
    return prompt, truncated_response


def prepare_train_examples(
    raw_examples: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    prompt_template: str,
    max_length: int,
) -> list[dict[str, str]]:
    prepared = []
    for example in raw_examples:
        prompt = build_prompt(example["problem"], prompt_template)
        response = example["reasoning_trace"]
        prompt, response = truncate_example(tokenizer, prompt, response, max_length)
        prepared.append({"prompt": prompt, "response": response})
    return prepared


def make_train_collate_fn(tokenizer: AutoTokenizer):
    def collate_fn(examples: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        batch = tokenize_prompt_and_output(
            prompt_strs=[example["prompt"] for example in examples],
            output_strs=[example["response"] for example in examples],
            tokenizer=tokenizer,
        )
        return batch

    return collate_fn


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def build_model_and_tokenizer(
    model_name_or_path: str,
    local_files_only: bool,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model, tokenizer


def select_trainable_parameters(
    model: AutoModelForCausalLM,
    num_trainable_layers: int,
) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_parameters: list[torch.nn.Parameter] = []
    layers = getattr(model.model, "layers", None)
    if layers is None:
        raise AttributeError("Expected model.model.layers for Qwen-style model")

    if num_trainable_layers > 0:
        for layer in layers[-num_trainable_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
                trainable_parameters.append(parameter)

    for module_name in ("norm",):
        module = getattr(model.model, module_name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True
                trainable_parameters.append(parameter)

    for parameter in model.lm_head.parameters():
        parameter.requires_grad = True
        trainable_parameters.append(parameter)

    return trainable_parameters


def train_one_run(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_examples: list[dict[str, str]],
    learning_rate: float,
    effective_batch_size: int,
    per_device_train_batch_size: int,
    num_epochs: int,
    weight_decay: float,
    warmup_ratio: float,
    seed: int,
    num_trainable_layers: int,
    log_every: int,
) -> dict[str, float]:
    if effective_batch_size % per_device_train_batch_size != 0:
        raise ValueError("effective_batch_size must be divisible by per_device_train_batch_size")

    gradient_accumulation_steps = effective_batch_size // per_device_train_batch_size
    device = model.device
    collate_fn = make_train_collate_fn(tokenizer)
    train_loader = DataLoader(
        train_examples,
        batch_size=per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )

    trainable_parameters = select_trainable_parameters(model, num_trainable_layers)
    optimizer = Adafactor(
        trainable_parameters,
        lr=learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=weight_decay,
    )

    total_optimizer_steps = math.ceil(len(train_loader) * num_epochs / gradient_accumulation_steps)
    warmup_steps = int(total_optimizer_steps * warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps == 0:
            return 1.0
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    optimizer_steps = 0

    for _epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            batch = move_to_device(batch, device)
            attention_mask = (batch["input_ids"] != tokenizer.pad_token_id).long()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            token_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=batch["labels"].unsqueeze(-1),
            ).squeeze(-1)

            response_mask = batch["response_mask"].float()
            denom = response_mask.sum().clamp_min(1.0)
            loss = -((token_log_probs * response_mask).sum() / denom)
            (loss / gradient_accumulation_steps).backward()
            losses.append(loss.item())

            if (step + 1) % log_every == 0:
                print(
                    json.dumps(
                        {
                            "stage": "train",
                            "micro_step": step + 1,
                            "num_micro_steps": len(train_loader),
                            "loss": loss.item(),
                        }
                    ),
                    flush=True,
                )

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

        if len(train_loader) % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

    return {
        "train_loss": mean(losses),
        "optimizer_steps": float(optimizer_steps),
        "trainable_parameters_millions": float(
            sum(parameter.numel() for parameter in trainable_parameters) / 1_000_000
        ),
    }


def trim_response(text: str) -> str:
    if "</answer>" in text:
        end = text.index("</answer>") + len("</answer>")
        return text[:end]
    return text


def normalize_ground_truth(
    ground_truth: str | float | int | list[str | float | int],
) -> str | list[str]:
    if isinstance(ground_truth, list):
        return [str(item) for item in ground_truth]
    return str(ground_truth)


@torch.inference_mode()
def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    val_examples: list[dict[str, Any]],
    prompt_template: str,
    per_device_eval_batch_size: int,
    max_new_tokens: int,
) -> dict[str, float]:
    device = model.device
    was_training = model.training
    model.eval()

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    rewards = []
    answer_rewards = []
    format_rewards = []

    for start in range(0, len(val_examples), per_device_eval_batch_size):
        batch_examples = val_examples[start : start + per_device_eval_batch_size]
        prompts = [build_prompt(example["problem"], prompt_template) for example in batch_examples]
        tokenized = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)
        generated = model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_lengths = tokenized["attention_mask"].sum(dim=1).tolist()
        response_ids = [
            generated[i, prompt_length:]
            for i, prompt_length in enumerate(prompt_lengths)
        ]
        responses = [trim_response(text) for text in tokenizer.batch_decode(response_ids, skip_special_tokens=True)]
        for response, example in zip(responses, batch_examples):
            reward = r1_zero_reward_fn(
                response,
                normalize_ground_truth(example["expected_answer"]),
            )
            rewards.append(reward["reward"])
            answer_rewards.append(reward["answer_reward"])
            format_rewards.append(reward["format_reward"])

    tokenizer.padding_side = original_padding_side
    if was_training:
        model.train()

    return {
        "accuracy": mean(answer_rewards),
        "reward": mean(rewards),
        "format_accuracy": mean(format_rewards),
    }


def maybe_plot(results: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    sizes = [result["dataset_size"] for result in results]
    accuracies = [result["accuracy"] for result in results]

    plt.figure(figsize=(7, 4.5))
    plt.plot(sizes, accuracies, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("Training examples")
    plt.ylabel("Validation accuracy")
    plt.title("Reasoning SFT Validation Accuracy vs Dataset Size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "validation_accuracy_curve.png", dpi=180)


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = config.prompt_path.read_text().strip()
    raw_train_examples = load_json(config.train_path)
    raw_val_examples = load_json(config.val_path)
    if config.eval_max_examples is not None:
        raw_val_examples = raw_val_examples[: config.eval_max_examples]

    dataset_sizes = list(config.dataset_sizes)
    if config.include_full:
        dataset_sizes.append(len(raw_train_examples))
    dataset_sizes = list(dict.fromkeys(dataset_sizes))

    model, tokenizer = build_model_and_tokenizer(
        config.model_name_or_path,
        config.local_files_only,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prepared_train_examples = prepare_train_examples(
        raw_examples=raw_train_examples,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        max_length=config.max_length,
    )

    shuffled_train_examples = prepared_train_examples[:]
    random.Random(config.seed).shuffle(shuffled_train_examples)

    all_results = []
    best_full_result = None

    for dataset_size in dataset_sizes:
        train_subset = shuffled_train_examples[:dataset_size]
        for learning_rate in config.learning_rates:
            for effective_batch_size in config.effective_batch_sizes:
                model, tokenizer = build_model_and_tokenizer(
                    config.model_name_or_path,
                    config.local_files_only,
                )
                model.to(device)
                train_metrics = train_one_run(
                    model=model,
                    tokenizer=tokenizer,
                    train_examples=train_subset,
                    learning_rate=learning_rate,
                    effective_batch_size=effective_batch_size,
                    per_device_train_batch_size=config.per_device_train_batch_size,
                    num_epochs=config.num_epochs,
                    weight_decay=config.weight_decay,
                    warmup_ratio=config.warmup_ratio,
                    seed=config.seed,
                    num_trainable_layers=config.num_trainable_layers,
                    log_every=config.log_every,
                )
                eval_metrics = evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    val_examples=raw_val_examples,
                    prompt_template=prompt_template,
                    per_device_eval_batch_size=config.per_device_eval_batch_size,
                    max_new_tokens=config.max_new_tokens,
                )
                result = {
                    "dataset_size": dataset_size,
                    "learning_rate": learning_rate,
                    "effective_batch_size": effective_batch_size,
                    **train_metrics,
                    **eval_metrics,
                }
                print(json.dumps({"stage": "result", **result}), flush=True)
                all_results.append(result)
                result_path = config.output_dir / (
                    f"size_{dataset_size}_lr_{learning_rate:g}_bs_{effective_batch_size}.json"
                )
                result_path.write_text(json.dumps(result, indent=2) + "\n")

                if dataset_size == len(raw_train_examples):
                    if best_full_result is None or result["accuracy"] > best_full_result["accuracy"]:
                        best_full_result = result

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    summary = {
        "results": all_results,
        "best_full_result": best_full_result,
        "num_val_examples": len(raw_val_examples),
        "train_dataset_path": str(config.train_path),
        "val_dataset_path": str(config.val_path),
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    by_size_best = []
    for dataset_size in sorted({result["dataset_size"] for result in all_results}):
        candidates = [result for result in all_results if result["dataset_size"] == dataset_size]
        by_size_best.append(max(candidates, key=lambda result: result["accuracy"]))
    maybe_plot(by_size_best, config.output_dir)
    return summary


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
