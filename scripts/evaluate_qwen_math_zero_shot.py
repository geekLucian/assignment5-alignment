"""
Evaluate a vLLM-served model zero-shot with the R1-Zero prompt on the
Countdown dataset and serialize generations plus metrics for later analysis.

Example:

```bash
uv run python scripts/evaluate_qwen_math_zero_shot.py \
    --model-name-or-path Qwen/Qwen2.5-Math-1.5B \
    --output-path outputs/qwen2.5-math-1.5b-countdown-r1-zero.jsonl \
    --summary-path outputs/qwen2.5-math-1.5b-countdown-r1-zero-summary.json
```
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from xopen import xopen

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

logger = logging.getLogger(__name__)

DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
DEFAULT_MODEL = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_MAX_EXAMPLES = 5000
DEFAULT_SEED = 42
PROMPT_PATH = Path(__file__).resolve().parents[1] / "cs336_alignment" / "prompts" / "r1_zero.prompt"


def load_prompt_template(prompt_path: Path) -> str:
    return prompt_path.read_text().strip()


def format_countdown_question(example: dict[str, Any]) -> str:
    if "question" in example and example["question"]:
        return str(example["question"])
    if "problem" in example and example["problem"]:
        return str(example["problem"])
    if "prompt" in example and example["prompt"]:
        return str(example["prompt"])

    numbers = None
    for key in ("nums", "numbers", "operands"):
        if key in example and example[key] is not None:
            numbers = example[key]
            break

    target = None
    for key in ("target", "answer", "value"):
        if key in example and example[key] is not None:
            target = example[key]
            break

    if numbers is not None and target is not None:
        joined_numbers = ", ".join(str(num) for num in numbers)
        return (
            "Use the numbers "
            f"{joined_numbers} exactly once with arithmetic operations to make {target}. "
            "Give the final expression."
        )

    raise KeyError(
        "Could not infer a question field for the dataset example. "
        f"Available keys: {sorted(example.keys())}"
    )


def get_ground_truth(example: dict[str, Any]) -> str | float | int | list[str]:
    if "answer" in example and isinstance(example["answer"], str):
        match = re.search(r"####\s*(.+?)\s*$", example["answer"], re.DOTALL)
        if match is not None:
            final_answer = match.group(1).strip()
            final_answer = final_answer.replace("$", "").replace(",", "")
            return final_answer
    for key in ("answer", "answers", "solution", "solutions", "target"):
        if key in example and example[key] is not None:
            return example[key]
    raise KeyError(
        "Could not infer a ground-truth field for the dataset example. "
        f"Available keys: {sorted(example.keys())}"
    )


def build_prompt(question: str, prompt_template: str) -> str:
    return prompt_template.format(question=question)


def extract_final_number(text: str) -> str:
    normalized = text.replace(",", "")
    matches = re.findall(r"-?\$?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", normalized)
    if not matches:
        return text.strip()
    return matches[-1].replace("$", "").strip()


def normalize_response_for_reward(response: str) -> str:
    if "<answer>" not in response or "</answer>" not in response:
        return response
    prefix, suffix = response.rsplit("<answer>", 1)
    answer_text, tail = suffix.split("</answer>", 1)
    normalized_answer = extract_final_number(answer_text)
    return f"{prefix}<answer> {normalized_answer} </answer>{tail}"


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str | float | int | list[str]],
    eval_sampling_params: SamplingParams,
    output_path: str,
    examples: list[dict[str, Any]],
    model_name_or_path: str,
    summary_path: str | None = None,
) -> dict[str, float]:
    """
    Evaluate a language model on a list of prompts, compute evaluation metrics,
    and serialize results to disk.
    """
    if not (len(prompts) == len(ground_truths) == len(examples)):
        raise ValueError("prompts, ground_truths, and examples must have the same length")

    raw_outputs = vllm_model.generate(prompts, eval_sampling_params)
    all_metrics: list[dict[str, float]] = []

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with xopen(output_path_obj, "w") as fout:
        for example, prompt, ground_truth, raw_output in tqdm(
            zip(examples, prompts, ground_truths, raw_outputs),
            total=len(prompts),
            desc="Scoring generations",
        ):
            generation = raw_output.outputs[0].text
            normalized_generation = normalize_response_for_reward(generation)
            metrics = reward_fn(normalized_generation, ground_truth)
            all_metrics.append(metrics)

            fout.write(
                json.dumps(
                    {
                        "model_name_or_path": model_name_or_path,
                        "prompt": prompt,
                        "generation": generation,
                        "normalized_generation": normalized_generation,
                        "ground_truth": ground_truth,
                        "metrics": metrics,
                        "example": example,
                    }
                )
                + "\n"
            )

    summary = {
        key: mean(metrics[key] for metrics in all_metrics)
        for key in sorted(all_metrics[0].keys())
    }
    summary["num_examples"] = float(len(all_metrics))

    if summary_path is not None:
        summary_path_obj = Path(summary_path)
        summary_path_obj.parent.mkdir(parents=True, exist_ok=True)
        summary_path_obj.write_text(json.dumps(summary, indent=2) + "\n")

    return summary


def main(
    model_name_or_path: str,
    output_path: str,
    summary_path: str | None,
    split: str,
    num_gpus: int,
    max_examples: int | None,
    seed: int,
    max_model_len: int,
) -> None:
    logger.info("Loading prompt template from %s", PROMPT_PATH)
    prompt_template = load_prompt_template(PROMPT_PATH)

    logger.info("Loading dataset %s/%s [%s]", DATASET_NAME, DATASET_CONFIG, split)
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    examples = [dict(example) for example in dataset]
    if max_examples is not None:
        rng = random.Random(seed)
        sample_size = min(max_examples, len(examples))
        examples = rng.sample(examples, sample_size)
    logger.info("Loaded %d examples", len(examples))

    prompts = []
    ground_truths = []
    for example in examples:
        question = format_countdown_question(example)
        prompts.append(build_prompt(question, prompt_template))
        ground_truths.append(get_ground_truth(example))

    logger.info("Initializing vLLM model %s", model_name_or_path)
    vllm_model = LLM(
        model=model_name_or_path,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
        max_model_len=max_model_len,
    )
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    summary = evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        output_path=output_path,
        examples=examples,
        model_name_or_path=model_name_or_path,
        summary_path=summary_path,
    )

    for key, value in summary.items():
        logger.info("%s: %.6f", key, value)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=DEFAULT_MODEL,
        help="HF model name or local path to evaluate with vLLM.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to write per-example generations and metrics as JSONL.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Optional path to write aggregate metrics as JSON.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=DEFAULT_MAX_EXAMPLES,
        help="Number of randomly sampled examples to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used when sampling examples.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length passed to vLLM.",
    )
    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))
    main(
        model_name_or_path=args.model_name_or_path,
        output_path=args.output_path,
        summary_path=args.summary_path,
        split=args.split,
        num_gpus=args.num_gpus,
        max_examples=args.max_examples,
        seed=args.seed,
        max_model_len=args.max_model_len,
    )
    logger.info("finished running %s", sys.argv[0])
