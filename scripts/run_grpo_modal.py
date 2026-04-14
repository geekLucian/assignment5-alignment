from __future__ import annotations

import os
from pathlib import Path

import modal


APP_NAME = "grpo-train"
REMOTE_ROOT = "/root/project"
REMOTE_OUTPUT_DIR = "/outputs/grpo"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers>=4.50.0",
        "accelerate>=1.5.2",
        "datasets>=2.14.6",
        "tqdm>=4.67.1",
        "xopen>=2.0.2",
        "math-verify[antlr4-13-2]>=0.7.0",
        "pylatexenc==2.10",
        "wandb>=0.19.8",
        "typer>=0.15.4",
        "vllm==0.7.2",
    )
    .add_local_dir("cs336_alignment", remote_path=f"{REMOTE_ROOT}/cs336_alignment")
    .add_local_dir("scripts", remote_path=f"{REMOTE_ROOT}/scripts")
)

outputs_volume = modal.Volume.from_name("grpo-outputs", create_if_missing=True)
wandb_secret = modal.Secret.from_dict({"WANDB_API_KEY": os.environ.get("WANDB_API_KEY")})


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 24,
    volumes={"/outputs": outputs_volume},
    secrets=[wandb_secret],
)
def run_grpo_remote(
    model_name_or_path: str,
    train_split: str = "train",
    validation_split: str = "test",
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.85,
    loss_type: str = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    validation_eval_interval: int = 10,
    validation_num_examples: int = 1024,
    seed: int = 42,
    max_model_len: int = 2048,
    checkpoint_interval: int = 10,
    use_wandb: bool = False,
    wandb_project: str = "cs336-grpo",
    wandb_run_name: str | None = None,
) -> dict[str, str]:
    import importlib.util
    import os
    import sys

    os.chdir(REMOTE_ROOT)
    sys.path.insert(0, REMOTE_ROOT)
    module_path = Path(REMOTE_ROOT) / "scripts" / "run_grpo.py"
    spec = importlib.util.spec_from_file_location("run_grpo_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load GRPO script from {module_path}")
    run_grpo_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_grpo_module)
    run_grpo_main = run_grpo_module.main

    run_grpo_main(
        model_name_or_path=model_name_or_path,
        train_split=train_split,
        validation_split=validation_split,
        n_grpo_steps=n_grpo_steps,
        learning_rate=learning_rate,
        advantage_eps=advantage_eps,
        rollout_batch_size=rollout_batch_size,
        group_size=group_size,
        sampling_temperature=sampling_temperature,
        sampling_min_tokens=sampling_min_tokens,
        sampling_max_tokens=sampling_max_tokens,
        epochs_per_rollout_batch=epochs_per_rollout_batch,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gpu_memory_utilization=gpu_memory_utilization,
        loss_type=loss_type,
        use_std_normalization=use_std_normalization,
        validation_eval_interval=validation_eval_interval,
        validation_num_examples=validation_num_examples,
        seed=seed,
        max_model_len=max_model_len,
        output_dir=Path(REMOTE_OUTPUT_DIR),
        checkpoint_interval=checkpoint_interval,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )
    outputs_volume.commit()
    return {"output_dir": REMOTE_OUTPUT_DIR}


@app.local_entrypoint()
def main(
    model_name_or_path: str,
    train_split: str = "train",
    validation_split: str = "test",
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.85,
    loss_type: str = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    validation_eval_interval: int = 10,
    validation_num_examples: int = 1024,
    seed: int = 42,
    max_model_len: int = 2048,
    checkpoint_interval: int = 10,
    use_wandb: bool = False,
    wandb_project: str = "cs336-grpo",
    wandb_run_name: str | None = None,
) -> None:
    summary = run_grpo_remote.remote(
        model_name_or_path=model_name_or_path,
        train_split=train_split,
        validation_split=validation_split,
        n_grpo_steps=n_grpo_steps,
        learning_rate=learning_rate,
        advantage_eps=advantage_eps,
        rollout_batch_size=rollout_batch_size,
        group_size=group_size,
        sampling_temperature=sampling_temperature,
        sampling_min_tokens=sampling_min_tokens,
        sampling_max_tokens=sampling_max_tokens,
        epochs_per_rollout_batch=epochs_per_rollout_batch,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gpu_memory_utilization=gpu_memory_utilization,
        loss_type=loss_type,
        use_std_normalization=use_std_normalization,
        validation_eval_interval=validation_eval_interval,
        validation_num_examples=validation_num_examples,
        seed=seed,
        max_model_len=max_model_len,
        checkpoint_interval=checkpoint_interval,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )
    print(summary)
