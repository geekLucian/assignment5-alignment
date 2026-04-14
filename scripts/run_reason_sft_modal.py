from __future__ import annotations

from pathlib import Path

import modal


APP_NAME = "reason-sft-sweep"
REMOTE_ROOT = "/root/project"
REMOTE_OUTPUT_DIR = "/outputs/reason_sft_sweep"

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
        "matplotlib",
    )
    .add_local_dir("cs336_alignment", remote_path=f"{REMOTE_ROOT}/cs336_alignment")
    .add_local_dir("scripts", remote_path=f"{REMOTE_ROOT}/scripts")
    .add_local_dir(
        "sft-cs336-assign5-datasets/sft-reason",
        remote_path=f"{REMOTE_ROOT}/sft-cs336-assign5-datasets/sft-reason",
    )
)

outputs_volume = modal.Volume.from_name("reason-sft-sweep-outputs", create_if_missing=True)


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60 * 24,
    volumes={"/outputs": outputs_volume},
)
def run_reason_sft_remote() -> dict:
    import os
    import sys

    os.chdir(REMOTE_ROOT)
    sys.path.insert(0, REMOTE_ROOT)

    from scripts.run_reason_sft_sweep import ExperimentConfig, run_experiment

    config = ExperimentConfig(
        output_dir=Path(REMOTE_OUTPUT_DIR),
        dataset_sizes=[128, 256, 512, 1024],
        include_full=True,
        learning_rates=[2e-5, 5e-5],
        effective_batch_sizes=[64],
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        num_epochs=1,
        max_length=1024,
        max_new_tokens=256,
        num_trainable_layers=8,
        log_every=50,
        local_files_only=False,
    )
    summary = run_experiment(config)
    outputs_volume.commit()
    return summary


@app.local_entrypoint()
def main() -> None:
    summary = run_reason_sft_remote.remote()
    print(summary)
