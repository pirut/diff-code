from __future__ import annotations

import os
import random
from datetime import UTC, datetime
from pathlib import Path

import modal
import torch

from code_diffusion.config import load_config
from code_diffusion.data import CodeDiffusionDataset
from code_diffusion.evaluation import load_cases_file
from code_diffusion.models import load_diffusion_model
from code_diffusion.training import run_preflight_batch, train

APP_NAME = os.environ.get("CODE_DIFFUSION_MODAL_APP", "code-diffusion-train")
DATA_VOLUME_NAME = os.environ.get("CODE_DIFFUSION_MODAL_DATA_VOLUME", "code-diffusion-data")
OUTPUT_VOLUME_NAME = os.environ.get(
    "CODE_DIFFUSION_MODAL_OUTPUTS_VOLUME",
    "code-diffusion-outputs",
)
CACHE_VOLUME_NAME = os.environ.get(
    "CODE_DIFFUSION_MODAL_CACHE_VOLUME",
    "code-diffusion-hf-cache",
)
GPU_TYPE = os.environ.get("CODE_DIFFUSION_MODAL_GPU", "A100-80GB")
CPU_COUNT = float(os.environ.get("CODE_DIFFUSION_MODAL_CPU", "8"))
MEMORY_MB = int(os.environ.get("CODE_DIFFUSION_MODAL_MEMORY_MB", str(64 * 1024)))
TIMEOUT_SECONDS = int(os.environ.get("CODE_DIFFUSION_MODAL_TIMEOUT_SECONDS", str(24 * 60 * 60)))
HF_SECRET_NAME = os.environ.get("CODE_DIFFUSION_MODAL_HF_SECRET") or "code-diffusion-hf"

REMOTE_DATA_ROOT = Path("/mnt/data")
REMOTE_OUTPUT_ROOT = Path("/mnt/outputs")
REMOTE_CACHE_ROOT = Path("/mnt/hf")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.7.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install_from_requirements("requirements-modal.txt")
    .env(
        {
            "HF_HOME": str(REMOTE_CACHE_ROOT),
            "HF_HUB_CACHE": str(REMOTE_CACHE_ROOT / "hub"),
            "TRANSFORMERS_CACHE": str(REMOTE_CACHE_ROOT / "hub"),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("code_diffusion")
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

function_kwargs = {
    "image": image,
    "gpu": GPU_TYPE,
    "cpu": CPU_COUNT,
    "memory": MEMORY_MB,
    "timeout": TIMEOUT_SECONDS,
    "volumes": {
        str(REMOTE_DATA_ROOT): data_volume.read_only(),
        str(REMOTE_OUTPUT_ROOT): output_volume,
        str(REMOTE_CACHE_ROOT): cache_volume,
    },
}
if HF_SECRET_NAME:
    function_kwargs["secrets"] = [modal.Secret.from_name(HF_SECRET_NAME)]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_overrides(raw_overrides: str) -> list[str]:
    return [item for item in raw_overrides.split(";;") if item]


def _timestamped_run_name() -> str:
    return datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")


def _prepare_remote_config(
    *,
    config_path: str,
    overrides: list[str],
    run_name: str,
) -> tuple[dict, dict, Path, str]:
    local_config = load_config(config_path, overrides=overrides)
    local_data_dir = Path(local_config["data_dir"]).resolve()
    remote_data_dir = REMOTE_DATA_ROOT / local_data_dir.name
    remote_output_dir = REMOTE_OUTPUT_ROOT / run_name

    remote_config = dict(local_config)
    remote_config["data_dir"] = str(remote_data_dir)
    remote_config["output_dir"] = str(remote_output_dir)
    remote_config["device"] = "cuda"
    remote_config.setdefault("dtype", "bfloat16")
    remote_config.setdefault("attn_implementation", "sdpa")

    return local_config, remote_config, local_data_dir, str(remote_data_dir)


def _find_latest_resume_checkpoint(output_dir: Path) -> Path | None:
    latest_step = -1
    latest_dir: Path | None = None
    if not output_dir.exists():
        return None
    for candidate in output_dir.glob("step-*"):
        if not candidate.is_dir():
            continue
        trainer_state_path = candidate / "trainer_state.pt"
        adapter_config_path = candidate / "adapter_config.json"
        if not trainer_state_path.exists() or not adapter_config_path.exists():
            continue
        try:
            step = int(candidate.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if step > latest_step:
            latest_step = step
            latest_dir = candidate
    return latest_dir


def _apply_resume_config(config: dict) -> tuple[dict, Path | None]:
    if not bool(config.get("resume_from_checkpoint", True)):
        return dict(config), None

    output_dir = Path(config["output_dir"])
    checkpoint_dir = _find_latest_resume_checkpoint(output_dir)
    if checkpoint_dir is None:
        return dict(config), None

    resumed = dict(config)
    resumed["model_name"] = str(checkpoint_dir)
    resumed["resume_checkpoint_dir"] = str(checkpoint_dir)
    return resumed, checkpoint_dir


def _build_dataset(
    *,
    tokenizer,
    config: dict,
    max_files: int | None = None,
    max_samples: int | None = None,
) -> CodeDiffusionDataset:
    return CodeDiffusionDataset(
        tokenizer=tokenizer,
        data_dir=config["data_dir"],
        seq_length=config["seq_length"],
        extensions=config["extensions"],
        mask_ratio_min=config["mask_ratio_min"],
        mask_ratio_max=config["mask_ratio_max"],
        max_files=max_files,
        max_samples=max_samples,
        config=config,
    )


@app.function(**function_kwargs)
def train_remote(
    config: dict,
    benchmark_cases: list[dict[str, object]] | None = None,
    benchmark_options: dict[str, object] | None = None,
) -> dict[str, float | str]:
    _set_seed(int(config.get("seed", 7)))

    runtime_config, resume_checkpoint = _apply_resume_config(config)
    if resume_checkpoint is not None:
        print(f"resuming_from_checkpoint={resume_checkpoint}")

    model, tokenizer = load_diffusion_model(runtime_config)
    dataset = _build_dataset(tokenizer=tokenizer, config=runtime_config)
    dataset.export_summary(runtime_config["output_dir"])
    metrics = train(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=runtime_config,
        benchmark_cases=benchmark_cases,
        benchmark_options=benchmark_options,
        checkpoint_callback=lambda _checkpoint_dir, _benchmark_result: output_volume.commit(),
    )

    output_volume.commit()
    cache_volume.commit()

    return {
        "run_dir": str(Path(runtime_config["output_dir"])),
        "final_loss": float(metrics["final_loss"]),
        "final_masked_accuracy": float(metrics["final_masked_accuracy"]),
        "best_benchmark_score": float(metrics.get("best_benchmark_score", 0.0)),
        "best_checkpoint": str(metrics.get("best_checkpoint", "")),
    }


@app.function(**function_kwargs)
def preflight_remote(
    config: dict,
    dataset_max_files: int = 4,
    dataset_max_samples: int = 4,
) -> dict[str, float | int | str]:
    _set_seed(int(config.get("seed", 7)))

    model, tokenizer = load_diffusion_model(config)
    dataset = _build_dataset(
        tokenizer=tokenizer,
        config=config,
        max_files=dataset_max_files,
        max_samples=dataset_max_samples,
    )
    dataset.export_summary(config["output_dir"])
    metrics = run_preflight_batch(model=model, dataset=dataset, config=config)

    output_volume.commit()
    cache_volume.commit()

    return {
        "status": "ok",
        "model_name": str(config["model_name"]),
        "data_dir": str(config["data_dir"]),
        "source_files": int(dataset.num_source_files),
        "dataset_samples": int(len(dataset)),
        **metrics,
    }


@app.local_entrypoint()
def main(
    config: str = "config.yaml",
    mode: str = "train",
    run_name: str = "",
    overrides: str = "",
    skip_upload: bool = False,
    preflight_max_files: int = 4,
    preflight_max_samples: int = 4,
    benchmark_cases: str = "",
    benchmark_steps: int = 0,
    benchmark_temperature: float = 0.0,
    benchmark_top_k: int = 0,
    benchmark_top_p: float = 1.0,
    benchmark_show_samples: int = 1,
    benchmark_output_subdir: str = "benchmarks",
):
    override_list = _parse_overrides(overrides)
    effective_run_name = run_name or _timestamped_run_name()
    local_config, remote_config, local_data_dir, remote_data_dir = _prepare_remote_config(
        config_path=config,
        overrides=override_list,
        run_name=effective_run_name,
    )
    loaded_benchmark_cases = load_cases_file(benchmark_cases) if benchmark_cases else None
    benchmark_options = None
    if loaded_benchmark_cases:
        benchmark_options = {
            "cases_label": str(Path(benchmark_cases).resolve()),
            "steps": benchmark_steps or remote_config["diffusion_steps"],
            "temperature": benchmark_temperature,
            "top_k": benchmark_top_k,
            "top_p": benchmark_top_p,
            "show_samples": benchmark_show_samples,
            "output_subdir": benchmark_output_subdir,
        }

    if not skip_upload:
        # batch_upload writes into the volume's own filesystem root, not the mount path.
        volume_data_dir = f"/{Path(remote_data_dir).name}"
        with data_volume.batch_upload(force=True) as batch:
            batch.put_directory(str(local_data_dir), volume_data_dir)

    if mode == "preflight":
        result = preflight_remote.remote(
            remote_config,
            dataset_max_files=preflight_max_files,
            dataset_max_samples=preflight_max_samples,
        )
    elif mode == "train":
        result = train_remote.remote(
            remote_config,
            benchmark_cases=loaded_benchmark_cases,
            benchmark_options=benchmark_options,
        )
        if loaded_benchmark_cases:
            _download_remote_subdir(
                volume=output_volume,
                remote_subdir=f"{effective_run_name}/{benchmark_output_subdir}",
                local_destination=Path(local_config["output_dir"]) / effective_run_name / benchmark_output_subdir,
            )
    else:
        raise ValueError("mode must be 'preflight' or 'train'")

    print(result)


def _download_remote_subdir(*, volume: modal.Volume, remote_subdir: str, local_destination: Path) -> None:
    try:
        entries = volume.listdir(remote_subdir, recursive=True)
    except FileNotFoundError:
        return

    files = [entry for entry in entries if getattr(entry.type, "name", "") != "DIRECTORY"]
    if not files:
        return

    local_destination.mkdir(parents=True, exist_ok=True)
    for entry in files:
        relative = Path(entry.path).relative_to(remote_subdir)
        destination = local_destination / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            volume.read_file_into_fileobj(entry.path, handle)
