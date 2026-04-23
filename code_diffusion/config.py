from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "model_name": "google/gemma-4-E4B",
    "data_dir": "./sample_corpus",
    "output_dir": "./outputs/code_diffusion",
    "seq_length": 256,
    "batch_size": 1,
    "train_steps": 50,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "mask_ratio_min": 0.1,
    "mask_ratio_max": 0.35,
    "warmup_steps": 20,
    "diffusion_steps": 8,
    "inference_mask_span": 16,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.95,
    "confidence_threshold": None,
    "remask_fraction": 0.5,
    "attention_mode": "full",
    "attn_implementation": "sdpa",
    "finetune_method": "qlora",
    "gradient_checkpointing": True,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "lora_bias": "none",
    "lora_target_modules": "all-linear",
    "qlora_quant_type": "nf4",
    "qlora_use_double_quant": True,
    "qlora_compute_dtype": "bfloat16",
    "task_type_weights": {
        "masked_reconstruction": 0.35,
        "fim": 0.20,
        "bug_fix": 0.20,
        "refinement": 0.15,
        "doc_test_based": 0.10,
    },
    "benchmark_controller_enabled": True,
    "benchmark_controller_score_weights": {
        "exact_match": 0.7,
        "similarity": 0.3,
    },
    "benchmark_controller_best_dirname": "best",
    "benchmark_controller_improvement_threshold": 1e-4,
    "benchmark_controller_plateau_patience": 2,
    "benchmark_controller_lr_decay_factor": 0.5,
    "benchmark_controller_min_learning_rate": 1e-5,
    "benchmark_controller_reweight_task_mix": True,
    "benchmark_controller_task_mix_strength": 1.0,
    "benchmark_controller_task_mix_momentum": 0.5,
    "benchmark_controller_min_task_weight": 0.05,
    "mask_strategy_weights": {
        "random": 0.30,
        "span": 0.25,
        "structure": 0.25,
        "smart": 0.20,
    },
    "mask_span_min_tokens": 3,
    "mask_span_max_tokens": 50,
    "random_mask_ratio_min": 0.10,
    "random_mask_ratio_max": 0.40,
    "structure_mask_features": {
        "function_bodies": True,
        "full_lines": True,
        "return_statements": True,
        "argument_lists": True,
    },
    "smart_mask_weights": {
        "function_logic": 1.0,
        "conditionals": 0.9,
        "return_expressions": 1.0,
        "argument_lists": 0.6,
        "full_lines": 0.4,
    },
    "deduplicate_dataset": True,
    "validate_python": True,
    "filter_generated_code": True,
    "filter_minified_code": True,
    "filter_boilerplate": True,
    "max_file_size_bytes": 200_000,
    "max_line_length": 400,
    "max_code_lines": 2_000,
    "dataset_preview_examples": 5,
    "prepared_examples_filename": "prepared_examples.jsonl",
    "prepare_output_dir": "./outputs/prepared_corpus/public_mix_v1",
    "prepare_include_local_data": True,
    "prepare_include_codesearchnet": True,
    "prepare_include_commitpackft": True,
    "prepare_include_swe_rebench": True,
    "prepare_languages": ["python", "javascript", "typescript", "go", "rust"],
    "prepare_codesearchnet_examples_per_language": 200,
    "prepare_commitpack_examples_per_language": 200,
    "prepare_swe_rebench_examples_per_language": 75,
    "prepare_max_local_files": 200,
    "prepare_context_lines": 24,
    "prepare_max_example_chars": 12000,
    "prepare_include_commit_context": True,
    "prepare_include_issue_context": True,
    "prepare_include_repo_context": True,
    "prepare_context_max_lines": 8,
    "synthetic_generation_enabled": False,
    "synthetic_provider": None,
    "synthetic_model": None,
    "synthetic_base_url": "https://api.openai.com/v1",
    "synthetic_api_key_env": "OPENAI_API_KEY",
    "synthetic_timeout_seconds": 60,
    "synthetic_cache_dir": "./outputs/synthetic_cache",
    "extensions": [
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".java",
        ".go",
        ".rs",
        ".cpp",
        ".c",
        ".h",
        ".json",
        ".yaml",
        ".yml",
    ],
    "num_workers": 0,
    "seed": 7,
    "device": "auto",
    "trust_remote_code": False,
    "log_every": 5,
    "save_every": 25,
    "resume_from_checkpoint": True,
    "dtype": "bfloat16",
}

INT_KEYS = {
    "seq_length",
    "batch_size",
    "train_steps",
    "warmup_steps",
    "diffusion_steps",
    "inference_mask_span",
    "top_k",
    "num_workers",
    "seed",
    "log_every",
    "save_every",
    "benchmark_controller_plateau_patience",
    "mask_span_min_tokens",
    "mask_span_max_tokens",
    "max_file_size_bytes",
    "max_line_length",
    "max_code_lines",
    "dataset_preview_examples",
    "prepare_codesearchnet_examples_per_language",
    "prepare_commitpack_examples_per_language",
    "prepare_swe_rebench_examples_per_language",
    "prepare_max_local_files",
    "prepare_context_lines",
    "prepare_max_example_chars",
    "prepare_context_max_lines",
    "synthetic_timeout_seconds",
}

FLOAT_KEYS = {
    "learning_rate",
    "benchmark_controller_improvement_threshold",
    "benchmark_controller_lr_decay_factor",
    "benchmark_controller_min_learning_rate",
    "benchmark_controller_task_mix_strength",
    "benchmark_controller_task_mix_momentum",
    "benchmark_controller_min_task_weight",
    "weight_decay",
    "grad_clip",
    "mask_ratio_min",
    "mask_ratio_max",
    "temperature",
    "top_p",
    "confidence_threshold",
    "remask_fraction",
    "lora_dropout",
    "random_mask_ratio_min",
    "random_mask_ratio_max",
}

BOOL_KEYS = {
    "gradient_checkpointing",
    "qlora_use_double_quant",
    "deduplicate_dataset",
    "validate_python",
    "filter_generated_code",
    "filter_minified_code",
    "filter_boilerplate",
    "prepare_include_local_data",
    "prepare_include_codesearchnet",
    "prepare_include_commitpackft",
    "prepare_include_swe_rebench",
    "prepare_include_commit_context",
    "prepare_include_issue_context",
    "prepare_include_repo_context",
    "synthetic_generation_enabled",
    "trust_remote_code",
    "resume_from_checkpoint",
    "benchmark_controller_enabled",
    "benchmark_controller_reweight_task_mix",
}


def _coerce_path(base_dir: Path, raw_path: str | None) -> str | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _assign_override(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[key] = next_value
        cursor = next_value
    cursor[keys[-1]] = value


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    config = _deep_merge(DEFAULT_CONFIG, loaded)

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected key=value.")
        key, raw_value = override.split("=", 1)
        _assign_override(config, key, yaml.safe_load(raw_value))

    base_dir = config_path.parent
    config["data_dir"] = _coerce_path(base_dir, config.get("data_dir"))
    config["output_dir"] = _coerce_path(base_dir, config.get("output_dir"))
    config["prepare_output_dir"] = _coerce_path(base_dir, config.get("prepare_output_dir"))
    config["synthetic_cache_dir"] = _coerce_path(base_dir, config.get("synthetic_cache_dir"))
    config["config_path"] = str(config_path)

    for key in INT_KEYS:
        if config.get(key) is not None:
            config[key] = int(config[key])
    for key in FLOAT_KEYS:
        if config.get(key) is not None:
            config[key] = float(config[key])
    for key in BOOL_KEYS:
        if config.get(key) is not None:
            config[key] = bool(config[key])

    if config["mask_ratio_min"] > config["mask_ratio_max"]:
        raise ValueError("mask_ratio_min must be <= mask_ratio_max")
    if config["random_mask_ratio_min"] > config["random_mask_ratio_max"]:
        raise ValueError("random_mask_ratio_min must be <= random_mask_ratio_max")
    if config["attention_mode"] not in {"full", "conditioned"}:
        raise ValueError("attention_mode must be 'full' or 'conditioned'")
    if config["finetune_method"] not in {"full", "lora", "qlora"}:
        raise ValueError("finetune_method must be 'full', 'lora', or 'qlora'")

    return config
