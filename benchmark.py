from __future__ import annotations

import argparse
import json
from pathlib import Path

from code_diffusion.config import load_config
from code_diffusion.evaluation import benchmark_loaded_model, load_cases_file, render_benchmark_markdown
from code_diffusion.models import load_diffusion_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one or more checkpoints on fixed diffusion prompts.")
    parser.add_argument("--config", default="config.public-data.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--cases",
        default="benchmarks/default_cases.yaml",
        help="Benchmark cases file. Supports YAML, JSON, or JSONL.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Checkpoint path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Training run directory containing final and/or step-* checkpoints. Can be passed multiple times.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a config value with key=value. Can be passed multiple times.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override diffusion steps.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Default is greedy.")
    parser.add_argument("--top-k", type=int, default=0, help="Optional top-k for non-greedy runs.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Optional top-p for non-greedy runs.")
    parser.add_argument("--show-samples", type=int, default=2, help="How many case outputs to include per checkpoint.")
    parser.add_argument("--output-json", default=None, help="Optional path to write JSON results.")
    parser.add_argument("--output-md", default=None, help="Optional path to write Markdown results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, overrides=args.override)
    cases = load_cases_file(args.cases)
    checkpoints = _discover_checkpoints(config=config, explicit_checkpoints=args.checkpoint, run_dirs=args.run_dir)
    if not checkpoints:
        raise ValueError("No checkpoints found. Pass --checkpoint or --run-dir, or ensure output_dir/final exists.")

    all_results = []
    for checkpoint in checkpoints:
        local_config = dict(config)
        local_config["model_name"] = checkpoint
        model, tokenizer = load_diffusion_model(local_config)
        result = benchmark_loaded_model(
            model=model,
            tokenizer=tokenizer,
            config=local_config,
            cases=cases,
            steps=args.steps or config["diffusion_steps"],
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p if 0 < args.top_p < 1.0 else None,
            show_samples=args.show_samples,
        )
        checkpoint_path = Path(checkpoint)
        result = {
            "checkpoint": checkpoint,
            "checkpoint_label": checkpoint_path.name or str(checkpoint_path),
            **result,
        }
        all_results.append(result)
        print(
            f"{result['checkpoint_label']}: "
            f"exact_match_rate={result['exact_match_rate']:.3f} "
            f"avg_similarity={result['avg_similarity']:.3f}"
        )

    payload = {
        "cases_file": str(Path(args.cases).resolve()),
        "checkpoint_count": len(all_results),
        "results": all_results,
    }
    rendered_json = json.dumps(payload, indent=2)
    print(rendered_json)

    if args.output_json:
        Path(args.output_json).write_text(rendered_json, encoding="utf-8")
    if args.output_md:
        Path(args.output_md).write_text(render_benchmark_markdown(payload), encoding="utf-8")


def _discover_checkpoints(
    *,
    config: dict,
    explicit_checkpoints: list[str],
    run_dirs: list[str],
) -> list[str]:
    discovered: list[str] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        resolved = str(path.resolve())
        if resolved not in seen and _is_checkpoint_dir(path):
            seen.add(resolved)
            discovered.append(resolved)

    for checkpoint in explicit_checkpoints:
        add(Path(checkpoint))

    for run_dir in run_dirs:
        root = Path(run_dir)
        if _is_checkpoint_dir(root):
            add(root)
        final_dir = root / "final"
        if final_dir.exists():
            add(final_dir)
        for step_dir in sorted(root.glob("step-*"), key=_step_sort_key):
            add(step_dir)

    if not discovered:
        final_dir = Path(config["output_dir"]) / "final"
        if final_dir.exists():
            add(final_dir)

    if not discovered:
        model_path = Path(str(config["model_name"]))
        if model_path.exists():
            add(model_path)

    return discovered


def _is_checkpoint_dir(path: Path) -> bool:
    return (path / "adapter_config.json").exists() or (path / "config.json").exists()


def _step_sort_key(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except Exception:
        return 0


if __name__ == "__main__":
    main()
