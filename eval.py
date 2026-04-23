from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from code_diffusion.config import load_config
from code_diffusion.data import CodeDiffusionDataset
from code_diffusion.inference import generate
from code_diffusion.models import load_diffusion_model
from code_diffusion.utils.tokenization import decode_tokens, resolve_mask_token_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a diffusion checkpoint on reconstruction and repair tasks.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a config value with key=value. Can be passed multiple times.",
    )
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path. Defaults to output_dir/final.")
    parser.add_argument("--max-samples", type=int, default=32, help="Maximum number of examples to score.")
    parser.add_argument(
        "--task-type",
        action="append",
        default=[],
        help="Optional task type filter. Can be passed multiple times.",
    )
    parser.add_argument("--show-samples", type=int, default=3, help="How many decoded examples to print.")
    parser.add_argument("--steps", type=int, default=None, help="Override diffusion steps.")
    parser.add_argument("--output-json", default=None, help="Optional path to write metrics as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, overrides=args.override)

    checkpoint = args.checkpoint
    if checkpoint is None:
        final_dir = Path(config["output_dir"]) / "final"
        if final_dir.exists():
            checkpoint = str(final_dir)
    if checkpoint is not None:
        config["model_name"] = checkpoint

    model, tokenizer = load_diffusion_model(config)
    dataset = CodeDiffusionDataset(
        tokenizer=tokenizer,
        data_dir=config["data_dir"],
        seq_length=config["seq_length"],
        extensions=config["extensions"],
        mask_ratio_min=config["mask_ratio_min"],
        mask_ratio_max=config["mask_ratio_max"],
        config=config,
    )
    dataset.set_mask_ratio(config["mask_ratio_min"])

    device = next(model.parameters()).device
    mask_token_id = resolve_mask_token_id(tokenizer)
    allowed_task_types = set(args.task_type)

    aggregate: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    decoded_examples: list[dict[str, object]] = []
    total_scored = 0

    for index in range(len(dataset)):
        example = dataset.get_example(index, deterministic=True)
        task_type = str(example["task_type"])
        if allowed_task_types and task_type not in allowed_task_types:
            continue

        batch = {
            "input_ids": example["input_ids"].unsqueeze(0).to(device),
            "labels": example["labels"].unsqueeze(0).to(device),
            "mask": example["mask"].unsqueeze(0).to(device),
            "attention_mask": example["attention_mask"].unsqueeze(0).to(device),
        }
        if int(batch["mask"].sum().item()) == 0:
            continue

        with torch.no_grad():
            outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                mask_positions=batch["mask"],
                use_cache=False,
            )
            masked_logits = outputs.logits[batch["mask"]]
            masked_labels = batch["labels"][batch["mask"]]
            loss = F.cross_entropy(masked_logits, masked_labels)
            predictions = masked_logits.argmax(dim=-1)
            masked_accuracy = (predictions == masked_labels).float().mean().item()

            generated = generate(
                model=model,
                initial_tokens=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                initial_mask=batch["mask"],
                steps=args.steps or config["diffusion_steps"],
                temperature=config["temperature"],
                top_k=config["top_k"],
                top_p=config["top_p"],
                confidence_threshold=config.get("confidence_threshold"),
                remask_fraction=config.get("remask_fraction", 0.5),
                mask_token_id=mask_token_id,
            )

        generated_text = decode_tokens(tokenizer, generated[0].detach().cpu())
        target_text = str(example["target_code"])
        normalized_match = _normalize_text(generated_text) == _normalize_text(target_text)

        bucket = aggregate[task_type]
        bucket["count"] += 1
        bucket["loss_sum"] += float(loss.item())
        bucket["masked_accuracy_sum"] += float(masked_accuracy)
        bucket["exact_match_sum"] += 1.0 if normalized_match else 0.0

        if len(decoded_examples) < args.show_samples:
            decoded_examples.append(
                {
                    "index": index,
                    "task_type": task_type,
                    "metadata": example["metadata"],
                    "corrupted_code": example["corrupted_code"],
                    "generated_code": generated_text,
                    "target_code": target_text,
                    "masked_accuracy": masked_accuracy,
                    "loss": float(loss.item()),
                    "exact_match": normalized_match,
                }
            )

        total_scored += 1
        if total_scored >= args.max_samples:
            break

    results = {
        "checkpoint": config["model_name"],
        "data_dir": config["data_dir"],
        "total_scored": total_scored,
        "by_task_type": {
            task_type: {
                "count": int(metrics["count"]),
                "avg_loss": metrics["loss_sum"] / max(metrics["count"], 1.0),
                "avg_masked_accuracy": metrics["masked_accuracy_sum"] / max(metrics["count"], 1.0),
                "exact_match_rate": metrics["exact_match_sum"] / max(metrics["count"], 1.0),
            }
            for task_type, metrics in aggregate.items()
        },
        "samples": decoded_examples,
    }

    print(json.dumps(results, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")


def _normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


if __name__ == "__main__":
    main()
