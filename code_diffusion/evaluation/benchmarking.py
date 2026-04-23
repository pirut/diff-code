from __future__ import annotations

import json
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import torch
import yaml

from code_diffusion.inference import generate
from code_diffusion.utils.tokenization import (
    decode_tokens,
    encode_prompt_with_masks,
    resolve_mask_token_id,
)


def load_cases_file(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    raw = source.read_text(encoding="utf-8")
    if source.suffix in {".yaml", ".yml"}:
        cases = yaml.safe_load(raw)
    elif source.suffix == ".json":
        cases = json.loads(raw)
    else:
        cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not isinstance(cases, list):
        raise ValueError("Benchmark cases file must contain a list of cases.")
    return [dict(case) for case in cases]


def benchmark_loaded_model(
    *,
    model,
    tokenizer,
    config: dict,
    cases: list[dict[str, object]],
    steps: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    show_samples: int = 2,
) -> dict[str, object]:
    device = next(model.parameters()).device
    mask_token_id = resolve_mask_token_id(tokenizer)

    totals = defaultdict(float)
    by_task_type: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    samples: list[dict[str, object]] = []

    for case in cases:
        prompt = str(case["prompt"])
        target = str(case["target"])
        task_type = str(case.get("task_type", "unknown"))
        prompt_ids, prompt_mask = encode_prompt_with_masks(
            prompt,
            tokenizer=tokenizer,
            mask_token_id=mask_token_id,
            default_mask_span=int(case.get("mask_span_tokens") or config["inference_mask_span"]),
        )
        prompt_ids = prompt_ids.unsqueeze(0).to(device)
        prompt_mask = prompt_mask.unsqueeze(0).to(device)
        attention_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=device)

        with torch.no_grad():
            generated = generate(
                model=model,
                initial_tokens=prompt_ids,
                attention_mask=attention_mask,
                initial_mask=prompt_mask,
                steps=int(case.get("steps") or steps),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                confidence_threshold=config.get("confidence_threshold"),
                remask_fraction=config.get("remask_fraction", 0.5),
                mask_token_id=mask_token_id,
            )

        generated_text = decode_tokens(tokenizer, generated[0].cpu())
        similarity = SequenceMatcher(None, _normalize_text(generated_text), _normalize_text(target)).ratio()
        exact_match = _normalize_text(generated_text) == _normalize_text(target)

        totals["count"] += 1
        totals["similarity_sum"] += similarity
        totals["exact_match_sum"] += 1.0 if exact_match else 0.0

        task_bucket = by_task_type[task_type]
        task_bucket["count"] += 1
        task_bucket["similarity_sum"] += similarity
        task_bucket["exact_match_sum"] += 1.0 if exact_match else 0.0

        if len(samples) < show_samples:
            samples.append(
                {
                    "id": case["id"],
                    "task_type": task_type,
                    "similarity": similarity,
                    "exact_match": exact_match,
                    "prompt": prompt,
                    "generated": generated_text,
                    "target": target,
                }
            )

    return {
        "case_count": int(totals["count"]),
        "exact_match_rate": totals["exact_match_sum"] / max(totals["count"], 1.0),
        "avg_similarity": totals["similarity_sum"] / max(totals["count"], 1.0),
        "by_task_type": {
            task_type: {
                "count": int(metrics["count"]),
                "exact_match_rate": metrics["exact_match_sum"] / max(metrics["count"], 1.0),
                "avg_similarity": metrics["similarity_sum"] / max(metrics["count"], 1.0),
            }
            for task_type, metrics in by_task_type.items()
        },
        "samples": samples,
    }


def render_benchmark_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Benchmark Results",
        "",
        f"- Cases file: {payload.get('cases_file', 'inline')}",
        f"- Checkpoints: {payload.get('checkpoint_count', 1)}",
        "",
    ]
    for result in payload.get("results", []):
        lines.append(f"## {result['checkpoint_label']}")
        lines.append(f"- Exact match rate: {result['exact_match_rate']:.3f}")
        lines.append(f"- Average similarity: {result['avg_similarity']:.3f}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())
