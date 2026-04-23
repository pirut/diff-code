from __future__ import annotations

import random
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import torch

from code_diffusion.data.synthetic import SyntheticExampleGenerator
from code_diffusion.utils.corruption import corrupt_code


SOURCE_BUCKETS = {"test", "doc"}


@dataclass(slots=True)
class SampleBlueprint:
    task_bucket: str
    task_type: str
    mask_strategy: str


def choose_blueprint(*, config: dict, source_type: str, rng: random.Random) -> SampleBlueprint:
    task_bucket = _weighted_choice(config.get("task_type_weights", {}), rng, default="masked_reconstruction")

    if task_bucket == "fim":
        return SampleBlueprint(task_bucket=task_bucket, task_type="fim", mask_strategy="span")
    if task_bucket == "bug_fix":
        return SampleBlueprint(task_bucket=task_bucket, task_type="bug_fix", mask_strategy="smart")
    if task_bucket == "refinement":
        return SampleBlueprint(task_bucket=task_bucket, task_type="refinement", mask_strategy="structure")
    if task_bucket == "doc_test_based":
        strategy = "structure" if source_type in SOURCE_BUCKETS else "smart"
        return SampleBlueprint(task_bucket=task_bucket, task_type="masked_reconstruction", mask_strategy=strategy)

    strategy = _weighted_choice(config.get("mask_strategy_weights", {}), rng, default="random")
    return SampleBlueprint(task_bucket="masked_reconstruction", task_type="masked_reconstruction", mask_strategy=strategy)


def build_training_example(
    *,
    tokenizer,
    clean_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    clean_text: str | None,
    offset_mapping: list[tuple[int, int]] | None,
    mask_token_id: int,
    current_mask_ratio: float,
    config: dict,
    source_path: str,
    source_type: str,
    quality_score: float,
    blueprint: SampleBlueprint,
    pad_token_id: int,
    synthetic_generator: SyntheticExampleGenerator | None = None,
    rng: random.Random | None = None,
    torch_seed: int | None = None,
    is_valid_target: bool | None = None,
) -> dict[str, object]:
    rng = rng or random.Random()
    torch_generator = torch.Generator()
    if torch_seed is not None:
        torch_generator.manual_seed(torch_seed)

    valid_length = int(attention_mask.sum().item())
    clean_valid_ids = clean_ids[:valid_length].clone()
    valid_offsets = offset_mapping[:valid_length] if offset_mapping else None

    metadata = {
        "mask_strategy": blueprint.mask_strategy,
        "source_type": source_type,
        "quality_score": float(quality_score),
        "source_path": source_path,
        "task_bucket": blueprint.task_bucket,
        "target_is_valid": is_valid_target,
        "synthetic": False,
    }

    if blueprint.task_type == "fim":
        example = _build_fim_example(
            clean_ids=clean_valid_ids,
            clean_text=clean_text,
            offset_mapping=valid_offsets,
            attention_mask=attention_mask,
            mask_token_id=mask_token_id,
            current_mask_ratio=current_mask_ratio,
            config=config,
            rng=rng,
            torch_generator=torch_generator,
        )
    elif blueprint.task_type == "bug_fix":
        example = _build_text_corruption_example(
            task_type="bug_fix",
            tokenizer=tokenizer,
            clean_ids=clean_valid_ids,
            clean_text=clean_text,
            attention_mask=attention_mask,
            mask_token_id=mask_token_id,
            config=config,
            source_path=source_path,
            source_type=source_type,
            synthetic_generator=synthetic_generator,
            rng=rng,
        )
    elif blueprint.task_type == "refinement":
        example = _build_text_corruption_example(
            task_type="refinement",
            tokenizer=tokenizer,
            clean_ids=clean_valid_ids,
            clean_text=clean_text,
            attention_mask=attention_mask,
            mask_token_id=mask_token_id,
            config=config,
            source_path=source_path,
            source_type=source_type,
            synthetic_generator=synthetic_generator,
            rng=rng,
        )
    else:
        example = _build_masked_reconstruction_example(
            clean_ids=clean_valid_ids,
            clean_text=clean_text,
            offset_mapping=valid_offsets,
            attention_mask=attention_mask,
            mask_token_id=mask_token_id,
            current_mask_ratio=current_mask_ratio,
            config=config,
            mode=blueprint.mask_strategy,
            rng=rng,
            torch_generator=torch_generator,
        )

    metadata.update(example["metadata"])
    return {
        "input_ids": _pad_tensor(example["input_ids"], attention_mask.shape[0], pad_token_id),
        "labels": clean_ids.clone(),
        "mask": _pad_bool_tensor(example["mask"], attention_mask.shape[0]),
        "attention_mask": attention_mask.clone(),
        "task_type": blueprint.task_type,
        "corrupted_code": example["corrupted_code"],
        "target_code": clean_text or tokenizer.decode(clean_valid_ids.tolist(), skip_special_tokens=True),
        "mask_metadata": example["mask_metadata"],
        "metadata": metadata,
    }


def build_prepared_training_example(
    *,
    tokenizer,
    task_type: str,
    target_code: str,
    corrupted_code: str,
    metadata: dict[str, object],
    mask_metadata: dict[str, object],
    seq_length: int,
    pad_token_id: int,
    mask_token_id: int,
) -> dict[str, object]:
    target_ids = tokenizer(target_code, add_special_tokens=False, truncation=False)["input_ids"]
    corrupted_ids = tokenizer(corrupted_code, add_special_tokens=False, truncation=False)["input_ids"]
    if not target_ids:
        raise ValueError("Prepared example target_code tokenized to zero tokens.")

    clean_tensor = torch.tensor(target_ids, dtype=torch.long)
    aligned_ids, mask_positions, alignment_metadata = _align_corrupted_tokens(
        clean_ids=clean_tensor,
        corrupted_ids=corrupted_ids,
        mask_token_id=mask_token_id,
    )
    if not mask_positions.any():
        corrupted_ids_tensor, mask_positions = corrupt_code(
            clean_tensor,
            mask_token_id=mask_token_id,
            mask_ratio=0.15,
            mode="span",
        )
        aligned_ids = corrupted_ids_tensor
        alignment_metadata = {"masked_token_count": int(mask_positions.sum().item()), "dropped_insertions": 0}

    clean_tensor, aligned_ids, mask_positions = _crop_aligned_tensors(
        clean_ids=clean_tensor,
        aligned_ids=aligned_ids,
        mask_positions=mask_positions,
        seq_length=seq_length,
    )
    attention_mask = torch.ones(clean_tensor.shape[0], dtype=torch.long)

    return {
        "input_ids": _pad_tensor(aligned_ids, seq_length, pad_token_id),
        "labels": _pad_tensor(clean_tensor, seq_length, pad_token_id),
        "mask": _pad_bool_tensor(mask_positions, seq_length),
        "attention_mask": _pad_tensor(attention_mask, seq_length, 0),
        "task_type": task_type,
        "corrupted_code": corrupted_code,
        "target_code": target_code,
        "mask_metadata": {**mask_metadata, **alignment_metadata},
        "metadata": metadata,
    }


def _build_masked_reconstruction_example(
    *,
    clean_ids: torch.LongTensor,
    clean_text: str | None,
    offset_mapping: list[tuple[int, int]] | None,
    attention_mask: torch.LongTensor,
    mask_token_id: int,
    current_mask_ratio: float,
    config: dict,
    mode: str,
    rng: random.Random,
    torch_generator: torch.Generator,
) -> dict[str, object]:
    special_token_mask = ~attention_mask[: clean_ids.shape[0]].bool()
    corrupted_ids, mask_positions, corruption_metadata = corrupt_code(
        clean_ids,
        mask_token_id=mask_token_id,
        mask_ratio=current_mask_ratio,
        ratio_range=(config["random_mask_ratio_min"], config["random_mask_ratio_max"]),
        special_token_mask=special_token_mask,
        text=clean_text,
        offset_mapping=offset_mapping,
        mode=mode,
        strategy_weights=config.get("mask_strategy_weights"),
        span_min_tokens=config.get("mask_span_min_tokens", 3),
        span_max_tokens=config.get("mask_span_max_tokens", 50),
        structure_features=config.get("structure_mask_features"),
        smart_weights=config.get("smart_mask_weights"),
        return_metadata=True,
        rng=rng,
        torch_generator=torch_generator,
    )
    return {
        "input_ids": corrupted_ids,
        "mask": mask_positions,
        "corrupted_code": _render_masked_text(clean_text, offset_mapping, mask_positions) if clean_text else None,
        "mask_metadata": corruption_metadata,
        "metadata": {
            "mask_strategy": corruption_metadata["mask_strategy"],
            "input_is_valid": True,
        },
    }


def _build_fim_example(
    *,
    clean_ids: torch.LongTensor,
    clean_text: str | None,
    offset_mapping: list[tuple[int, int]] | None,
    attention_mask: torch.LongTensor,
    mask_token_id: int,
    current_mask_ratio: float,
    config: dict,
    rng: random.Random,
    torch_generator: torch.Generator,
) -> dict[str, object]:
    special_token_mask = ~attention_mask[: clean_ids.shape[0]].bool()
    corrupted_ids, mask_positions, corruption_metadata = corrupt_code(
        clean_ids,
        mask_token_id=mask_token_id,
        mask_ratio=max(current_mask_ratio, config.get("random_mask_ratio_min", 0.10)),
        ratio_range=(config["random_mask_ratio_min"], config["random_mask_ratio_max"]),
        special_token_mask=special_token_mask,
        text=clean_text,
        offset_mapping=offset_mapping,
        mode="span",
        strategy_weights=config.get("mask_strategy_weights"),
        span_min_tokens=config.get("mask_span_min_tokens", 3),
        span_max_tokens=config.get("mask_span_max_tokens", 50),
        structure_features=config.get("structure_mask_features"),
        smart_weights=config.get("smart_mask_weights"),
        return_metadata=True,
        rng=rng,
        torch_generator=torch_generator,
    )
    prefix_chars, middle_chars, suffix_chars = _split_text_for_mask(clean_text, offset_mapping, mask_positions)
    corruption_metadata.update(
        {
            "task_type": "fim",
            "prefix_chars": prefix_chars,
            "middle_chars": middle_chars,
            "suffix_chars": suffix_chars,
        }
    )
    return {
        "input_ids": corrupted_ids,
        "mask": mask_positions,
        "corrupted_code": _render_masked_text(clean_text, offset_mapping, mask_positions) if clean_text else None,
        "mask_metadata": corruption_metadata,
        "metadata": {
            "mask_strategy": "fim",
            "input_is_valid": True,
        },
    }


def _build_text_corruption_example(
    *,
    task_type: str,
    tokenizer,
    clean_ids: torch.LongTensor,
    clean_text: str | None,
    attention_mask: torch.LongTensor,
    mask_token_id: int,
    config: dict,
    source_path: str,
    source_type: str,
    synthetic_generator: SyntheticExampleGenerator | None,
    rng: random.Random,
) -> dict[str, object]:
    if clean_text:
        synthetic = synthetic_generator.maybe_generate(
            task_type=task_type,
            clean_code=clean_text,
            source_path=source_path,
            source_type=source_type,
        ) if synthetic_generator else None
    else:
        synthetic = None

    if synthetic:
        corrupted_text = synthetic.corrupted_code
        target_text = synthetic.target_code
        metadata = {
            "mask_strategy": "synthetic",
            "input_is_valid": task_type != "bug_fix",
            "synthetic": True,
            "synthetic_prompt_path": synthetic.prompt_path,
            "synthetic_response_path": synthetic.response_path,
            "synthetic_provider": synthetic.provider,
            "synthetic_model": synthetic.model,
        }
    else:
        target_text = clean_text or tokenizer.decode(clean_ids.tolist(), skip_special_tokens=True)
        corrupted_text, local_metadata = _build_local_text_corruption(
            task_type=task_type,
            clean_text=target_text,
            source_path=source_path,
            rng=rng,
        )
        metadata = {
            "mask_strategy": local_metadata["mask_strategy"],
            "input_is_valid": local_metadata["input_is_valid"],
            "synthetic": False,
            "corruption_kind": local_metadata["corruption_kind"],
        }

    corrupted_ids = tokenizer(corrupted_text, add_special_tokens=False, truncation=False)["input_ids"]
    aligned_ids, mask_positions, alignment_metadata = _align_corrupted_tokens(
        clean_ids=clean_ids,
        corrupted_ids=corrupted_ids,
        mask_token_id=mask_token_id,
    )
    if not mask_positions.any():
        fallback = _build_masked_reconstruction_example(
            clean_ids=clean_ids,
            clean_text=clean_text,
            offset_mapping=None,
            attention_mask=attention_mask,
            mask_token_id=mask_token_id,
            current_mask_ratio=config["mask_ratio_min"],
            config=config,
            mode="smart" if task_type == "bug_fix" else "structure",
            rng=rng,
            torch_generator=torch.Generator().manual_seed(rng.randint(0, 2**31 - 1)),
        )
        fallback["metadata"].update(metadata)
        return fallback

    return {
        "input_ids": aligned_ids,
        "mask": mask_positions,
        "corrupted_code": corrupted_text,
        "mask_metadata": {
            "task_type": task_type,
            **alignment_metadata,
        },
        "metadata": metadata,
    }


def _build_local_text_corruption(
    *,
    task_type: str,
    clean_text: str,
    source_path: str,
    rng: random.Random,
) -> tuple[str, dict[str, object]]:
    if task_type == "bug_fix":
        candidates = [
            _operator_bug,
            _condition_bug,
            _remove_return_bug,
            _remove_import_bug,
            _variable_swap_bug,
        ]
        input_is_valid = False
        mask_strategy = "synthetic_bug"
    else:
        candidates = [
            _remove_logic_lines,
            _simplify_return_expression,
            _drop_branch_body,
        ]
        input_is_valid = False
        mask_strategy = "draft_refinement"

    for transform in rng.sample(candidates, k=len(candidates)):
        corrupted = transform(clean_text, source_path, rng)
        if corrupted and corrupted != clean_text:
            return corrupted, {
                "corruption_kind": transform.__name__,
                "mask_strategy": mask_strategy,
                "input_is_valid": input_is_valid,
            }

    return clean_text, {
        "corruption_kind": "no_op",
        "mask_strategy": mask_strategy,
        "input_is_valid": True,
    }


def _align_corrupted_tokens(
    *,
    clean_ids: torch.LongTensor,
    corrupted_ids: list[int],
    mask_token_id: int,
) -> tuple[torch.LongTensor, torch.BoolTensor, dict[str, int]]:
    aligned = clean_ids.clone()
    mask = torch.zeros_like(clean_ids, dtype=torch.bool)
    dropped_insertions = 0

    matcher = SequenceMatcher(a=clean_ids.tolist(), b=corrupted_ids, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            if i2 > i1:
                aligned[i1:i2] = torch.tensor(corrupted_ids[j1:j2], dtype=torch.long)
            continue

        if tag == "insert":
            dropped_insertions += j2 - j1
            continue

        span_len = i2 - i1
        if span_len <= 0:
            continue

        replacement = corrupted_ids[j1:j2] if tag == "replace" else []
        replacement = replacement[:span_len]
        if replacement:
            aligned[i1 : i1 + len(replacement)] = torch.tensor(replacement, dtype=torch.long)
        if len(replacement) < span_len:
            aligned[i1 + len(replacement) : i2] = mask_token_id
        mask[i1:i2] = True

    return aligned, mask, {
        "masked_token_count": int(mask.sum().item()),
        "dropped_insertions": dropped_insertions,
    }


def _render_masked_text(
    text: str | None,
    offset_mapping: list[tuple[int, int]] | None,
    mask_positions: torch.BoolTensor,
) -> str | None:
    if not text or not offset_mapping:
        return None
    spans = _mask_positions_to_char_spans(offset_mapping, mask_positions)
    if not spans:
        return text

    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            parts.append(text[cursor:start])
        parts.append("[MASK]")
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _split_text_for_mask(
    text: str | None,
    offset_mapping: list[tuple[int, int]] | None,
    mask_positions: torch.BoolTensor,
) -> tuple[int | None, int | None, int | None]:
    if not text or not offset_mapping:
        return None, None, None
    spans = _mask_positions_to_char_spans(offset_mapping, mask_positions)
    if not spans:
        return len(text), 0, 0
    start, end = spans[0]
    return start, end - start, max(0, len(text) - end)


def _mask_positions_to_char_spans(
    offset_mapping: list[tuple[int, int]],
    mask_positions: torch.BoolTensor,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    active_start: int | None = None
    active_end: int | None = None

    for is_masked, (start, end) in zip(mask_positions.tolist(), offset_mapping, strict=False):
        if not is_masked or end <= start:
            if active_start is not None and active_end is not None:
                spans.append((active_start, active_end))
                active_start = None
                active_end = None
            continue
        if active_start is None:
            active_start = start
            active_end = end
            continue
        if start <= active_end:
            active_end = max(active_end, end)
        else:
            spans.append((active_start, active_end))
            active_start = start
            active_end = end

    if active_start is not None and active_end is not None:
        spans.append((active_start, active_end))
    return spans


def _pad_tensor(values: torch.LongTensor, target_length: int, pad_token_id: int) -> torch.LongTensor:
    if values.shape[0] >= target_length:
        return values[:target_length].clone()
    padded = torch.full((target_length,), pad_token_id, dtype=torch.long)
    padded[: values.shape[0]] = values
    return padded


def _pad_bool_tensor(values: torch.BoolTensor, target_length: int) -> torch.BoolTensor:
    if values.shape[0] >= target_length:
        return values[:target_length].clone()
    padded = torch.zeros(target_length, dtype=torch.bool)
    padded[: values.shape[0]] = values
    return padded


def _crop_aligned_tensors(
    *,
    clean_ids: torch.LongTensor,
    aligned_ids: torch.LongTensor,
    mask_positions: torch.BoolTensor,
    seq_length: int,
) -> tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]:
    if clean_ids.shape[0] <= seq_length:
        return clean_ids, aligned_ids, mask_positions

    mask_indices = torch.nonzero(mask_positions, as_tuple=False).flatten()
    if mask_indices.numel() == 0:
        return (
            clean_ids[:seq_length].clone(),
            aligned_ids[:seq_length].clone(),
            mask_positions[:seq_length].clone(),
        )

    first = int(mask_indices[0].item())
    last = int(mask_indices[-1].item())
    masked_span = last - first + 1
    available_context = max(0, seq_length - masked_span)
    left_context = available_context // 2
    start = max(0, first - left_context)
    end = min(clean_ids.shape[0], start + seq_length)
    start = max(0, end - seq_length)
    return (
        clean_ids[start:end].clone(),
        aligned_ids[start:end].clone(),
        mask_positions[start:end].clone(),
    )


def _weighted_choice(weights: dict[str, float], rng: random.Random, default: str) -> str:
    filtered = [(key, float(value)) for key, value in weights.items() if float(value) > 0]
    if not filtered:
        return default
    labels = [label for label, _ in filtered]
    probs = [weight for _, weight in filtered]
    return rng.choices(labels, weights=probs, k=1)[0]


def _replace_match(text: str, match: re.Match[str], replacement: str) -> str:
    return text[: match.start()] + replacement + text[match.end() :]


def _operator_bug(text: str, _source_path: str, rng: random.Random) -> str | None:
    replacements = {
        "==": "!=",
        "!=": "==",
        "<=": ">=",
        ">=": "<=",
        "<": ">",
        ">": "<",
        "+": "-",
        "-": "+",
        "*": "+",
        "/": "*",
    }
    matches = [match for match in re.finditer(r"==|!=|<=|>=|<|>|\+|-|\*|/", text) if match.group(0) in replacements]
    if not matches:
        return None
    match = rng.choice(matches)
    return _replace_match(text, match, replacements[match.group(0)])


def _condition_bug(text: str, source_path: str, rng: random.Random) -> str | None:
    if Path(source_path).suffix.lower() == ".py":
        patterns = [
            (re.compile(r"(?m)^(\s*if\s+)(.+?)(\s*:)\s*$"), lambda m: f"{m.group(1)}not ({m.group(2)}){m.group(3)}"),
            (re.compile(r"(?m)^(\s*while\s+)(.+?)(\s*:)\s*$"), lambda m: f"{m.group(1)}not ({m.group(2)}){m.group(3)}"),
        ]
    else:
        patterns = [
            (re.compile(r"(?m)\bif\s*\((.+?)\)"), lambda m: f"if (!({m.group(1)}))"),
            (re.compile(r"(?m)\bwhile\s*\((.+?)\)"), lambda m: f"while (!({m.group(1)}))"),
        ]
    options: list[tuple[re.Match[str], object]] = []
    for pattern, transform in patterns:
        options.extend((match, transform) for match in pattern.finditer(text))
    if not options:
        return None
    match, transform = rng.choice(options)
    return _replace_match(text, match, transform(match))


def _remove_return_bug(text: str, source_path: str, rng: random.Random) -> str | None:
    if Path(source_path).suffix.lower() == ".py":
        pattern = re.compile(r"(?m)^(\s*)return\b.*(?:\n|$)")
        matches = list(pattern.finditer(text))
        if not matches:
            return None
        match = rng.choice(matches)
        replacement = f"{match.group(1)}pass\n"
        return _replace_match(text, match, replacement)

    pattern = re.compile(r"(?m)^(\s*)return\b.*(?:\n|$)")
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    match = rng.choice(matches)
    replacement = f"{match.group(1)};\n"
    return _replace_match(text, match, replacement)


def _remove_import_bug(text: str, source_path: str, rng: random.Random) -> str | None:
    if Path(source_path).suffix.lower() == ".py":
        pattern = re.compile(r"(?m)^(\s*(?:from\s+\S+\s+import\s+.+|import\s+.+))(?:\n|$)")
    else:
        pattern = re.compile(r"(?m)^(\s*(?:import\s+.+|export\s+.+from\s+.+))(?:\n|$)")
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    match = rng.choice(matches)
    return _replace_match(text, match, "")


def _variable_swap_bug(text: str, _source_path: str, rng: random.Random) -> str | None:
    identifiers = [
        match
        for match in re.finditer(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", text)
        if match.group(0) not in {"return", "import", "from", "class", "function", "export", "const", "let", "var"}
    ]
    if len(identifiers) < 2:
        return None
    first, second = rng.sample(identifiers, k=2)
    if first.group(0) == second.group(0):
        return None
    a_start, a_end = first.span()
    b_start, b_end = second.span()
    if a_start > b_start:
        (a_start, a_end, first), (b_start, b_end, second) = (b_start, b_end, second), (a_start, a_end, first)
    return (
        text[:a_start]
        + second.group(0)
        + text[a_end:b_start]
        + first.group(0)
        + text[b_end:]
    )


def _remove_logic_lines(text: str, source_path: str, rng: random.Random) -> str | None:
    lines = text.splitlines(keepends=True)
    candidates = [index for index, line in enumerate(lines) if _is_logic_line(line, source_path)]
    if not candidates:
        return None
    remove_count = max(1, min(2, len(candidates)))
    chosen = set(rng.sample(candidates, k=remove_count))
    return "".join(line for index, line in enumerate(lines) if index not in chosen)


def _simplify_return_expression(text: str, source_path: str, rng: random.Random) -> str | None:
    suffix = Path(source_path).suffix.lower()
    if suffix == ".py":
        pattern = re.compile(r"(?m)^(\s*return\s+)(.+?)(\s*)$")
        replacements = ["None", "0", "result"]
    else:
        pattern = re.compile(r"(?m)^(\s*return\s+)(.+?)(;?\s*)$")
        replacements = ["undefined", "0", "result"]
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    match = rng.choice(matches)
    replacement = f"{match.group(1)}{rng.choice(replacements)}{match.group(3)}"
    return _replace_match(text, match, replacement)


def _drop_branch_body(text: str, source_path: str, rng: random.Random) -> str | None:
    lines = text.splitlines(keepends=True)
    if Path(source_path).suffix.lower() == ".py":
        branch_lines = [index for index, line in enumerate(lines) if re.match(r"^\s*(if|elif|else|for|while|try|except)\b", line)]
        if not branch_lines:
            return None
        start = rng.choice(branch_lines) + 1
        end = min(len(lines), start + rng.randint(1, 2))
        kept: list[str] = []
        for index, line in enumerate(lines):
            if start <= index < end and line.startswith((" ", "\t")):
                continue
            kept.append(line)
        return "".join(kept)

    branch_lines = [index for index, line in enumerate(lines) if re.search(r"\b(if|for|while|switch)\b", line)]
    if not branch_lines:
        return None
    start = rng.choice(branch_lines) + 1
    end = min(len(lines), start + rng.randint(1, 2))
    return "".join(line for index, line in enumerate(lines) if not (start <= index < end))


def _is_logic_line(line: str, source_path: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith(("#", "//", "/*", "*")):
        return False
    if Path(source_path).suffix.lower() == ".py":
        return not stripped.startswith(("def ", "class ", "import ", "from "))
    return not stripped.startswith(("import ", "export ", "class ", "function "))
