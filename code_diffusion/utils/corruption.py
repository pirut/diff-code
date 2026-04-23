from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Iterable

import torch


DEFAULT_MODE_WEIGHTS = {
    "random": 0.4,
    "span": 0.4,
    "structure": 0.2,
    "smart": 0.0,
}

DEFAULT_STRUCTURE_FEATURES = {
    "function_bodies": True,
    "full_lines": True,
    "return_statements": True,
    "argument_lists": True,
}

DEFAULT_SMART_WEIGHTS = {
    "function_logic": 1.0,
    "conditionals": 0.9,
    "return_expressions": 1.0,
    "argument_lists": 0.6,
    "full_lines": 0.4,
}


@dataclass(slots=True)
class MaskCandidate:
    kind: str
    span: tuple[int, int]
    weight: float


def corrupt_code(
    tokens: torch.LongTensor,
    *,
    mask_token_id: int,
    mask_ratio: float | None = None,
    ratio_range: tuple[float, float] = (0.15, 0.4),
    special_token_mask: torch.BoolTensor | None = None,
    text: str | None = None,
    offset_mapping: list[tuple[int, int]] | None = None,
    mode: str | None = None,
    strategy_weights: dict[str, float] | None = None,
    span_min_tokens: int = 3,
    span_max_tokens: int = 50,
    structure_features: dict[str, bool] | None = None,
    smart_weights: dict[str, float] | None = None,
    return_metadata: bool = False,
    rng: random.Random | None = None,
    torch_generator: torch.Generator | None = None,
) -> tuple[torch.LongTensor, torch.BoolTensor] | tuple[torch.LongTensor, torch.BoolTensor, dict[str, object]]:
    """
    Returns:
        corrupted_tokens
        mask_positions (bool mask)
        optional metadata
    """

    if tokens.ndim != 1:
        raise ValueError("corrupt_code expects a 1D token tensor.")

    rng = rng or random.Random()
    strategy_weights = {**DEFAULT_MODE_WEIGHTS, **(strategy_weights or {})}
    structure_features = {**DEFAULT_STRUCTURE_FEATURES, **(structure_features or {})}
    smart_weights = {**DEFAULT_SMART_WEIGHTS, **(smart_weights or {})}

    if special_token_mask is None:
        special_token_mask = torch.zeros_like(tokens, dtype=torch.bool)
    else:
        special_token_mask = special_token_mask.to(dtype=torch.bool)

    valid_positions = ~special_token_mask
    valid_count = int(valid_positions.sum().item())
    if valid_count == 0:
        empty_mask = torch.zeros_like(tokens, dtype=torch.bool)
        if return_metadata:
            return tokens.clone(), empty_mask, {
                "mask_strategy": "none",
                "masked_token_count": 0,
                "target_mask_ratio": 0.0,
                "candidate_count": 0,
                "selected_span_count": 0,
                "candidate_kinds": [],
            }
        return tokens.clone(), empty_mask

    if mask_ratio is None:
        mask_ratio = rng.uniform(*ratio_range)
    target_count = max(1, min(valid_count - 1 if valid_count > 1 else 1, math.ceil(valid_count * mask_ratio)))

    selected_mode = mode or _choose_mode(strategy_weights, rng)
    candidates = _find_mask_candidates(
        text=text,
        structure_features=structure_features,
        smart_weights=smart_weights,
    )

    if selected_mode == "random":
        mask_positions = _random_mask(valid_positions, target_count, torch_generator=torch_generator)
        selected_candidates: list[MaskCandidate] = []
    elif selected_mode == "span":
        mask_positions = _span_mask(
            valid_positions,
            target_count,
            span_min_tokens=span_min_tokens,
            span_max_tokens=span_max_tokens,
            rng=rng,
            torch_generator=torch_generator,
        )
        selected_candidates = []
    elif selected_mode == "structure":
        mask_positions, selected_candidates = _candidate_mask(
            valid_positions=valid_positions,
            target_count=target_count,
            offset_mapping=offset_mapping,
            candidates=candidates,
            selection_mode="shuffle",
            rng=rng,
            torch_generator=torch_generator,
        )
    elif selected_mode == "smart":
        mask_positions, selected_candidates = _candidate_mask(
            valid_positions=valid_positions,
            target_count=target_count,
            offset_mapping=offset_mapping,
            candidates=candidates,
            selection_mode="weighted",
            rng=rng,
            torch_generator=torch_generator,
        )
    else:
        raise ValueError(f"Unknown corruption mode: {selected_mode}")

    if not mask_positions.any():
        mask_positions = _random_mask(valid_positions, target_count, torch_generator=torch_generator)
        selected_mode = "random"
        selected_candidates = []

    corrupted_tokens = tokens.clone()
    corrupted_tokens[mask_positions] = mask_token_id

    if not return_metadata:
        return corrupted_tokens, mask_positions

    metadata = {
        "mask_strategy": selected_mode,
        "masked_token_count": int(mask_positions.sum().item()),
        "target_mask_ratio": float(mask_ratio),
        "candidate_count": len(candidates),
        "selected_span_count": len(selected_candidates),
        "candidate_kinds": [candidate.kind for candidate in selected_candidates],
        "char_spans": [list(candidate.span) for candidate in selected_candidates],
    }
    return corrupted_tokens, mask_positions, metadata


def _random_mask(
    valid_positions: torch.BoolTensor,
    target_count: int,
    *,
    torch_generator: torch.Generator | None,
) -> torch.BoolTensor:
    indices = torch.nonzero(valid_positions, as_tuple=False).flatten()
    if indices.numel() == 0:
        return torch.zeros_like(valid_positions, dtype=torch.bool)
    permutation = torch.randperm(indices.numel(), generator=torch_generator)
    chosen = indices[permutation[:target_count]]
    mask = torch.zeros_like(valid_positions, dtype=torch.bool)
    mask[chosen] = True
    return mask


def _span_mask(
    valid_positions: torch.BoolTensor,
    target_count: int,
    *,
    span_min_tokens: int,
    span_max_tokens: int,
    rng: random.Random,
    torch_generator: torch.Generator | None,
) -> torch.BoolTensor:
    indices = torch.nonzero(valid_positions, as_tuple=False).flatten().tolist()
    if not indices:
        return torch.zeros_like(valid_positions, dtype=torch.bool)

    mask = torch.zeros_like(valid_positions, dtype=torch.bool)
    contiguous = _contiguous_ranges(indices)
    attempts = 0

    while int(mask.sum().item()) < target_count and attempts < target_count * 8:
        attempts += 1
        start, end = rng.choice(contiguous)
        max_length = min(span_max_tokens, end - start + 1)
        min_length = min(span_min_tokens, max_length)
        span_length = rng.randint(max(1, min_length), max(1, max_length))
        span_start = rng.randint(start, max(start, end - span_length + 1))
        span_end = span_start + span_length - 1
        mask[span_start : span_end + 1] = valid_positions[span_start : span_end + 1]

    return _finalize_mask(
        mask=mask,
        valid_positions=valid_positions,
        target_count=target_count,
        torch_generator=torch_generator,
    )


def _candidate_mask(
    *,
    valid_positions: torch.BoolTensor,
    target_count: int,
    offset_mapping: list[tuple[int, int]] | None,
    candidates: list[MaskCandidate],
    selection_mode: str,
    rng: random.Random,
    torch_generator: torch.Generator | None,
) -> tuple[torch.BoolTensor, list[MaskCandidate]]:
    if not offset_mapping or not candidates:
        return (
            _span_mask(
                valid_positions,
                target_count,
                span_min_tokens=3,
                span_max_tokens=50,
                rng=rng,
                torch_generator=torch_generator,
            ),
            [],
        )

    selected: list[MaskCandidate] = []
    remaining = list(candidates)
    mask = torch.zeros_like(valid_positions, dtype=torch.bool)

    if selection_mode == "shuffle":
        rng.shuffle(remaining)

    while remaining and int(mask.sum().item()) < target_count:
        if selection_mode == "weighted":
            weights = [max(0.0, candidate.weight) for candidate in remaining]
            if sum(weights) <= 0:
                break
            candidate = rng.choices(remaining, weights=weights, k=1)[0]
            remaining.remove(candidate)
        else:
            candidate = remaining.pop(0)

        span_mask = _char_span_to_token_mask(offset_mapping, candidate.span)
        span_mask &= valid_positions
        if not span_mask.any():
            continue
        mask |= span_mask
        selected.append(candidate)

    return (
        _finalize_mask(
            mask=mask,
            valid_positions=valid_positions,
            target_count=target_count,
            torch_generator=torch_generator,
        ),
        selected,
    )


def _finalize_mask(
    *,
    mask: torch.BoolTensor,
    valid_positions: torch.BoolTensor,
    target_count: int,
    torch_generator: torch.Generator | None,
) -> torch.BoolTensor:
    masked_count = int(mask.sum().item())
    if masked_count > target_count:
        return _trim_mask(mask, target_count, torch_generator=torch_generator)
    if masked_count < target_count:
        remaining = valid_positions & ~mask
        needed = min(target_count - masked_count, int(remaining.sum().item()))
        if needed > 0:
            mask |= _random_mask(remaining, needed, torch_generator=torch_generator)
    return mask


def _find_mask_candidates(
    *,
    text: str | None,
    structure_features: dict[str, bool],
    smart_weights: dict[str, float],
) -> list[MaskCandidate]:
    if not text:
        return []

    candidates: list[MaskCandidate] = []
    if structure_features.get("function_bodies", True):
        candidates.extend(
            MaskCandidate("function_logic", span, smart_weights.get("function_logic", 1.0))
            for span in _find_function_logic_spans(text)
        )
    if structure_features.get("full_lines", True):
        candidates.extend(
            MaskCandidate("full_lines", span, smart_weights.get("full_lines", 0.4))
            for span in _find_full_line_spans(text)
        )
    if structure_features.get("return_statements", True):
        candidates.extend(
            MaskCandidate("return_expressions", span, smart_weights.get("return_expressions", 1.0))
            for span in _find_return_spans(text)
        )
    if structure_features.get("argument_lists", True):
        candidates.extend(
            MaskCandidate("argument_lists", span, smart_weights.get("argument_lists", 0.6))
            for span in _find_argument_list_spans(text)
        )
    candidates.extend(
        MaskCandidate("conditionals", span, smart_weights.get("conditionals", 0.9))
        for span in _find_conditional_spans(text)
    )
    return _dedupe_candidates(candidates)


def _find_function_logic_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    spans.extend(_find_brace_spans(text))
    spans.extend(_find_python_body_spans(text))
    return spans


def _find_brace_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    stack: list[int] = []
    for index, char in enumerate(text):
        if char == "{":
            stack.append(index)
        elif char == "}" and stack:
            start = stack.pop()
            if index - start > 2:
                spans.append((start + 1, index))
    return spans


def _find_python_body_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    lines = text.splitlines(keepends=True)
    starts: list[int] = []
    cursor = 0
    for line in lines:
        starts.append(cursor)
        cursor += len(line)

    block_header = re.compile(
        r"^\s*(async\s+def|def|class|if|elif|else:|for|while|with|try:|except|finally:|match|case)\b"
    )

    for index, line in enumerate(lines[:-1]):
        stripped = line.strip()
        if not stripped or not stripped.endswith(":"):
            continue
        if not block_header.match(stripped) and ":" not in stripped:
            continue

        base_indent = len(line) - len(line.lstrip(" "))
        next_index = index + 1
        while next_index < len(lines) and not lines[next_index].strip():
            next_index += 1
        if next_index >= len(lines):
            continue

        next_line = lines[next_index]
        next_indent = len(next_line) - len(next_line.lstrip(" "))
        if next_indent <= base_indent:
            continue

        end_index = next_index + 1
        while end_index < len(lines):
            candidate = lines[end_index]
            if not candidate.strip():
                end_index += 1
                continue
            candidate_indent = len(candidate) - len(candidate.lstrip(" "))
            if candidate_indent <= base_indent:
                break
            end_index += 1

        spans.append((starts[next_index], starts[end_index] if end_index < len(starts) else len(text)))

    return spans


def _find_full_line_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        line_end = cursor + len(line)
        if line.strip():
            spans.append((cursor, line_end))
        cursor = line_end
    return spans


def _find_return_spans(text: str) -> list[tuple[int, int]]:
    patterns = [
        re.compile(r"(?m)^\s*return\b.*(?:\n|$)"),
        re.compile(r"(?m)^\s*yield\b.*(?:\n|$)"),
    ]
    spans: list[tuple[int, int]] = []
    for pattern in patterns:
        spans.extend((match.start(), match.end()) for match in pattern.finditer(text))
    return spans


def _find_argument_list_spans(text: str) -> list[tuple[int, int]]:
    patterns = [
        re.compile(r"(?m)\b(?:async\s+def|def|class)\s+\w+\((.*?)\)\s*:"),
        re.compile(r"(?m)\bfunction\s+\w+\((.*?)\)"),
        re.compile(r"(?m)\b(?:if|for|while|switch|catch)\s*\((.*?)\)"),
    ]
    spans: list[tuple[int, int]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            start, end = match.span(1)
            if end > start:
                spans.append((start, end))
    return spans


def _find_conditional_spans(text: str) -> list[tuple[int, int]]:
    patterns = [
        re.compile(r"(?m)^\s*(?:if|elif|while)\s+(.+?)\s*:\s*$"),
        re.compile(r"(?m)\b(?:if|while|switch)\s*\((.*?)\)"),
    ]
    spans: list[tuple[int, int]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            start, end = match.span(1)
            if end > start:
                spans.append((start, end))
    return spans


def _char_span_to_token_mask(
    offset_mapping: list[tuple[int, int]],
    span: tuple[int, int],
) -> torch.BoolTensor:
    start_char, end_char = span
    mask = torch.zeros(len(offset_mapping), dtype=torch.bool)
    for index, (token_start, token_end) in enumerate(offset_mapping):
        if token_start == token_end:
            continue
        if token_end <= start_char or token_start >= end_char:
            continue
        mask[index] = True
    return mask


def _trim_mask(
    mask: torch.BoolTensor,
    target_count: int,
    *,
    torch_generator: torch.Generator | None,
) -> torch.BoolTensor:
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() <= target_count:
        return mask
    chosen = indices[torch.randperm(indices.numel(), generator=torch_generator)[:target_count]]
    trimmed = torch.zeros_like(mask, dtype=torch.bool)
    trimmed[chosen] = True
    return trimmed


def _contiguous_ranges(indices: Iterable[int]) -> list[tuple[int, int]]:
    ordered = list(indices)
    if not ordered:
        return []
    ranges: list[tuple[int, int]] = []
    start = ordered[0]
    end = ordered[0]
    for index in ordered[1:]:
        if index == end + 1:
            end = index
            continue
        ranges.append((start, end))
        start = index
        end = index
    ranges.append((start, end))
    return ranges


def _dedupe_candidates(candidates: list[MaskCandidate]) -> list[MaskCandidate]:
    unique: list[MaskCandidate] = []
    seen: set[tuple[str, tuple[int, int]]] = set()
    for candidate in candidates:
        if candidate.span[1] <= candidate.span[0]:
            continue
        key = (candidate.kind, candidate.span)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _choose_mode(strategy_weights: dict[str, float], rng: random.Random) -> str:
    filtered = [(name, weight) for name, weight in strategy_weights.items() if float(weight) > 0]
    if not filtered:
        return "random"
    labels = [name for name, _ in filtered]
    weights = [float(weight) for _, weight in filtered]
    return rng.choices(labels, weights=weights, k=1)[0]
