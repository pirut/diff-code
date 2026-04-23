from __future__ import annotations

from pathlib import Path
import re

import torch


def ensure_padding_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token


def resolve_mask_token_id(tokenizer) -> int:
    candidates = [
        getattr(tokenizer, "mask_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "unk_token_id", None),
    ]
    for token_id in candidates:
        if token_id is not None:
            return int(token_id)
    raise ValueError("Tokenizer has no usable special token for masking.")


def encode_prompt_with_masks(
    prompt: str,
    tokenizer,
    mask_token_id: int,
    prepend_bos: bool = True,
    default_mask_span: int = 16,
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    token_ids: list[int] = []
    mask_flags: list[bool] = []
    pattern = re.compile(r"\[MASK(?::(\d+))?\]")

    if prepend_bos and tokenizer.bos_token_id is not None:
        token_ids.append(int(tokenizer.bos_token_id))
        mask_flags.append(False)

    cursor = 0
    for match in pattern.finditer(prompt):
        segment = prompt[cursor : match.start()]
        if segment:
            encoded = tokenizer(segment, add_special_tokens=False)["input_ids"]
            token_ids.extend(encoded)
            mask_flags.extend([False] * len(encoded))
        span_length = int(match.group(1) or default_mask_span)
        span_length = max(1, span_length)
        token_ids.extend([mask_token_id] * span_length)
        mask_flags.extend([True] * span_length)
        cursor = match.end()

    trailing = prompt[cursor:]
    if trailing:
        encoded = tokenizer(trailing, add_special_tokens=False)["input_ids"]
        token_ids.extend(encoded)
        mask_flags.extend([False] * len(encoded))

    return torch.tensor(token_ids, dtype=torch.long), torch.tensor(mask_flags, dtype=torch.bool)


def decode_tokens(tokenizer, token_ids: torch.Tensor) -> str:
    if token_ids.ndim != 1:
        raise ValueError("decode_tokens expects a 1D tensor.")
    return tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)


def list_code_files(data_dir: str | Path, extensions: list[str]) -> list[Path]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    normalized = {ext.lower() for ext in extensions}
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized
    ]
    if not files:
        raise FileNotFoundError(
            f"No code files found in {root} for extensions: {sorted(normalized)}"
        )
    return sorted(files)
