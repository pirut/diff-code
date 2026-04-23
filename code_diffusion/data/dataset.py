from __future__ import annotations

import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import Dataset

from code_diffusion.config import DEFAULT_CONFIG
from code_diffusion.data.example_builder import SampleBlueprint, build_training_example, choose_blueprint
from code_diffusion.data.quality import (
    FileQualityReport,
    assess_file_quality,
    build_chunk_quality_metadata,
    export_quality_reports,
)
from code_diffusion.data.synthetic import SyntheticExampleGenerator
from code_diffusion.utils.tokenization import list_code_files, resolve_mask_token_id


class CodeDiffusionDataset(Dataset):
    def __init__(
        self,
        *,
        tokenizer,
        data_dir: str,
        seq_length: int,
        extensions: list[str],
        mask_ratio_min: float,
        mask_ratio_max: float,
        max_files: int | None = None,
        max_samples: int | None = None,
        config: dict | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length
        self.extensions = extensions
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.max_files = max_files
        self.max_samples = max_samples
        self.current_mask_ratio = mask_ratio_min
        self.mask_token_id = resolve_mask_token_id(tokenizer)
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.mask_token_id

        self.config = _build_dataset_config(
            config=config,
            data_dir=data_dir,
            seq_length=seq_length,
            extensions=extensions,
            mask_ratio_min=mask_ratio_min,
            mask_ratio_max=mask_ratio_max,
        )
        self.seed = int(self.config.get("seed", 7))
        self.synthetic_generator = (
            SyntheticExampleGenerator(self.config)
            if self.config.get("synthetic_generation_enabled")
            else None
        )

        self.num_source_files = 0
        self.num_accepted_source_files = 0
        self.file_quality_reports: list[FileQualityReport] = []
        self.rejection_stats: Counter[str] = Counter()
        self.task_type_counts: Counter[str] = Counter()
        self.task_bucket_counts: Counter[str] = Counter()
        self.mask_strategy_counts: Counter[str] = Counter()
        self.samples = self._build_samples()
        self.access_counts = [0 for _ in self.samples]

    def set_mask_ratio(self, mask_ratio: float) -> None:
        self.current_mask_ratio = max(self.mask_ratio_min, min(self.mask_ratio_max, mask_ratio))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | dict[str, object]]:
        return self._materialize_sample(index=index, deterministic=False)

    def export_summary(self, output_dir: str | Path) -> dict[str, object]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        preview_count = min(int(self.config.get("dataset_preview_examples", 5)), len(self.samples))
        preview_examples = []
        synthetic_preview_hits = 0
        for index in range(preview_count):
            realized = self._materialize_sample(index=index, deterministic=True)
            preview_examples.append(
                {
                    "task_type": realized["task_type"],
                    "metadata": realized["metadata"],
                    "mask_metadata": realized["mask_metadata"],
                    "corrupted_code": realized["corrupted_code"],
                    "target_code": realized["target_code"],
                }
            )
            if bool(realized["metadata"].get("synthetic")):
                synthetic_preview_hits += 1

        summary = {
            "total_samples": len(self.samples),
            "source_files_total": self.num_source_files,
            "source_files_accepted": self.num_accepted_source_files,
            "task_type_distribution": dict(self.task_type_counts),
            "task_bucket_distribution": dict(self.task_bucket_counts),
            "mask_strategy_distribution": dict(self.mask_strategy_counts),
            "average_code_length_tokens": sum(sample["valid_length"] for sample in self.samples) / max(len(self.samples), 1),
            "average_quality_score": sum(float(sample["quality_score"]) for sample in self.samples) / max(len(self.samples), 1),
            "rejection_stats": dict(self.rejection_stats),
            "synthetic_generation_enabled": bool(self.config.get("synthetic_generation_enabled")),
            "synthetic_ratio_preview": synthetic_preview_hits / max(preview_count, 1),
            "preview_examples": preview_examples,
        }

        (output_path / "dataset_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        (output_path / "dataset_summary.md").write_text(
            _render_summary_markdown(summary),
            encoding="utf-8",
        )
        export_quality_reports(output_path / "file_quality_reports.json", self.file_quality_reports)
        return summary

    def _build_samples(self) -> list[dict[str, object]]:
        files = list_code_files(self.data_dir, self.extensions)
        if self.max_files is not None:
            files = files[: self.max_files]

        self.num_source_files = len(files)
        rng = random.Random(self.seed)
        samples: list[dict[str, object]] = []
        seen_file_hashes: set[str] = set()
        seen_chunk_hashes: set[str] = set()

        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            quality = assess_file_quality(path, text, self.config)
            self.file_quality_reports.append(quality)

            if not quality.accepted:
                self.rejection_stats[quality.rejection_reason or "rejected"] += 1
                continue
            if self.config.get("deduplicate_dataset", True) and quality.normalized_hash in seen_file_hashes:
                self.rejection_stats["duplicate_file"] += 1
                continue

            seen_file_hashes.add(quality.normalized_hash)
            self.num_accepted_source_files += 1

            input_ids, offsets = self._tokenize_with_offsets(text)
            if not input_ids:
                self.rejection_stats["tokenization_empty"] += 1
                continue

            for start in range(0, len(input_ids), self.seq_length):
                chunk_ids = input_ids[start : start + self.seq_length]
                chunk_offsets = offsets[start : start + self.seq_length] if offsets is not None else None
                chunk_text, normalized_offsets = _extract_chunk_text(text, chunk_offsets)

                if len(chunk_ids) < 8:
                    self.rejection_stats["chunk_too_short"] += 1
                    continue
                if chunk_text and _is_low_signal_chunk(chunk_text):
                    self.rejection_stats["low_signal_chunk"] += 1
                    continue

                chunk_quality = build_chunk_quality_metadata(chunk_text=chunk_text or "", source_report=quality)
                if self.config.get("deduplicate_dataset", True):
                    normalized_hash = str(chunk_quality["normalized_hash"])
                    if normalized_hash in seen_chunk_hashes:
                        self.rejection_stats["duplicate_chunk"] += 1
                        continue
                    seen_chunk_hashes.add(normalized_hash)

                blueprint = choose_blueprint(config=self.config, source_type=quality.source_type, rng=rng)
                sample = self._build_sample_record(
                    chunk_ids=chunk_ids,
                    normalized_offsets=normalized_offsets,
                    chunk_text=chunk_text,
                    quality=quality,
                    chunk_quality=chunk_quality,
                    blueprint=blueprint,
                )
                samples.append(sample)
                self.task_type_counts[blueprint.task_type] += 1
                self.task_bucket_counts[blueprint.task_bucket] += 1
                self.mask_strategy_counts[blueprint.mask_strategy] += 1

                if self.max_samples is not None and len(samples) >= self.max_samples:
                    return samples

        if not samples:
            raise ValueError(f"No tokenized samples found in {self.data_dir}")
        return samples

    def _build_sample_record(
        self,
        *,
        chunk_ids: list[int],
        normalized_offsets: list[tuple[int, int]] | None,
        chunk_text: str | None,
        quality: FileQualityReport,
        chunk_quality: dict[str, object],
        blueprint: SampleBlueprint,
    ) -> dict[str, object]:
        valid_length = len(chunk_ids)
        attention_mask = torch.ones(self.seq_length, dtype=torch.long)
        if valid_length < self.seq_length:
            padding = self.seq_length - valid_length
            chunk_ids = chunk_ids + [self.pad_token_id] * padding
            attention_mask[-padding:] = 0
            if normalized_offsets is not None:
                normalized_offsets.extend([(0, 0)] * padding)

        return {
            "target_ids": torch.tensor(chunk_ids, dtype=torch.long),
            "attention_mask": attention_mask,
            "text": chunk_text,
            "offset_mapping": normalized_offsets,
            "source_path": quality.path,
            "source_type": quality.source_type,
            "quality_score": float(chunk_quality["quality_score"]),
            "is_valid_target": chunk_quality.get("is_valid_target"),
            "valid_length": valid_length,
            "blueprint": blueprint,
        }

    def _materialize_sample(
        self,
        *,
        index: int,
        deterministic: bool,
    ) -> dict[str, torch.Tensor | str | dict[str, object]]:
        sample = self.samples[index]
        seed_offset = 0 if deterministic else self.access_counts[index]
        rng = random.Random(self.seed + index * 1009 + seed_offset)
        torch_seed = self.seed + index * 7919 + seed_offset

        if not deterministic:
            self.access_counts[index] += 1

        return build_training_example(
            tokenizer=self.tokenizer,
            clean_ids=sample["target_ids"],
            attention_mask=sample["attention_mask"],
            clean_text=sample["text"],
            offset_mapping=sample["offset_mapping"],
            mask_token_id=self.mask_token_id,
            current_mask_ratio=self.current_mask_ratio,
            config=self.config,
            source_path=str(sample["source_path"]),
            source_type=str(sample["source_type"]),
            quality_score=float(sample["quality_score"]),
            blueprint=sample["blueprint"],
            pad_token_id=self.pad_token_id,
            synthetic_generator=self.synthetic_generator,
            rng=rng,
            torch_seed=torch_seed,
            is_valid_target=sample["is_valid_target"],
        )

    def _tokenize_with_offsets(
        self,
        text: str,
    ) -> tuple[list[int], list[tuple[int, int]] | None]:
        try:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
                return_offsets_mapping=True,
            )
            offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
        except (NotImplementedError, TypeError, ValueError):
            encoded = self.tokenizer(text, add_special_tokens=False, truncation=False)
            offsets = None

        return list(encoded["input_ids"]), offsets


def diffusion_collate_fn(batch: list[dict[str, torch.Tensor | str | dict[str, object]]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
    }


def _build_dataset_config(
    *,
    config: dict | None,
    data_dir: str,
    seq_length: int,
    extensions: list[str],
    mask_ratio_min: float,
    mask_ratio_max: float,
) -> dict:
    merged = deepcopy(DEFAULT_CONFIG)
    if config:
        merged.update(config)
    merged["data_dir"] = str(data_dir)
    merged["seq_length"] = int(seq_length)
    merged["extensions"] = list(extensions)
    merged["mask_ratio_min"] = float(mask_ratio_min)
    merged["mask_ratio_max"] = float(mask_ratio_max)
    return merged


def _extract_chunk_text(
    full_text: str,
    offsets: list[tuple[int, int]] | None,
) -> tuple[str | None, list[tuple[int, int]] | None]:
    if not offsets:
        return None, None

    non_empty = [(start, end) for start, end in offsets if end > start]
    if not non_empty:
        return None, None

    start_char = non_empty[0][0]
    end_char = non_empty[-1][1]
    chunk_text = full_text[start_char:end_char]
    normalized = []
    for start, end in offsets:
        if end <= start:
            normalized.append((0, 0))
        else:
            normalized.append((start - start_char, end - start_char))
    return chunk_text, normalized


def _is_low_signal_chunk(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) < 24:
        return True
    alpha_chars = sum(1 for char in stripped if char.isalpha())
    if alpha_chars < 12:
        return True
    unique_lines = {line.strip() for line in text.splitlines() if line.strip()}
    if unique_lines and len(unique_lines) <= 2 and len(text.splitlines()) >= 4:
        return True
    return False


def _render_summary_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Dataset Summary",
        "",
        f"- Total samples: {summary['total_samples']}",
        f"- Source files accepted: {summary['source_files_accepted']} / {summary['source_files_total']}",
        f"- Average code length (tokens): {summary['average_code_length_tokens']:.2f}",
        f"- Average quality score: {summary['average_quality_score']:.3f}",
        f"- Synthetic generation enabled: {summary['synthetic_generation_enabled']}",
        f"- Synthetic ratio (preview): {summary['synthetic_ratio_preview']:.3f}",
        "",
        "## Task Type Distribution",
    ]
    lines.extend(f"- {name}: {count}" for name, count in sorted(summary["task_type_distribution"].items()))
    lines.append("")
    lines.append("## Task Bucket Distribution")
    lines.extend(f"- {name}: {count}" for name, count in sorted(summary["task_bucket_distribution"].items()))
    lines.append("")
    lines.append("## Mask Strategy Distribution")
    lines.extend(f"- {name}: {count}" for name, count in sorted(summary["mask_strategy_distribution"].items()))
    lines.append("")
    lines.append("## Rejection Stats")
    rejection_stats = summary.get("rejection_stats", {})
    if rejection_stats:
        lines.extend(f"- {name}: {count}" for name, count in sorted(rejection_stats.items()))
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"
