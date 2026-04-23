from __future__ import annotations

import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import Dataset

from code_diffusion.config import DEFAULT_CONFIG
from code_diffusion.data.example_builder import (
    SampleBlueprint,
    build_prepared_training_example,
    build_training_example,
    choose_blueprint,
)
from code_diffusion.data.quality import (
    FileQualityReport,
    assess_file_quality,
    build_chunk_quality_metadata,
    export_quality_reports,
    hash_normalized_code,
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
        self.num_prepared_records = 0
        self.source_dataset_counts: Counter[str] = Counter()
        self.file_quality_reports: list[FileQualityReport] = []
        self.rejection_stats: Counter[str] = Counter()
        self.task_type_counts: Counter[str] = Counter()
        self.task_bucket_counts: Counter[str] = Counter()
        self.mask_strategy_counts: Counter[str] = Counter()
        self.samples = self._build_samples()
        self.access_counts = [0 for _ in self.samples]
        self.sample_task_buckets = [self._resolve_task_bucket(sample) for sample in self.samples]
        self.current_task_type_weights = _normalize_weight_mapping(self.config.get("task_type_weights", {}))

    def set_mask_ratio(self, mask_ratio: float) -> None:
        self.current_mask_ratio = max(self.mask_ratio_min, min(self.mask_ratio_max, mask_ratio))

    def get_task_type_weights(self) -> dict[str, float]:
        return deepcopy(self.current_task_type_weights)

    def set_task_type_weights(self, task_type_weights: dict[str, float]) -> dict[str, float]:
        normalized = _normalize_weight_mapping(task_type_weights)
        self.current_task_type_weights = normalized
        self.config["task_type_weights"] = deepcopy(normalized)
        return deepcopy(normalized)

    def get_weighted_sample_weights(
        self,
        task_type_weights: dict[str, float] | None = None,
    ) -> torch.DoubleTensor:
        normalized = _normalize_weight_mapping(task_type_weights or self.current_task_type_weights)
        if not normalized:
            return torch.ones(len(self.samples), dtype=torch.double)

        bucket_counts = Counter(self.sample_task_buckets)
        fallback_weight = 1.0 / max(len(normalized), 1)
        weights = []
        for bucket in self.sample_task_buckets:
            bucket_weight = float(normalized.get(bucket, fallback_weight))
            count = max(bucket_counts.get(bucket, 0), 1)
            weights.append(bucket_weight / count)
        return torch.tensor(weights, dtype=torch.double)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | dict[str, object]]:
        return self._materialize_sample(index=index, deterministic=False)

    def get_example(
        self,
        index: int,
        *,
        deterministic: bool = True,
    ) -> dict[str, torch.Tensor | str | dict[str, object]]:
        return self._materialize_sample(index=index, deterministic=deterministic)

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
            "prepared_records": self.num_prepared_records,
            "source_dataset_distribution": dict(self.source_dataset_counts),
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
        rng = random.Random(self.seed)
        samples: list[dict[str, object]] = []
        samples.extend(self._load_prepared_samples())
        if self.max_samples is not None and len(samples) >= self.max_samples:
            return samples[: self.max_samples]

        files = _safe_list_code_files(self.data_dir, self.extensions)
        if self.max_files is not None:
            files = files[: self.max_files]

        self.num_source_files = len(files)
        seen_file_hashes: set[str] = set()
        seen_chunk_hashes: set[str] = set()
        for sample in samples:
            if sample.get("record_type") == "prepared":
                seen_chunk_hashes.add(str(sample["dedupe_hash"]))

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

    def _load_prepared_samples(self) -> list[dict[str, object]]:
        prepared_paths = sorted(self.data_dir.rglob(self.config.get("prepared_examples_filename", "prepared_examples.jsonl")))
        if not prepared_paths:
            return []

        samples: list[dict[str, object]] = []
        seen_hashes: set[str] = set()
        for path in prepared_paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                sample = self._build_prepared_sample_record(record)
                if sample is None:
                    continue
                if self.config.get("deduplicate_dataset", True) and sample["dedupe_hash"] in seen_hashes:
                    self.rejection_stats["duplicate_prepared_record"] += 1
                    continue
                seen_hashes.add(str(sample["dedupe_hash"]))
                samples.append(sample)
                self.num_prepared_records += 1
                self.task_type_counts[str(sample["task_type"])] += 1
                self.task_bucket_counts[str(sample["metadata"].get("task_bucket", sample["task_type"]))] += 1
                self.mask_strategy_counts[str(sample["metadata"].get("mask_strategy", "prepared_pair"))] += 1
                self.source_dataset_counts[str(sample["metadata"].get("source_dataset", "prepared"))] += 1
                if self.max_samples is not None and len(samples) >= self.max_samples:
                    return samples
        return samples

    def _build_prepared_sample_record(self, record: dict[str, object]) -> dict[str, object] | None:
        target_code = record.get("target_code")
        corrupted_code = record.get("corrupted_code")
        task_type = str(record.get("task_type", "masked_reconstruction"))
        metadata = dict(record.get("metadata") or {})
        mask_metadata = dict(record.get("mask_metadata") or {})
        if not isinstance(target_code, str) or not isinstance(corrupted_code, str):
            self.rejection_stats["prepared_missing_text"] += 1
            return None
        if _is_low_signal_chunk(target_code):
            self.rejection_stats["prepared_low_signal"] += 1
            return None

        dedupe_hash = hash_normalized_code(f"{task_type}\n{target_code}\n{corrupted_code}")
        quality_score = float(metadata.get("quality_score", 0.9))
        source_type = str(metadata.get("source_type", "implementation"))
        source_path = str(metadata.get("source_path", metadata.get("source_id", "prepared")))
        valid_length = len(self.tokenizer(target_code, add_special_tokens=False, truncation=False)["input_ids"])
        if valid_length < 8:
            self.rejection_stats["prepared_too_short"] += 1
            return None

        metadata.setdefault("mask_strategy", "prepared_pair")
        metadata.setdefault("task_bucket", task_type)
        metadata.setdefault("source_dataset", "prepared")
        metadata.setdefault("synthetic", False)
        metadata.setdefault("source_type", source_type)
        metadata.setdefault("source_path", source_path)

        return {
            "record_type": "prepared",
            "task_type": task_type,
            "target_code": target_code,
            "corrupted_code": corrupted_code,
            "mask_metadata": mask_metadata,
            "metadata": metadata,
            "quality_score": quality_score,
            "source_type": source_type,
            "source_path": source_path,
            "is_valid_target": metadata.get("target_is_valid"),
            "valid_length": min(valid_length, self.seq_length),
            "dedupe_hash": dedupe_hash,
        }

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

        if sample.get("record_type") == "prepared":
            return build_prepared_training_example(
                tokenizer=self.tokenizer,
                task_type=str(sample["task_type"]),
                target_code=str(sample["target_code"]),
                corrupted_code=str(sample["corrupted_code"]),
                metadata=dict(sample["metadata"]),
                mask_metadata=dict(sample["mask_metadata"]),
                seq_length=self.seq_length,
                pad_token_id=self.pad_token_id,
                mask_token_id=self.mask_token_id,
            )

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

    def _resolve_task_bucket(self, sample: dict[str, object]) -> str:
        if sample.get("record_type") == "prepared":
            metadata = sample.get("metadata")
            if isinstance(metadata, dict):
                return str(metadata.get("task_bucket", sample.get("task_type", "masked_reconstruction")))
            return str(sample.get("task_type", "masked_reconstruction"))
        blueprint = sample.get("blueprint")
        if blueprint is not None and hasattr(blueprint, "task_bucket"):
            return str(blueprint.task_bucket)
        return "masked_reconstruction"

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


def _safe_list_code_files(data_dir: Path, extensions: list[str]) -> list[Path]:
    try:
        return list_code_files(data_dir, extensions)
    except FileNotFoundError:
        return []


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


def _normalize_weight_mapping(weights: dict[str, object] | None) -> dict[str, float]:
    if not weights:
        return {}

    normalized = {str(key): max(float(value), 0.0) for key, value in weights.items()}
    total = sum(normalized.values())
    if total <= 0:
        uniform = 1.0 / max(len(normalized), 1)
        return {key: uniform for key in normalized}
    return {key: value / total for key, value in normalized.items()}
