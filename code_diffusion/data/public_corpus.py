from __future__ import annotations

import ast
import json
import re
import shutil
import warnings
from collections import Counter, defaultdict
from hashlib import sha1
from itertools import islice
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from code_diffusion.utils.tokenization import list_code_files


CODESEARCHNET_LANGS = {"python", "javascript", "go", "java", "php", "ruby"}
FILE_EXTENSION_MAP = {
    "python": ".py",
    "py": ".py",
    "javascript": ".js",
    "js": ".js",
    "typescript": ".ts",
    "ts": ".ts",
    "go": ".go",
    "rust": ".rs",
    "java": ".java",
    "php": ".php",
    "ruby": ".rb",
}


def prepare_public_corpus(
    *,
    config: dict,
    output_dir: str | Path | None = None,
    clean_output: bool = True,
) -> dict[str, object]:
    output_root = Path(output_dir or config.get("prepare_output_dir") or "./outputs/prepared_corpus/public_mix_v1").resolve()
    if clean_output and output_root.exists():
        shutil.rmtree(output_root)
    files_root = output_root / "files"
    prepared_examples_path = output_root / str(config.get("prepared_examples_filename", "prepared_examples.jsonl"))
    files_root.mkdir(parents=True, exist_ok=True)
    prepared_examples_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "output_dir": str(output_root),
        "prepared_examples_path": str(prepared_examples_path),
        "raw_file_count": 0,
        "prepared_example_count": 0,
        "source_counts": Counter(),
        "language_counts": Counter(),
        "task_type_counts": Counter(),
        "skipped_counts": Counter(),
    }

    with prepared_examples_path.open("w", encoding="utf-8") as prepared_handle:
        if config.get("prepare_include_local_data", True):
            _copy_local_files(config=config, output_root=files_root, summary=summary)
        if config.get("prepare_include_codesearchnet", True):
            _ingest_codesearchnet(config=config, output_root=files_root, summary=summary)
        if config.get("prepare_include_commitpackft", True):
            _ingest_commitpackft(config=config, handle=prepared_handle, summary=summary)
        if config.get("prepare_include_swe_rebench", True):
            _ingest_swe_rebench(config=config, handle=prepared_handle, summary=summary)

    final_summary = {
        **summary,
        "source_counts": dict(summary["source_counts"]),
        "language_counts": dict(summary["language_counts"]),
        "task_type_counts": dict(summary["task_type_counts"]),
        "skipped_counts": dict(summary["skipped_counts"]),
    }
    (output_root / "prepare_summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    (output_root / "prepare_summary.md").write_text(_render_summary_markdown(final_summary), encoding="utf-8")
    return final_summary


def _copy_local_files(*, config: dict, output_root: Path, summary: dict[str, object]) -> None:
    source_dir = Path(config["data_dir"]).resolve()
    output_parent = output_root.resolve()
    if not source_dir.exists() or source_dir == output_parent or output_parent in source_dir.parents:
        return

    local_files = list_code_files(source_dir, config["extensions"])
    for path in islice(local_files, int(config.get("prepare_max_local_files", 200))):
        relative = path.relative_to(source_dir)
        destination = output_root / "local" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        summary["raw_file_count"] += 1
        summary["source_counts"]["local_files"] += 1
        summary["language_counts"][path.suffix.lower()] += 1


def _ingest_codesearchnet(*, config: dict, output_root: Path, summary: dict[str, object]) -> None:
    requested_langs = [lang for lang in config.get("prepare_languages", []) if lang in CODESEARCHNET_LANGS]
    per_language = int(config.get("prepare_codesearchnet_examples_per_language", 200))

    for language in requested_langs:
        dataset = load_dataset("code_search_net", language, split="train", streaming=True)
        written = 0
        for row in dataset:
            code = str(row.get("whole_func_string") or row.get("func_code_string") or "")
            if not _is_useful_code(code):
                summary["skipped_counts"]["codesearchnet_low_signal"] += 1
                continue
            repository = str(row.get("repository_name") or "repo")
            path_in_repo = str(row.get("func_path_in_repository") or row.get("func_name") or "snippet")
            extension = _infer_extension(language=language, path=path_in_repo)
            file_id = sha1(f"{repository}:{path_in_repo}:{row.get('func_name','')}".encode("utf-8")).hexdigest()[:12]
            destination = output_root / "codesearchnet" / language / f"{_slugify(repository)}__{file_id}{extension}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(code, encoding="utf-8")
            written += 1
            summary["raw_file_count"] += 1
            summary["source_counts"]["code_search_net"] += 1
            summary["language_counts"][language] += 1
            if written >= per_language:
                break


def _ingest_commitpackft(*, config: dict, handle, summary: dict[str, object]) -> None:
    requested_langs = [lang for lang in config.get("prepare_languages", []) if lang in FILE_EXTENSION_MAP]
    per_language = int(config.get("prepare_commitpack_examples_per_language", 200))
    context_lines = int(config.get("prepare_context_lines", 24))
    max_chars = int(config.get("prepare_max_example_chars", 12000))

    for language in requested_langs:
        filename = f"data/{language}/data.jsonl"
        try:
            local_path = hf_hub_download(repo_id="bigcode/commitpackft", repo_type="dataset", filename=filename)
        except Exception:
            summary["skipped_counts"][f"commitpackft_missing_{language}"] += 1
            continue

        written = 0
        with Path(local_path).open("r", encoding="utf-8") as source_handle:
            for raw_line in source_handle:
                row = json.loads(raw_line)
                before, after = _extract_changed_window(
                    before_text=str(row.get("old_contents") or ""),
                    after_text=str(row.get("new_contents") or ""),
                    context_lines=context_lines,
                    max_chars=max_chars,
                    path=str(row.get("new_file") or row.get("old_file") or ""),
                )
                if not before or not after or before == after:
                    summary["skipped_counts"]["commitpackft_unchanged_or_empty"] += 1
                    continue

                task_type = _classify_commit_task(str(row.get("subject") or row.get("message") or ""))
                source_path = f"{row.get('repos','unknown')}:{row.get('new_file') or row.get('old_file')}"
                context_prefix = _build_repo_edit_context(
                    path=str(row.get("new_file") or row.get("old_file") or ""),
                    repo=str(row.get("repos") or "unknown"),
                    source_dataset="bigcode/commitpackft",
                    include_repo_context=bool(config.get("prepare_include_repo_context", True)),
                    include_commit_context=bool(config.get("prepare_include_commit_context", True)),
                    include_issue_context=False,
                    max_lines=int(config.get("prepare_context_max_lines", 8)),
                    subject=str(row.get("subject") or ""),
                    message=str(row.get("message") or ""),
                )
                before = context_prefix + before
                after = context_prefix + after
                record = {
                    "task_type": task_type,
                    "corrupted_code": before,
                    "target_code": after,
                    "mask_metadata": {
                        "source_dataset": "bigcode/commitpackft",
                        "pair_strategy": "diff_window",
                        "commit": row.get("commit"),
                        "message": row.get("message"),
                    },
                    "metadata": {
                        "mask_strategy": "prepared_pair",
                        "source_type": "implementation",
                        "source_path": source_path,
                        "source_dataset": "bigcode/commitpackft",
                        "quality_score": 0.92,
                        "task_bucket": task_type,
                        "target_is_valid": _validate_target_for_language(after, language),
                        "synthetic": False,
                        "language": language,
                        "license": row.get("license"),
                        "subject": row.get("subject"),
                        "message": row.get("message"),
                        "repos": row.get("repos"),
                    },
                }
                handle.write(json.dumps(record) + "\n")
                written += 1
                summary["prepared_example_count"] += 1
                summary["source_counts"]["bigcode/commitpackft"] += 1
                summary["language_counts"][language] += 1
                summary["task_type_counts"][task_type] += 1
                if written >= per_language:
                    break


def _ingest_swe_rebench(*, config: dict, handle, summary: dict[str, object]) -> None:
    requested_langs = {_normalize_language_name(lang) for lang in config.get("prepare_languages", [])}
    per_language = int(config.get("prepare_swe_rebench_examples_per_language", 75))
    max_chars = int(config.get("prepare_max_example_chars", 12000))
    counters: defaultdict[str, int] = defaultdict(int)

    dataset = load_dataset("nebius/SWE-rebench-V2", split="train", streaming=True)
    for row in dataset:
        language = _normalize_language_name(str(row.get("language") or ""))
        if language not in requested_langs or counters[language] >= per_language:
            continue

        records = list(
            _extract_swe_records(
                row=row,
                max_chars=max_chars,
                config=config,
            )
        )
        if not records:
            summary["skipped_counts"]["swe_rebench_no_patch_records"] += 1
            continue

        for record in records:
            handle.write(json.dumps(record) + "\n")
            counters[language] += 1
            summary["prepared_example_count"] += 1
            summary["source_counts"]["nebius/SWE-rebench-V2"] += 1
            summary["language_counts"][language] += 1
            summary["task_type_counts"][record["task_type"]] += 1
            if counters[language] >= per_language:
                break

        if all(counters[lang] >= per_language for lang in requested_langs if lang in counters or lang in requested_langs):
            # The dataset is large; stop once all requested quotas are met.
            if all(counters[lang] >= per_language for lang in requested_langs):
                break


def _extract_swe_records(*, row: dict[str, object], max_chars: int, config: dict) -> Iterable[dict[str, object]]:
    language = _normalize_language_name(str(row.get("language") or ""))
    base_metadata = {
        "source_dataset": "nebius/SWE-rebench-V2",
        "quality_score": 0.97,
        "synthetic": False,
        "language": language,
        "repo": row.get("repo"),
        "instance_id": row.get("instance_id"),
        "problem_statement": row.get("problem_statement"),
        "pr_description": row.get("pr_description"),
        "interface": row.get("interface"),
        "license": row.get("license"),
    }

    for source_type, patch_text in (
        ("implementation", str(row.get("patch") or "")),
        ("test", str(row.get("test_patch") or "")),
    ):
        for path, before, after in _extract_unified_diff_pairs(patch_text, max_chars=max_chars):
            if not before or not after or before == after:
                continue
            context_prefix = _build_repo_edit_context(
                path=path,
                repo=str(row.get("repo") or "unknown"),
                source_dataset="nebius/SWE-rebench-V2",
                include_repo_context=bool(config.get("prepare_include_repo_context", True)),
                include_commit_context=False,
                include_issue_context=bool(config.get("prepare_include_issue_context", True)),
                max_lines=int(config.get("prepare_context_max_lines", 8)),
                problem_statement=str(row.get("problem_statement") or ""),
                pr_description=str(row.get("pr_description") or ""),
                interface=str(row.get("interface") or ""),
                instance_id=str(row.get("instance_id") or ""),
            )
            yield {
                "task_type": "bug_fix",
                "corrupted_code": context_prefix + before,
                "target_code": context_prefix + after,
                "mask_metadata": {
                    "source_dataset": "nebius/SWE-rebench-V2",
                    "pair_strategy": "patch_hunk",
                    "instance_id": row.get("instance_id"),
                    "path": path,
                },
                "metadata": {
                    **base_metadata,
                    "mask_strategy": "prepared_pair",
                    "source_type": "test" if _is_test_path(path) or source_type == "test" else "implementation",
                    "source_path": f"{row.get('repo')}:{path}",
                    "task_bucket": "bug_fix",
                    "target_is_valid": _validate_target_for_language(after, language),
                },
            }


def _extract_changed_window(
    *,
    before_text: str,
    after_text: str,
    context_lines: int,
    max_chars: int,
    path: str,
) -> tuple[str | None, str | None]:
    before_lines = before_text.splitlines()
    after_lines = after_text.splitlines()
    matcher = __import__("difflib").SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
    changed = [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"]
    if not changed:
        return None, None

    before_start = max(0, min(i1 for i1, _, _, _ in changed) - context_lines)
    before_end = min(len(before_lines), max(i2 for _, i2, _, _ in changed) + context_lines)
    after_start = max(0, min(j1 for _, _, j1, _ in changed) - context_lines)
    after_end = min(len(after_lines), max(j2 for _, _, _, j2 in changed) + context_lines)

    header = _comment_prefix_for_path(path) + f" path: {path}\n"
    before_snippet = header + "\n".join(before_lines[before_start:before_end]).strip() + "\n"
    after_snippet = header + "\n".join(after_lines[after_start:after_end]).strip() + "\n"
    if len(before_snippet) > max_chars:
        before_snippet = before_snippet[:max_chars]
    if len(after_snippet) > max_chars:
        after_snippet = after_snippet[:max_chars]
    return before_snippet, after_snippet


def _extract_unified_diff_pairs(patch_text: str, *, max_chars: int) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    current_path: str | None = None
    before_lines: list[str] = []
    after_lines: list[str] = []

    def flush() -> None:
        nonlocal current_path, before_lines, after_lines
        if current_path and before_lines and after_lines:
            header = _comment_prefix_for_path(current_path) + f" path: {current_path}\n"
            before = header + "\n".join(before_lines).strip() + "\n"
            after = header + "\n".join(after_lines).strip() + "\n"
            pairs.append((current_path, before[:max_chars], after[:max_chars]))
        current_path = None
        before_lines = []
        after_lines = []

    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            flush()
            continue
        if line.startswith("+++ b/"):
            current_path = line[6:]
            continue
        if line.startswith("--- a/") or line.startswith("@@") or not current_path:
            continue
        if line.startswith(" "):
            before_lines.append(line[1:])
            after_lines.append(line[1:])
        elif line.startswith("-"):
            before_lines.append(line[1:])
        elif line.startswith("+"):
            after_lines.append(line[1:])
    flush()
    return [(path, before, after) for path, before, after in pairs if before and after and before != after]


def _classify_commit_task(message: str) -> str:
    lowered = message.lower()
    if any(keyword in lowered for keyword in ("fix", "bug", "repair", "correct", "regression")):
        return "bug_fix"
    return "refinement"


def _build_repo_edit_context(
    *,
    path: str,
    repo: str,
    source_dataset: str,
    include_repo_context: bool,
    include_commit_context: bool,
    include_issue_context: bool,
    max_lines: int,
    subject: str = "",
    message: str = "",
    problem_statement: str = "",
    pr_description: str = "",
    interface: str = "",
    instance_id: str = "",
) -> str:
    prefix = _comment_prefix_for_path(path)
    lines: list[str] = []
    if include_repo_context:
        lines.append(f"{prefix} repo: {repo}")
        lines.append(f"{prefix} path: {path}")
        lines.append(f"{prefix} source_dataset: {source_dataset}")
        if instance_id:
            lines.append(f"{prefix} instance_id: {instance_id}")
    if include_commit_context:
        if subject:
            lines.append(f"{prefix} change_request: {subject.strip()}")
        for line in _trim_text_lines(message, max_lines=max_lines):
            lines.append(f"{prefix} change_detail: {line}")
    if include_issue_context:
        for line in _trim_text_lines(problem_statement, max_lines=max_lines):
            lines.append(f"{prefix} issue: {line}")
        for line in _trim_text_lines(pr_description, max_lines=max_lines):
            lines.append(f"{prefix} patch_note: {line}")
        for line in _trim_text_lines(interface, max_lines=max_lines):
            lines.append(f"{prefix} interface_note: {line}")

    if not lines:
        return ""
    return "\n".join(lines[:max_lines]) + "\n\n"


def _normalize_language_name(language: str) -> str:
    aliases = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
    }
    return aliases.get(language.lower(), language.lower())


def _infer_extension(*, language: str, path: str) -> str:
    suffix = Path(path).suffix
    if suffix:
        return suffix
    return FILE_EXTENSION_MAP.get(language, ".txt")


def _validate_target_for_language(text: str, language: str) -> bool | None:
    if _normalize_language_name(language) != "python":
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(text)
    except SyntaxError:
        return False
    return True


def _is_test_path(path: str) -> bool:
    lowered = path.lower()
    return any(part in lowered for part in ("test", "tests", "spec", "__tests__"))


def _is_useful_code(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 32:
        return False
    alpha = sum(1 for char in stripped if char.isalpha())
    return alpha >= 16


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower() or "item"


def _comment_prefix_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".py", ".sh", ".rb", ".yaml", ".yml"}:
        return "#"
    return "//"


def _trim_text_lines(text: str, *, max_lines: int) -> list[str]:
    cleaned = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    cleaned = [line for line in cleaned if line]
    return cleaned[:max_lines]


def _render_summary_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Prepared Corpus Summary",
        "",
        f"- Output dir: {summary['output_dir']}",
        f"- Raw files: {summary['raw_file_count']}",
        f"- Prepared examples: {summary['prepared_example_count']}",
        "",
        "## Source Counts",
    ]
    lines.extend(f"- {name}: {count}" for name, count in sorted(summary["source_counts"].items()))
    lines.append("")
    lines.append("## Language Counts")
    lines.extend(f"- {name}: {count}" for name, count in sorted(summary["language_counts"].items()))
    lines.append("")
    lines.append("## Task Type Counts")
    lines.extend(f"- {name}: {count}" for name, count in sorted(summary["task_type_counts"].items()))
    lines.append("")
    lines.append("## Skipped")
    if summary["skipped_counts"]:
        lines.extend(f"- {name}: {count}" for name, count in sorted(summary["skipped_counts"].items()))
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"
