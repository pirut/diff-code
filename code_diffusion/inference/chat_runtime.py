from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import torch

from code_diffusion.config import load_config
from code_diffusion.inference import generate
from code_diffusion.models import load_diffusion_model
from code_diffusion.utils.tokenization import (
    decode_tokens,
    encode_prompt_with_masks,
    resolve_mask_token_id,
)


@dataclass(slots=True)
class PromptEnvelope:
    prompt: str
    response_marker: str
    loaded_files: list[str]
    warnings: list[str]
    comment_prefix: str


class DiffusionChatRuntime:
    def __init__(
        self,
        *,
        config_path: str,
        checkpoint: str | None = None,
        overrides: list[str] | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root or Path.cwd()).resolve()
        self.config = load_config(config_path, overrides=overrides)

        if checkpoint:
            self.config["model_name"] = checkpoint
        else:
            final_dir = Path(self.config["output_dir"]) / "final"
            if final_dir.exists():
                self.config["model_name"] = str(final_dir)

        self.model, self.tokenizer = load_diffusion_model(self.config)
        self.device = next(self.model.parameters()).device
        self.runtime_dtype = str(next(self.model.parameters()).dtype).replace("torch.", "")
        self.mask_token_id = resolve_mask_token_id(self.tokenizer)
        self._lock = threading.Lock()
        self.startup_warnings = self._build_startup_warnings()

    def chat(
        self,
        *,
        message: str,
        history: list[dict[str, str]] | None = None,
        code_context: str = "",
        draft_template: str = "",
        file_paths_text: str = "",
        steps: int | None = None,
        mask_span_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        confidence_threshold: float | None = None,
    ) -> dict[str, object]:
        envelope = self.build_prompt(
            message=message,
            history=history or [],
            code_context=code_context,
            draft_template=draft_template,
            file_paths_text=file_paths_text,
            mask_span_tokens=mask_span_tokens or int(self.config["inference_mask_span"]),
        )

        completed = self.complete_prompt(
            prompt=envelope.prompt,
            steps=steps or int(self.config["diffusion_steps"]),
            mask_span_tokens=mask_span_tokens or int(self.config["inference_mask_span"]),
            temperature=temperature if temperature is not None else float(self.config["temperature"]),
            top_k=top_k if top_k is not None else int(self.config["top_k"]),
            top_p=top_p if top_p is not None else float(self.config["top_p"]),
            confidence_threshold=(
                confidence_threshold
                if confidence_threshold is not None
                else self.config.get("confidence_threshold")
            ),
        )
        response = self.extract_response(completed=completed, response_marker=envelope.response_marker)
        return {
            "response": response,
            "full_completion": completed,
            "prompt": envelope.prompt,
            "loaded_files": envelope.loaded_files,
            "warnings": envelope.warnings,
            "comment_prefix": envelope.comment_prefix,
            "model_name": str(self.config["model_name"]),
        }

    def build_prompt(
        self,
        *,
        message: str,
        history: list[dict[str, str]],
        code_context: str,
        draft_template: str,
        file_paths_text: str,
        mask_span_tokens: int,
    ) -> PromptEnvelope:
        comment_prefix = _detect_comment_prefix(draft_template=draft_template, code_context=code_context)
        warnings: list[str] = []
        loaded_files, file_context = _load_file_context(
            file_paths_text=file_paths_text,
            repo_root=self.repo_root,
            comment_prefix=comment_prefix,
        )

        sections: list[str] = []
        if history:
            history_lines = []
            for item in history[-4:]:
                role = item.get("role", "user").strip().lower()
                content = item.get("content", "").strip()
                if not content:
                    continue
                history_lines.append(f"{role}: {content}")
            if history_lines:
                sections.append(_comment_section("Recent Conversation", "\n".join(history_lines), comment_prefix))

        if message.strip():
            sections.append(_comment_section("Request", message.strip(), comment_prefix))

        merged_context = "\n\n".join(part for part in (code_context.strip(), file_context.strip()) if part)
        if merged_context:
            sections.append(_comment_section("Code Context", merged_context, comment_prefix))

        response_marker = f"{comment_prefix} ===== RESPONSE ====="
        rendered_draft = draft_template.strip()
        if not rendered_draft:
            rendered_draft = _auto_draft_template(message=message, mask_span_tokens=mask_span_tokens)
            warnings.append("No draft template provided; generated a draft scaffold from the request.")
        elif "[MASK" not in rendered_draft:
            rendered_draft = rendered_draft.rstrip() + f"\n[MASK:{mask_span_tokens}]"
            warnings.append("Draft template had no [MASK] span; appended one at the end.")

        prompt = "\n\n".join(section for section in sections if section).strip()
        if prompt:
            prompt += "\n\n"
        prompt += response_marker + "\n" + rendered_draft.rstrip() + "\n"

        return PromptEnvelope(
            prompt=prompt,
            response_marker=response_marker,
            loaded_files=loaded_files,
            warnings=warnings,
            comment_prefix=comment_prefix,
        )

    def complete_prompt(
        self,
        *,
        prompt: str,
        steps: int,
        mask_span_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        confidence_threshold: float | None,
    ) -> str:
        prompt_ids, prompt_mask = encode_prompt_with_masks(
            prompt,
            tokenizer=self.tokenizer,
            mask_token_id=self.mask_token_id,
            default_mask_span=mask_span_tokens,
        )
        prompt_ids = prompt_ids.unsqueeze(0).to(self.device)
        prompt_mask = prompt_mask.unsqueeze(0).to(self.device)
        attention_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=self.device)

        with self._lock:
            generated = generate(
                model=self.model,
                initial_tokens=prompt_ids,
                attention_mask=attention_mask,
                initial_mask=prompt_mask,
                steps=steps,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                confidence_threshold=confidence_threshold,
                remask_fraction=float(self.config.get("remask_fraction", 0.5)),
                mask_token_id=self.mask_token_id,
            )
        return decode_tokens(self.tokenizer, generated[0].cpu())

    @staticmethod
    def extract_response(*, completed: str, response_marker: str) -> str:
        if response_marker in completed:
            extracted = completed.split(response_marker, 1)[1].lstrip("\n")
            if extracted.strip():
                return extracted
        return completed

    def _build_startup_warnings(self) -> list[str]:
        warnings: list[str] = []
        model_name = str(self.config.get("model_name", ""))
        if str(self.device) == "mps":
            warnings.append(
                "Apple Silicon local inference with Gemma 4 can take a long time per response. "
                "The UI is running, but completions may take minutes."
            )
        if "gemma-4" in model_name.lower():
            warnings.append(
                "This frontend is wrapping an infill model, not a native chat model. "
                "Best results come from a draft template with one or more [MASK:n] spans."
            )
        return warnings


def _comment_section(title: str, body: str, prefix: str) -> str:
    lines = [f"{prefix} {title}"]
    for line in body.splitlines():
        if line.strip():
            lines.append(f"{prefix} {line}")
        else:
            lines.append(prefix)
    return "\n".join(lines)


def _detect_comment_prefix(*, draft_template: str, code_context: str) -> str:
    sample = f"{draft_template}\n{code_context}".lower()
    signal = sum(token in sample for token in ("function ", "const ", "let ", "=>", ";", "{", "}"))
    return "//" if signal >= 2 else "#"


def _load_file_context(
    *,
    file_paths_text: str,
    repo_root: Path,
    comment_prefix: str,
) -> tuple[list[str], str]:
    loaded_files: list[str] = []
    rendered_blocks: list[str] = []

    requested = [line.strip() for line in file_paths_text.splitlines() if line.strip()]
    for raw_path in requested[:8]:
        resolved = (repo_root / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path).resolve()
        if repo_root not in resolved.parents and resolved != repo_root:
            continue
        if not resolved.is_file():
            continue
        try:
            text = resolved.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        relative = str(resolved.relative_to(repo_root))
        loaded_files.append(relative)
        snippet = text[:6000]
        rendered_blocks.append(f"{comment_prefix} File: {relative}\n{snippet}".rstrip())

    return loaded_files, "\n\n".join(rendered_blocks)


def serialize_chat_result(result: dict[str, object]) -> str:
    return json.dumps(result, indent=2)


def _auto_draft_template(*, message: str, mask_span_tokens: int) -> str:
    lower = message.lower()
    span = max(mask_span_tokens, 32)

    if "button" in lower and any(token in lower for token in ("ts", "tsx", "typescript", "react")):
        component_name = _guess_component_name(message, fallback="HelloWorldButton")
        return (
            "import React from \"react\";\n\n"
            f"export function {component_name}() {{\n"
            f"  [MASK:{span}]\n"
            "}\n"
        )

    if any(token in lower for token in ("typescript", "tsx", "ts ", "react", "component")):
        symbol = _guess_component_name(message, fallback="ExampleComponent")
        return (
            f"export function {symbol}() {{\n"
            f"  [MASK:{span}]\n"
            "}\n"
        )

    if any(token in lower for token in ("python", "py ", "def ", "function")):
        func_name = _guess_function_name(message, fallback="implement_feature")
        return (
            f"def {func_name}():\n"
            f"    [MASK:{max(mask_span_tokens, 16)}]\n"
        )

    return f"[MASK:{span}]"


def _guess_component_name(message: str, *, fallback: str) -> str:
    candidates = re.findall(r"[A-Za-z][A-Za-z0-9]+", message)
    interesting = [token for token in candidates if len(token) > 2 and token.lower() not in {"typescript", "react", "button", "implement"}]
    if not interesting:
        return fallback
    return "".join(token[:1].upper() + token[1:] for token in interesting[:3])


def _guess_function_name(message: str, *, fallback: str) -> str:
    candidates = re.findall(r"[A-Za-z][A-Za-z0-9_]+", message.lower())
    filtered = [token for token in candidates if token not in {"python", "function", "implement", "create", "make"}]
    if not filtered:
        return fallback
    return "_".join(filtered[:3])
