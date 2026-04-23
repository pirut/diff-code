from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SyntheticResult:
    task_type: str
    corrupted_code: str
    target_code: str
    prompt_path: str
    response_path: str
    provider: str
    model: str


class SyntheticExampleGenerator:
    def __init__(self, config: dict) -> None:
        self.enabled = bool(config.get("synthetic_generation_enabled"))
        self.provider = config.get("synthetic_provider") or "openai_compatible"
        self.model = config.get("synthetic_model")
        self.base_url = str(config.get("synthetic_base_url", "https://api.openai.com/v1")).rstrip("/")
        self.api_key_env = str(config.get("synthetic_api_key_env", "OPENAI_API_KEY"))
        self.timeout_seconds = int(config.get("synthetic_timeout_seconds", 60))
        self.cache_dir = Path(config.get("synthetic_cache_dir", "./outputs/synthetic_cache")).resolve()
        self.prompts_dir = self.cache_dir / "prompts"
        self.responses_dir = self.cache_dir / "responses"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)

    def maybe_generate(
        self,
        *,
        task_type: str,
        clean_code: str,
        source_path: str,
        source_type: str,
    ) -> SyntheticResult | None:
        if not self.enabled or not self.model:
            return None

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            return None

        prompt = _build_prompt(
            task_type=task_type,
            clean_code=clean_code,
            source_path=source_path,
            source_type=source_type,
        )
        cache_key = _cache_key(
            provider=self.provider,
            model=self.model,
            task_type=task_type,
            clean_code=clean_code,
            source_path=source_path,
        )
        prompt_path = self.prompts_dir / f"{cache_key}.json"
        response_path = self.responses_dir / f"{cache_key}.json"

        if response_path.exists():
            payload = json.loads(response_path.read_text(encoding="utf-8"))
            return SyntheticResult(
                task_type=task_type,
                corrupted_code=str(payload["corrupted_code"]),
                target_code=str(payload["target_code"]),
                prompt_path=str(prompt_path),
                response_path=str(response_path),
                provider=self.provider,
                model=self.model,
            )

        prompt_payload = {
            "task_type": task_type,
            "source_path": source_path,
            "source_type": source_type,
            "prompt": prompt,
        }
        prompt_path.write_text(json.dumps(prompt_payload, indent=2), encoding="utf-8")

        try:
            payload = self._request_openai_compatible(prompt=prompt, api_key=api_key)
        except (OSError, ValueError, urllib.error.URLError):
            return None

        response_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if "corrupted_code" not in payload or "target_code" not in payload:
            return None

        return SyntheticResult(
            task_type=task_type,
            corrupted_code=str(payload["corrupted_code"]),
            target_code=str(payload["target_code"]),
            prompt_path=str(prompt_path),
            response_path=str(response_path),
            provider=self.provider,
            model=self.model,
        )

    def _request_openai_compatible(self, *, prompt: str, api_key: str) -> dict[str, object]:
        if self.provider not in {"openai", "openai_compatible"}:
            raise ValueError(f"Unsupported synthetic provider: {self.provider}")

        body = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You generate code-diffusion training pairs. "
                        "Return JSON with keys corrupted_code and target_code only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))

        message = payload["choices"][0]["message"]["content"]
        if isinstance(message, list):
            message = "".join(part.get("text", "") for part in message if isinstance(part, dict))
        if not isinstance(message, str):
            raise ValueError("Unexpected synthetic response format.")
        return json.loads(message)


def _cache_key(
    *,
    provider: str,
    model: str,
    task_type: str,
    clean_code: str,
    source_path: str,
) -> str:
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "task_type": task_type,
            "clean_code": clean_code,
            "source_path": source_path,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_prompt(
    *,
    task_type: str,
    clean_code: str,
    source_path: str,
    source_type: str,
) -> str:
    if task_type == "bug_fix":
        instruction = (
            "Create a realistically broken version of this code by introducing a small, reviewable bug. "
            "Do not change the programming language. Keep the code length roughly similar."
        )
    elif task_type == "refinement":
        instruction = (
            "Create a weaker draft of this code by partially deleting or simplifying logic while keeping the file coherent."
        )
    elif task_type == "fim":
        instruction = "Remove a meaningful middle span from this code."
    else:
        instruction = "Create a masked or degraded version of this code for denoising."

    return (
        f"Source path: {source_path}\n"
        f"Source type: {source_type}\n"
        f"Task type: {task_type}\n"
        f"{instruction}\n\n"
        "Return strict JSON with:\n"
        '- "corrupted_code": the degraded input\n'
        '- "target_code": the original clean code\n\n'
        f"Original code:\n```code\n{clean_code}\n```"
    )
