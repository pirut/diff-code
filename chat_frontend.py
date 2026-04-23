from __future__ import annotations

import argparse
import json
import mimetypes
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from code_diffusion.inference.chat_runtime import DiffusionChatRuntime


ASSETS_DIR = Path(__file__).resolve().parent / "frontend" / "chat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local browser chat UI for the code diffusion model.")
    parser.add_argument("--config", default="config.public-data.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path to load.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a config value with key=value. Can be passed multiple times.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=7860, help="Bind port.")
    parser.add_argument("--repo-root", default=".", help="Workspace root for loading file context.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = DiffusionChatRuntime(
        config_path=args.config,
        checkpoint=args.checkpoint,
        overrides=args.override,
        repo_root=args.repo_root,
    )

    handler = partial(ChatHandler, runtime=runtime, repo_root=Path(args.repo_root).resolve())
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Code Diffusion Chat running at http://{args.host}:{args.port}")
    server.serve_forever()


class ChatHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, runtime: DiffusionChatRuntime, repo_root: Path, **kwargs) -> None:
        self.runtime = runtime
        self.repo_root = repo_root
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        if self.path == "/api/state":
            self._send_json(
                {
                    "ok": True,
                    "model_name": str(self.runtime.config["model_name"]),
                    "config_path": str(self.runtime.config["config_path"]),
                    "repo_root": str(self.repo_root),
                    "device": str(self.runtime.device),
                    "runtime_dtype": self.runtime.runtime_dtype,
                    "default_steps": int(self.runtime.config["diffusion_steps"]),
                    "default_mask_span": int(self.runtime.config["inference_mask_span"]),
                    "startup_warnings": list(self.runtime.startup_warnings),
                }
            )
            return

        asset_path = self._resolve_asset_path(self.path)
        if asset_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        content = asset_path.read_bytes()
        content_type = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"
        try:
            payload = json.loads(raw_body or "{}")
            result = self.runtime.chat(
                message=str(payload.get("message", "")),
                history=list(payload.get("history") or []),
                code_context=str(payload.get("code_context", "")),
                draft_template=str(payload.get("draft_template", "")),
                file_paths_text=str(payload.get("file_paths_text", "")),
                steps=_maybe_int(payload.get("steps")),
                mask_span_tokens=_maybe_int(payload.get("mask_span_tokens")),
                temperature=_maybe_float(payload.get("temperature")),
                top_k=_maybe_int(payload.get("top_k")),
                top_p=_maybe_float(payload.get("top_p")),
                confidence_threshold=_maybe_float(payload.get("confidence_threshold")),
            )
            self._send_json({"ok": True, **result})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, fmt: str, *args) -> None:
        return

    def _resolve_asset_path(self, raw_path: str) -> Path | None:
        path = raw_path.split("?", 1)[0]
        if path in {"", "/"}:
            path = "/index.html"
        candidate = (ASSETS_DIR / path.lstrip("/")).resolve()
        if ASSETS_DIR not in candidate.parents and candidate != ASSETS_DIR:
            return None
        if not candidate.exists() or not candidate.is_file():
            return None
        return candidate

    def _send_json(self, payload: dict[str, object], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _maybe_int(value) -> int | None:
    if value in (None, "", "null"):
        return None
    return int(value)


def _maybe_float(value) -> float | None:
    if value in (None, "", "null"):
        return None
    return float(value)


if __name__ == "__main__":
    main()
