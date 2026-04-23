from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import modal

from code_diffusion.inference.chat_runtime import DiffusionChatRuntime

APP_NAME = os.environ.get("CODE_DIFFUSION_MODAL_CHAT_APP", "code-diffusion-chat")
OUTPUT_VOLUME_NAME = os.environ.get(
    "CODE_DIFFUSION_MODAL_OUTPUTS_VOLUME",
    "code-diffusion-outputs",
)
CACHE_VOLUME_NAME = os.environ.get(
    "CODE_DIFFUSION_MODAL_CACHE_VOLUME",
    "code-diffusion-hf-cache",
)
GPU_TYPE = os.environ.get("CODE_DIFFUSION_MODAL_GPU", "A100-80GB")
CPU_COUNT = float(os.environ.get("CODE_DIFFUSION_MODAL_CPU", "8"))
MEMORY_MB = int(os.environ.get("CODE_DIFFUSION_MODAL_MEMORY_MB", str(64 * 1024)))
TIMEOUT_SECONDS = int(os.environ.get("CODE_DIFFUSION_MODAL_TIMEOUT_SECONDS", str(24 * 60 * 60)))
HF_SECRET_NAME = os.environ.get("CODE_DIFFUSION_MODAL_HF_SECRET") or "code-diffusion-hf"

CHAT_CONFIG_PATH = os.environ.get("CODE_DIFFUSION_CHAT_CONFIG", "/workspace/config.public-data.yaml")
CHAT_RUN_NAME = os.environ.get("CODE_DIFFUSION_CHAT_RUN_NAME", "gemma4-public-poc-50-bench-ctrl")
CHAT_CHECKPOINT_SUBDIR = os.environ.get("CODE_DIFFUSION_CHAT_CHECKPOINT_SUBDIR", "best")
CHAT_LABEL = os.environ.get("CODE_DIFFUSION_MODAL_CHAT_LABEL", "code-diffusion-chat")

REMOTE_OUTPUT_ROOT = Path("/mnt/outputs")
REMOTE_CACHE_ROOT = Path("/mnt/hf")
WORKSPACE_ROOT = Path("/workspace")
FRONTEND_ROOT = WORKSPACE_ROOT / "frontend" / "chat"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install_from_requirements("requirements-modal.txt")
    .env(
        {
            "HF_HOME": str(REMOTE_CACHE_ROOT),
            "HF_HUB_CACHE": str(REMOTE_CACHE_ROOT / "hub"),
            "TRANSFORMERS_CACHE": str(REMOTE_CACHE_ROOT / "hub"),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("code_diffusion")
    .add_local_dir("frontend", remote_path="/workspace/frontend")
    .add_local_dir("code_diffusion", remote_path="/workspace/code_diffusion")
    .add_local_dir("benchmarks", remote_path="/workspace/benchmarks")
    .add_local_file("config.yaml", remote_path="/workspace/config.yaml")
    .add_local_file("config.public-data.yaml", remote_path="/workspace/config.public-data.yaml")
    .add_local_file("README.md", remote_path="/workspace/README.md")
)

app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

function_kwargs: dict[str, Any] = {
    "image": image,
    "gpu": GPU_TYPE,
    "cpu": CPU_COUNT,
    "memory": MEMORY_MB,
    "timeout": TIMEOUT_SECONDS,
    "scaledown_window": 15 * 60,
    "volumes": {
        str(REMOTE_OUTPUT_ROOT): output_volume.read_only(),
        str(REMOTE_CACHE_ROOT): cache_volume,
    },
}
if HF_SECRET_NAME:
    function_kwargs["secrets"] = [modal.Secret.from_name(HF_SECRET_NAME)]


@app.function(**function_kwargs)
@modal.asgi_app(label=CHAT_LABEL)
def serve():
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    globals()["FastAPI"] = FastAPI
    globals()["Request"] = Request
    globals()["FileResponse"] = FileResponse
    globals()["StaticFiles"] = StaticFiles

    runtime_holder: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        checkpoint_path = REMOTE_OUTPUT_ROOT / CHAT_RUN_NAME / CHAT_CHECKPOINT_SUBDIR
        runtime_holder["runtime"] = DiffusionChatRuntime(
            config_path=CHAT_CONFIG_PATH,
            checkpoint=str(checkpoint_path),
            overrides=[],
            repo_root=WORKSPACE_ROOT,
        )
        yield
        runtime_holder.clear()

    web_app = FastAPI(lifespan=lifespan)

    @web_app.get("/api/state")
    def get_state():
        runtime = runtime_holder["runtime"]
        return {
            "ok": True,
            "model_name": str(runtime.config["model_name"]),
            "config_path": str(runtime.config["config_path"]),
            "repo_root": str(runtime.repo_root),
            "device": str(runtime.device),
            "runtime_dtype": runtime.runtime_dtype,
            "default_steps": int(runtime.config["diffusion_steps"]),
            "default_mask_span": int(runtime.config["inference_mask_span"]),
            "startup_warnings": list(runtime.startup_warnings),
        }

    @web_app.post("/api/chat")
    async def chat(request: Request):
        runtime = runtime_holder["runtime"]
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                payload = {}
            result = runtime.chat(
                message=str(payload.get("message", "")),
                history=list(payload.get("history") or []),
                code_context=str(payload.get("code_context", "")),
                draft_template=str(payload.get("draft_template", "")),
                file_paths_text=str(payload.get("file_paths_text", "")),
                steps=_coerce_int(payload.get("steps")),
                mask_span_tokens=_coerce_int(payload.get("mask_span_tokens")),
                temperature=_coerce_float(payload.get("temperature")),
                top_k=_coerce_int(payload.get("top_k")),
                top_p=_coerce_float(payload.get("top_p")),
                confidence_threshold=_coerce_float(payload.get("confidence_threshold")),
            )
            return {"ok": True, **result}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @web_app.get("/favicon.ico")
    def favicon():
        return FileResponse(FRONTEND_ROOT / "app.js", media_type="text/plain")

    @web_app.get("/")
    def index():
        return FileResponse(FRONTEND_ROOT / "index.html")

    @web_app.get("/styles.css")
    def styles():
        return FileResponse(FRONTEND_ROOT / "styles.css")

    @web_app.get("/app.js")
    def app_js():
        return FileResponse(FRONTEND_ROOT / "app.js")

    web_app.mount("/assets", StaticFiles(directory=str(FRONTEND_ROOT)), name="assets")
    return web_app


def _coerce_int(value) -> int | None:
    if value in (None, "", "null"):
        return None
    return int(value)


def _coerce_float(value) -> float | None:
    if value in (None, "", "null"):
        return None
    return float(value)
