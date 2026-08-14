from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "sentilight_qwen3_0_6b_sft_q4km.gguf"
SYSTEM_PROMPT = (
    "당신은 감정 기반 스마트 조명 제어 모델입니다. 반드시 JSON만 출력하세요. "
    "출력 키는 H,S,B,Dimmer,CT만 사용하세요."
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"Qwen did not return a JSON object: {text[:300]}")


def build_prompt(text: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        "다음 문장의 감정에 어울리는 H,S,B,Dimmer,CT 값을 예측하세요. JSON으로만 답하세요.\n\n"
        f"문장: {text}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


class QwenRuntime:
    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path or os.getenv("SENTILIGHT_QWEN_MODEL", DEFAULT_MODEL_PATH)).expanduser()
        self._model: Any | None = None
        self._lock = Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"Qwen GGUF model not found: {self.model_path}")
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("Install backend/qwen_service/requirements.txt to use the Qwen GGUF service") from exc

        self._model = Llama(
            model_path=str(self.model_path),
            n_ctx=int(os.getenv("SENTILIGHT_QWEN_CONTEXT", "1024")),
            n_threads=int(os.getenv("SENTILIGHT_QWEN_THREADS", str(os.cpu_count() or 4))),
            n_gpu_layers=int(os.getenv("SENTILIGHT_QWEN_GPU_LAYERS", "0")),
            verbose=False,
        )

    def predict(self, text: str) -> tuple[dict[str, Any], str, float]:
        self.load()
        started = perf_counter()
        with self._lock:
            response = self._model.create_completion(
                prompt=build_prompt(text),
                temperature=0,
                max_tokens=int(os.getenv("SENTILIGHT_QWEN_MAX_TOKENS", "96")),
                stop=["<|im_end|>"],
            )
        raw = str(response["choices"][0]["text"])
        return extract_json_object(raw), raw, round((perf_counter() - started) * 1000, 2)


def create_app(runtime: QwenRuntime | None = None) -> FastAPI:
    resolved_runtime = runtime or QwenRuntime()
    app = FastAPI(title="Sentilight Qwen GGUF Service", version="0.1.0")
    app.state.runtime = resolved_runtime

    @app.get("/health")
    def health(load_model: bool = False) -> dict[str, Any]:
        error = None
        if load_model:
            try:
                resolved_runtime.load()
            except Exception as exc:
                error = str(exc)
        return {
            "ok": error is None and resolved_runtime.model_path.exists(),
            "model_path": str(resolved_runtime.model_path),
            "model_exists": resolved_runtime.model_path.exists(),
            "loaded": resolved_runtime.loaded,
            "error": error,
        }

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict[str, Any]:
        try:
            lighting, raw, latency_ms = resolved_runtime.predict(request.text)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"lighting": lighting, "raw": raw, "latency_ms": latency_ms}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "qwen_service.main:app",
        host=os.getenv("SENTILIGHT_QWEN_HOST", "127.0.0.1"),
        port=int(os.getenv("SENTILIGHT_QWEN_PORT", "8103")),
        reload=False,
    )
