from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from common.config_loader import load_config
from common.model_registry import ModelRegistry


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)


def create_app() -> FastAPI:
    config_path = os.getenv("SENTILIGHT_MODELS_CONFIG")
    app_config = load_config(config_path)
    eager_load = os.getenv("SENTILIGHT_EAGER_LOAD", "").lower() in {"1", "true", "yes"}
    registry = ModelRegistry(app_config, eager_load=eager_load)

    app = FastAPI(title="Sentilight Multi-Model Service", version="0.1.0")
    app.state.app_config = app_config
    app.state.registry = registry

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "config": str(app_config.source_path),
            "backend_root": str(Path(__file__).resolve().parents[1]),
            "used_example_config": app_config.used_example,
            "enabled_model_count": len(registry.models),
            "eager_load": eager_load,
        }

    @app.get("/models")
    def models() -> dict[str, Any]:
        return {"models": registry.list_models()}

    @app.get("/preflight")
    def preflight(expected_model_count: int = 0, load_models: bool = False) -> dict[str, Any]:
        return registry.preflight(expected_model_count=expected_model_count, load_models=load_models)

    @app.post("/predict/{model_id}")
    def predict(model_id: str, request: PredictRequest) -> dict[str, Any]:
        result = registry.predict(model_id, request.text)
        if result["error"] and result["error"].startswith("Unknown or disabled model"):
            raise HTTPException(status_code=404, detail=result)
        return result

    @app.post("/predict_all")
    def predict_all(request: PredictRequest) -> dict[str, Any]:
        return {"text": request.text, "results": registry.predict_all(request.text)}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    runtime = app.state.app_config.runtime
    uvicorn.run("server.main:app", host=runtime.host, port=runtime.port, reload=False)
