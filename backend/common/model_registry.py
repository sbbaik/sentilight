from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import importlib.util
import os
import requests
from threading import Lock
from time import perf_counter
from typing import Any

from common.config_loader import AppConfig, ModelConfig


@dataclass
class RegisteredModel:
    config: ModelConfig
    adapter: Any | None = None
    load_error: str | None = None
    lock: Lock = field(default_factory=Lock)


class ModelRegistry:
    def __init__(self, app_config: AppConfig, eager_load: bool = False) -> None:
        self.app_config = app_config
        self.eager_load = eager_load
        self.models: dict[str, RegisteredModel] = {}
        self._register_enabled_models()

    def _adapter_class(self, adapter_name: str):
        if adapter_name == "mock":
            from server.adapters.mock_adapter import MockAdapter

            return MockAdapter
        if adapter_name == "gemini":
            from server.adapters.gemini_adapter import GeminiAdapter

            return GeminiAdapter
        if adapter_name == "custom_pytorch":
            from server.adapters.custom_model_adapter import CustomPytorchAdapter

            return CustomPytorchAdapter
        if adapter_name == "remote":
            from server.adapters.remote_adapter import RemoteAdapter

            return RemoteAdapter
        if adapter_name == "sbert":
            from server.adapters.sbert_adapter import SbertAdapter

            return SbertAdapter
        raise ValueError(f"Unknown adapter: {adapter_name}")

    def _validate_config(self, config: ModelConfig) -> None:
        if config.mode == "local_adapter":
            model_dir = config.resolved_model_dir
            if model_dir is None:
                raise ValueError(f"{config.id}: model_dir is required for local_adapter")
            if not model_dir.exists():
                raise FileNotFoundError(f"{config.id}: model_dir does not exist: {model_dir}")
        elif config.mode == "remote_endpoint":
            if not config.endpoint:
                raise ValueError(f"{config.id}: endpoint is required for remote_endpoint")
        else:
            raise ValueError(f"{config.id}: unsupported mode: {config.mode}")

    def _register_enabled_models(self) -> None:
        for config in self.app_config.enabled_models:
            registered = RegisteredModel(config=config)
            try:
                self._validate_config(config)
                self._populate_env_key_from_options(config)
                adapter_cls = self._adapter_class(config.adapter)
                registered.adapter = adapter_cls(config)
                if self.eager_load:
                    registered.adapter.load()
            except Exception as exc:
                registered.load_error = str(exc)
            self.models[config.id] = registered

    @staticmethod
    def _populate_env_key_from_options(config: ModelConfig) -> None:
        if not config.env_key:
            return
        api_key = str((config.options or {}).get("api_key", "")).strip()
        if api_key and not os.getenv(config.env_key):
            os.environ[config.env_key] = api_key

    def list_models(self) -> list[dict[str, Any]]:
        rows = []
        for model_id, registered in self.models.items():
            cfg = registered.config
            rows.append(
                {
                    "id": model_id,
                    "name": cfg.name,
                    "enabled": cfg.enabled,
                    "mode": cfg.mode,
                    "adapter": cfg.adapter,
                    "ready": registered.adapter is not None and registered.load_error is None,
                    "error": registered.load_error,
                }
            )
        return rows

    def preflight(self, expected_model_count: int = 0, load_models: bool = False) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        if expected_model_count:
            checks.append(
                {
                    "name": "model_count",
                    "ok": len(self.models) == expected_model_count,
                    "detail": f"configured={len(self.models)}, expected={expected_model_count}",
                }
            )

        for model_id, registered in self.models.items():
            cfg = registered.config
            errors: list[str] = []
            if registered.load_error:
                errors.append(registered.load_error)
            configured_api_key = bool((cfg.options or {}).get("api_key"))
            if cfg.env_key and not os.getenv(cfg.env_key) and not configured_api_key:
                errors.append(f"missing environment variable {cfg.env_key}")
            if cfg.adapter == "custom_pytorch" and cfg.resolved_model_dir:
                if cfg.checkpoint and not (cfg.resolved_model_dir / cfg.checkpoint).exists():
                    errors.append(f"checkpoint not found: {cfg.resolved_model_dir / cfg.checkpoint}")
                if not (cfg.resolved_model_dir / "inference.py").exists():
                    errors.append(f"inference.py not found in {cfg.resolved_model_dir}")
                tokenizer_dir = cfg.resolved_model_dir / (cfg.tokenizer_dir or "tokenizer")
                if not (tokenizer_dir / "tokenizer.json").exists():
                    errors.append(f"tokenizer.json not found in {tokenizer_dir}")
            if cfg.adapter == "sbert" and importlib.util.find_spec("sentence_transformers") is None:
                errors.append("sentence-transformers is not installed")
            if load_models and cfg.adapter == "remote":
                health_endpoint = (cfg.options or {}).get("health_endpoint")
                if health_endpoint:
                    try:
                        response = requests.get(
                            str(health_endpoint),
                            timeout=float((cfg.options or {}).get("health_timeout_seconds", 3)),
                        )
                        if not response.ok:
                            errors.append(f"remote health check returned HTTP {response.status_code}")
                    except requests.RequestException as exc:
                        errors.append(f"remote health check failed: {exc}")

            if load_models and not errors and registered.adapter is not None:
                try:
                    with registered.lock:
                        registered.adapter.load()
                        registered.adapter.health_check()
                except Exception as exc:
                    errors.append(f"readiness check failed: {exc}")

            checks.append({"name": model_id, "ok": not errors, "detail": "; ".join(errors) or "ready"})

        return {"ok": bool(checks) and all(check["ok"] for check in checks), "checks": checks}

    def predict(self, model_id: str, text: str) -> dict[str, Any]:
        registered = self.models.get(model_id)
        if registered is None:
            return {
                "model_id": model_id,
                "model_name": model_id,
                "success": False,
                "lighting": None,
                "raw": None,
                "latency_ms": None,
                "error": f"Unknown or disabled model: {model_id}",
            }
        cfg = registered.config
        if registered.load_error:
            return self._error_result(cfg, registered.load_error)
        try:
            # Keep a single model instance safe across concurrent API requests.
            # Different registered models still execute in parallel.
            with registered.lock:
                started_at = perf_counter()
                prediction = registered.adapter.predict(text)
                latency_ms = round((perf_counter() - started_at) * 1000, 2)
            return {
                "model_id": cfg.id,
                "model_name": cfg.name,
                "success": True,
                "lighting": prediction.lighting.to_dict(),
                "raw": prediction.raw,
                "latency_ms": latency_ms,
                "error": None,
            }
        except Exception as exc:
            return self._error_result(cfg, str(exc))

    def predict_all(self, text: str) -> list[dict[str, Any]]:
        model_ids = list(self.models)
        if not model_ids:
            return []
        with ThreadPoolExecutor(max_workers=len(model_ids), thread_name_prefix="sentilight-model") as executor:
            return list(executor.map(lambda model_id: self.predict(model_id, text), model_ids))

    @staticmethod
    def _error_result(config: ModelConfig, error: str) -> dict[str, Any]:
        return {
            "model_id": config.id,
            "model_name": config.name,
            "success": False,
            "lighting": None,
            "raw": None,
            "latency_ms": None,
            "error": error,
        }
