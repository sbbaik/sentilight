from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "models.yaml"
EXAMPLE_CONFIG_PATH = PROJECT_ROOT / "config" / "models.example.yaml"


@dataclass(frozen=True)
class RuntimeSettings:
    host: str = "127.0.0.1"
    port: int = 8100
    request_timeout_seconds: float = 20.0


@dataclass(frozen=True)
class ModelConfig:
    id: str
    name: str
    enabled: bool
    mode: str
    adapter: str
    model_dir: str | None = None
    checkpoint: str | None = None
    tokenizer_dir: str | None = None
    endpoint: str | None = None
    env_key: str | None = None
    options: dict[str, Any] | None = None

    @property
    def resolved_model_dir(self) -> Path | None:
        if not self.model_dir:
            return None
        path = Path(self.model_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class AppConfig:
    models: list[ModelConfig]
    runtime: RuntimeSettings
    source_path: Path
    used_example: bool = False

    @property
    def enabled_models(self) -> list[ModelConfig]:
        return [model for model in self.models if model.enabled]


def _validate_models(models: list[ModelConfig]) -> None:
    ids: set[str] = set()
    for model in models:
        if model.id in ids:
            raise ValueError(f"Duplicate model id: {model.id}")
        ids.add(model.id)


def load_config(path: str | Path | None = None, allow_example_fallback: bool = True) -> AppConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    used_example = False

    if not config_path.exists():
        if not allow_example_fallback:
            raise FileNotFoundError(
                f"{config_path} does not exist. Copy backend/config/models.example.yaml "
                "to backend/config/models.yaml and edit model settings."
            )
        print(
            f"[config] {config_path} does not exist. Using {EXAMPLE_CONFIG_PATH} for demo. "
            "Copy it to backend/config/models.yaml for real use."
        )
        config_path = EXAMPLE_CONFIG_PATH
        used_example = True

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    runtime_raw = raw.get("runtime", {}) or {}
    runtime = RuntimeSettings(
        host=str(runtime_raw.get("host", "127.0.0.1")),
        port=int(runtime_raw.get("port", 8100)),
        request_timeout_seconds=float(runtime_raw.get("request_timeout_seconds", 20.0)),
    )

    models = []
    for item in raw.get("models", []) or []:
        models.append(
            ModelConfig(
                id=str(item["id"]),
                name=str(item.get("name") or item["id"]),
                enabled=bool(item.get("enabled", False)),
                mode=str(item.get("mode", "local_adapter")),
                adapter=str(item.get("adapter", "mock")),
                model_dir=item.get("model_dir"),
                checkpoint=item.get("checkpoint"),
                tokenizer_dir=item.get("tokenizer_dir"),
                endpoint=item.get("endpoint"),
                env_key=item.get("env_key"),
                options=item.get("options") or {},
            )
        )

    _validate_models(models)
    return AppConfig(models=models, runtime=runtime, source_path=config_path, used_example=used_example)
