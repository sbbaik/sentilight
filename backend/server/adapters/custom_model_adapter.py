from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Callable

from server.adapters.base import COMPACTLM_JSON_PROMPT, AdapterPrediction, ModelAdapter


class CustomPytorchAdapter(ModelAdapter):
    def __init__(self, config):
        super().__init__(config)
        self._predict_fn: Callable[[str, object], object] | None = None
        self._model: object | None = None

    def load(self) -> None:
        model_dir = self.config.resolved_model_dir
        if model_dir is None:
            raise RuntimeError("custom_pytorch model_dir is required")
        checkpoint = self.config.checkpoint
        if checkpoint and not (Path(model_dir) / checkpoint).exists():
            raise RuntimeError(f"Checkpoint not found: {Path(model_dir) / checkpoint}")

        inference_py = Path(model_dir) / "inference.py"
        if not inference_py.exists():
            raise RuntimeError(
                "custom_pytorch adapter expects inference.py in model_dir with "
                "load_model(config) and predict(text, model) functions"
            )

        module_name = f"sentilight_{self.config.id}_inference"
        spec = importlib.util.spec_from_file_location(module_name, inference_py)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import inference module: {inference_py}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "load_model") or not hasattr(module, "predict"):
            raise RuntimeError("inference.py must define load_model(config) and predict(text, model)")
        self._model = module.load_model(self.config)
        self._predict_fn = module.predict

    def predict(self, text: str) -> AdapterPrediction:
        if self._predict_fn is None:
            self.load()
        input_mode = str((self.config.options or {}).get("input_mode", "")).strip().lower()
        if input_mode == "raw_text":
            model_input = text
        else:
            model_input = COMPACTLM_JSON_PROMPT.format(text=text)
        raw = self._predict_fn(model_input, self._model)
        return self.parse(raw)
