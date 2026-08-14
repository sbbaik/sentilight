from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_inference_module():
    base_dir = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "compactlm_fp32_inference",
        base_dir / "inference.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to import inference.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_runtime(
    model_dir: str | Path | None = None,
    checkpoint: str | None = None,
    *,
    device: str = "cpu",
    max_new_tokens: int = 96,
):
    resolved_dir = Path(model_dir or Path(__file__).resolve().parent).resolve()
    inference_module = _load_inference_module()
    config = inference_module.build_config(
        model_dir=resolved_dir,
        checkpoint=checkpoint,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    return inference_module.load_model(config)


if __name__ == "__main__":
    runtime = load_runtime()
    print(runtime.checkpoint_path)
