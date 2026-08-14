from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


SYSTEM_PROMPT = (
    "당신은 감정 기반 스마트 조명 제어 모델입니다. "
    "반드시 JSON만 출력하세요. "
    "규칙: 명시적 색상 요청이 있으면 hue는 그 색상을 우선합니다. "
    "감정은 saturation, brightness, dimmer, color temperature에 반영하세요. "
    "애매한 감정이면 안전하고 무난한 조명값을 사용하세요. "
    "출력 키는 H,S,B,Dimmer,CT만 사용하세요."
)
USER_PROMPT_TEMPLATE = "문장: {text}"
REQUIRED_FIELDS = ("H", "S", "B", "Dimmer", "CT")
FIELD_RANGES = {
    "H": (0, 360),
    "S": (0, 100),
    "B": (0, 100),
    "Dimmer": (0, 100),
    "CT": (153, 500),  # matches the CT range declared in the model prompt
}


@dataclass
class RuntimeModel:
    torch: Any
    model: Any
    tokenizer: Any
    device: Any
    max_new_tokens: int
    eos_id: Optional[int]
    checkpoint_path: Path


def _load_definition(model_dir: Path):
    definition_path = model_dir / "model_definition.py"
    spec = importlib.util.spec_from_file_location("sentilight_compactlm_definition", definition_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model definition: {definition_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise RuntimeError("CompactLM checkpoint must be a dictionary")
    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    return {
        key.removeprefix("module.").removeprefix("_orig_mod."): value
        for key, value in state.items()
        if hasattr(value, "shape")
    }


def _find_tokenizer(model_dir: Path, tokenizer_dir: str | None) -> Path:
    base = model_dir / tokenizer_dir if tokenizer_dir else model_dir / "tokenizer"
    candidates = [base / "tokenizer.json", model_dir / "tokenizer.json"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"tokenizer.json not found under {base}")


def _resolve_eos_id(tokenizer: Any) -> Optional[int]:
    for token in ("<eos>", "</s>"):
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            return int(token_id)
    return None


def _normalize_json_object(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError("CompactLM JSON output must be an object")

    normalized: dict[str, int] = {}
    for field in REQUIRED_FIELDS:
        if field not in value:
            raise ValueError(f"Missing required field: {field}")
        raw_value = value[field]
        if isinstance(raw_value, bool):
            raise ValueError(f"Field {field} must be numeric, got bool")
        if not isinstance(raw_value, (int, float)):
            raise ValueError(f"Field {field} must be numeric")
        int_value = int(round(float(raw_value)))
        lower, upper = FIELD_RANGES[field]
        if not (lower <= int_value <= upper):
            raise ValueError(f"Field {field} out of range: {int_value}")
        normalized[field] = int_value
    return normalized


def _try_parse_completed_json(text: str) -> Optional[dict[str, int]]:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    return _normalize_json_object(json.loads(candidate))
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


def _extract_json(text: str) -> dict[str, int]:
    parsed = _try_parse_completed_json(text)
    if parsed is None:
        raise ValueError(f"CompactLM did not generate a valid control JSON: {text[:300]}")
    return parsed


def build_prompt(text: str) -> str:
    stripped = text.strip()
    if "<|system|>" in stripped or "<|user|>" in stripped:
        return stripped
    return (
        "<|system|>\n"
        f"{SYSTEM_PROMPT}\n"
        "<|user|>\n"
        f"{USER_PROMPT_TEMPLATE.format(text=stripped)}\n"
        "<|assistant|>\n"
    )


def load_model(config) -> RuntimeModel:
    try:
        import torch
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("CompactLM requires torch and tokenizers") from exc

    model_dir = Path(config.resolved_model_dir)
    checkpoint_path = model_dir / str(config.checkpoint)
    tokenizer_path = _find_tokenizer(model_dir, config.tokenizer_dir)
    options = config.options or {}
    device = torch.device(str(options.get("device", "cuda" if torch.cuda.is_available() else "cpu")))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    definition = _load_definition(model_dir)
    config_data = checkpoint.get("model_config") or checkpoint.get("config") or checkpoint.get("cfg") or {}
    model_config = definition.ModelConfig(**config_data)
    model = definition.SentilightCompactLM(model_config)
    model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
    model.to(device).eval()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return RuntimeModel(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=int(options.get("max_new_tokens", 96)),
        eos_id=_resolve_eos_id(tokenizer),
        checkpoint_path=checkpoint_path,
    )


def predict(text: str, runtime: RuntimeModel) -> dict[str, int]:
    prompt_text = build_prompt(text)
    prompt_ids = runtime.tokenizer.encode(prompt_text).ids[-runtime.model.cfg.block_size :]
    input_ids = runtime.torch.tensor([prompt_ids], dtype=runtime.torch.long, device=runtime.device)

    with runtime.torch.inference_mode():
        output = runtime.model.generate(
            input_ids,
            max_new_tokens=runtime.max_new_tokens,
            temperature=0.0,
            eos_id=runtime.eos_id,
        )

    generated_ids = output[0, len(prompt_ids) :].tolist()
    decoded = runtime.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return _extract_json(decoded)


def _parse_simple_yaml(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def build_config(model_dir: Path, checkpoint: Optional[str], device: str, max_new_tokens: int):
    config_data = _parse_simple_yaml(model_dir / "model_config.yaml")
    return type(
        "Config",
        (),
        {
            "resolved_model_dir": str(model_dir),
            "checkpoint": checkpoint or config_data["checkpoint"],
            "tokenizer_dir": config_data.get("tokenizer_dir", "tokenizer"),
            "options": {
                "device": device,
                "max_new_tokens": max_new_tokens,
            },
        },
    )()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CompactLM FP32 inference for a single prompt.")
    parser.add_argument("--text", required=True, help="Input text prompt")
    parser.add_argument("--model-dir", default=Path(__file__).resolve().parent, type=Path)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", default=96, type=int)
    args = parser.parse_args()

    runtime = load_model(
        build_config(
            model_dir=args.model_dir.resolve(),
            checkpoint=args.checkpoint,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
    )
    print(json.dumps(predict(args.text, runtime), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
