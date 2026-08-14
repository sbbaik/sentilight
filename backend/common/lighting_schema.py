from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping


COMMAND_PATTERN = re.compile(
    r"HSBCOLOR\s+(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*;\s*Dimmer\s+(-?\d+)\s*;\s*CT\s+(-?\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LightingValues:
    H: int
    S: int
    B: int
    Dimmer: int
    CT: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    def to_tasmota_command(self) -> str:
        return f"HSBCOLOR {self.H},{self.S},{self.B};Dimmer {self.Dimmer};CT {self.CT}"


def clamp_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = minimum
    return max(minimum, min(maximum, parsed))


def normalize_lighting(data: Mapping[str, Any]) -> LightingValues:
    lower = {str(key).lower(): value for key, value in data.items()}
    return LightingValues(
        H=int(round(float(lower.get("h", lower.get("hue", 0)) or 0))) % 360,  # hue is circular: modulo, never clamp
        S=clamp_int(lower.get("s", lower.get("saturation", 0)), 0, 100),
        B=clamp_int(lower.get("b", lower.get("brightness", 0)), 0, 100),
        Dimmer=clamp_int(lower.get("dimmer", lower.get("d", 0)), 0, 100),
        CT=clamp_int(lower.get("ct", lower.get("temperature", 153)), 153, 500),
    )


def parse_tasmota_command(text: str) -> LightingValues | None:
    match = COMMAND_PATTERN.search(text or "")
    if not match:
        return None
    h, s, b, dimmer, ct = match.groups()
    return normalize_lighting({"H": h, "S": s, "B": b, "Dimmer": dimmer, "CT": ct})


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def parse_lighting_response(raw: Any) -> LightingValues:
    if isinstance(raw, LightingValues):
        return raw
    if isinstance(raw, Mapping):
        return normalize_lighting(raw)
    if isinstance(raw, str):
        json_obj = extract_json_object(raw)
        if json_obj is not None:
            return normalize_lighting(json_obj)
        command_values = parse_tasmota_command(raw)
        if command_values is not None:
            return command_values
    raise ValueError("Could not parse lighting values from model response")
