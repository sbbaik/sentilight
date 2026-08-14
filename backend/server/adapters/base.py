from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

from common.config_loader import ModelConfig
from common.lighting_schema import LightingValues, parse_lighting_response


GEMINI_COMMAND_PROMPT = (
    "사용자 기분: '{text}'. 이를 Tasmota 전구 제어 명령으로 변환하세요. "
    "결과 형식은 [COMMAND: HSBCOLOR hue,saturation,brightness;Dimmer value;CT temperature] "
    "이 세 가지 명령 조합으로만 출력하세요. "
    "[EXPLANATION: 기분 변화에 대한 설명] 으로만 출력하세요. "
    "(hue:0-359, saturation/brightness:0-100, Dimmer:0-100, CT:153-500). "
    "예: [COMMAND: HSBCOLOR 60,100,100;Dimmer 70;CT 250] [EXPLANATION: 밝고 따뜻한 노란색으로 활력을 줍니다.]"
)

COMPACTLM_JSON_PROMPT = (
    "<|system|>\n당신은 조명 제어를 위한 어시스턴트입니다. 반드시 JSON만 출력하세요.\n"
    "<|user|>\n다음 문장의 감정에 어울리는 H,S,B,Dimmer,CT 값을 예측하세요. JSON으로만 답하세요.\n\n"
    "문장: {text}\n"
    "<|assistant|>\n"
)


@dataclass(frozen=True)
class AdapterPrediction:
    lighting: LightingValues
    raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"lighting": self.lighting.to_dict(), "raw": self.raw}


class ModelAdapter(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def load(self) -> None:
        """Load heavyweight model resources if needed."""

    def health_check(self) -> None:
        """Validate external dependencies when a deep preflight is requested."""

    @abstractmethod
    def predict(self, text: str) -> AdapterPrediction:
        raise NotImplementedError

    def parse(self, raw: Any) -> AdapterPrediction:
        return AdapterPrediction(lighting=parse_lighting_response(raw), raw=raw)

    def metadata(self) -> dict[str, Any]:
        data = asdict(self.config)
        data.pop("env_key", None)
        return data
