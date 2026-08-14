from __future__ import annotations

from server.adapters.base import AdapterPrediction, ModelAdapter
from common.lighting_schema import LightingValues


class MockAdapter(ModelAdapter):
    def predict(self, text: str) -> AdapterPrediction:
        normalized = text.strip()
        if any(word in normalized for word in ("행복", "기뻐", "신나", "좋아")):
            lighting = LightingValues(H=52, S=82, B=92, Dimmer=86, CT=250)
        elif any(word in normalized for word in ("우울", "슬퍼", "불안", "힘들")):
            lighting = LightingValues(H=215, S=55, B=38, Dimmer=42, CT=420)
        elif any(word in normalized for word in ("편안", "차분", "평온", "쉬고")):
            lighting = LightingValues(H=145, S=38, B=62, Dimmer=58, CT=320)
        else:
            lighting = LightingValues(H=35, S=45, B=70, Dimmer=65, CT=300)
        raw = f"[COMMAND: {lighting.to_tasmota_command()}] [EXPLANATION: mock_model heuristic]"
        return AdapterPrediction(lighting=lighting, raw=raw)

