from __future__ import annotations

import requests

from server.adapters.base import AdapterPrediction, ModelAdapter


class RemoteAdapter(ModelAdapter):
    def load(self) -> None:
        if not self.config.endpoint:
            raise RuntimeError("remote_endpoint mode requires endpoint")

    def predict(self, text: str) -> AdapterPrediction:
        self.load()
        timeout_seconds = float((self.config.options or {}).get("timeout_seconds", 30))
        response = requests.post(self.config.endpoint, json={"text": text}, timeout=timeout_seconds)
        if not response.ok:
            raise RuntimeError(f"Remote model error HTTP {response.status_code}: {response.text[:300]}")
        data = response.json()
        if isinstance(data, dict) and "lighting" in data:
            return self.parse(data["lighting"])
        return self.parse(data)
