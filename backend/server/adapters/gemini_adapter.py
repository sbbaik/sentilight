from __future__ import annotations

import os

import requests

from server.adapters.base import GEMINI_COMMAND_PROMPT, AdapterPrediction, ModelAdapter


class GeminiAdapter(ModelAdapter):
    def _api_key(self) -> str:
        # Try to get API key from options first
        api_key = (self.config.options or {}).get("api_key")
        # Fall back to environment variable if not in options
        if not api_key:
            api_key = os.getenv(self.config.env_key or "GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(f"Missing Gemini API key. Set it in options.api_key or environment variable {self.config.env_key or 'GEMINI_API_KEY'}")
        return api_key

    def _model_name(self) -> str:
        model_name = (self.config.options or {}).get("model_name", "gemini-3.1-flash-lite")
        if not str(model_name).startswith("models/"):
            model_name = f"models/{model_name}"
        return str(model_name)

    def health_check(self) -> None:
        api_key = self._api_key()
        model_name = self._model_name()
        timeout_seconds = float((self.config.options or {}).get("health_timeout_seconds", 10))
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}"
        response = requests.get(url, params={"key": api_key}, timeout=timeout_seconds)
        if not response.ok:
            raise RuntimeError(f"Gemini readiness check HTTP {response.status_code}: {response.text[:300]}")

        data = response.json()
        methods = data.get("supportedGenerationMethods") or []
        if "generateContent" not in methods:
            raise RuntimeError(f"Gemini model {model_name} does not support generateContent")

    def predict(self, text: str) -> AdapterPrediction:
        api_key = self._api_key()
        model_name = self._model_name()
        url = f"https://generativelanguage.googleapis.com/v1/{model_name}:generateContent"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": GEMINI_COMMAND_PROMPT.format(text=text)}],
                }
            ]
        }
        timeout_seconds = float((self.config.options or {}).get("timeout_seconds", 30))
        response = requests.post(url, params={"key": api_key}, json=payload, timeout=timeout_seconds)
        if not response.ok:
            raise RuntimeError(f"Gemini API error HTTP {response.status_code}: {response.text[:300]}")

        data = response.json()
        try:
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Gemini response shape: {data}") from exc
        return self.parse(raw_text)
