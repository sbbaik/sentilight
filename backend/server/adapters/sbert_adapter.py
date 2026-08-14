from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common.lighting_schema import normalize_lighting
from server.adapters.base import AdapterPrediction, ModelAdapter


DEFAULT_PROFILES = [
    {
        "label": "joy",
        "examples": ["행복하고 기쁘다", "신나고 즐겁다", "기분이 매우 좋다"],
        "lighting": {"H": 52, "S": 82, "B": 92, "Dimmer": 86, "CT": 250},
    },
    {
        "label": "sadness",
        "examples": ["우울하고 슬프다", "마음이 힘들고 외롭다", "눈물이 날 것 같다"],
        "lighting": {"H": 215, "S": 55, "B": 38, "Dimmer": 42, "CT": 420},
    },
    {
        "label": "calm",
        "examples": ["마음이 편안하고 평온하다", "차분하게 쉬고 싶다", "긴장이 풀렸다"],
        "lighting": {"H": 145, "S": 38, "B": 62, "Dimmer": 58, "CT": 320},
    },
    {
        "label": "anger",
        "examples": ["화가 나고 짜증난다", "분노를 참기 어렵다", "너무 답답하다"],
        "lighting": {"H": 5, "S": 90, "B": 72, "Dimmer": 68, "CT": 280},
    },
]


class SbertAdapter(ModelAdapter):
    def __init__(self, config) -> None:
        super().__init__(config)
        self._model: Any | None = None
        self._profiles: list[dict[str, Any]] = []
        self._profile_embeddings: Any | None = None
        self._example_profile_indexes: list[int] = []

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sbert adapter requires sentence-transformers; install backend/requirements-sbert.txt"
            ) from exc

        options = self.config.options or {}
        model_dir = self.config.resolved_model_dir
        default_finetuned = model_dir / "finetuned_model" if model_dir else None
        model_name = options.get("model_name")
        if not model_name and default_finetuned and default_finetuned.exists():
            model_name = str(default_finetuned)
        model_name = str(model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        profiles = options.get("profiles")
        profiles_path = options.get("profiles_path")
        if profiles is None:
            candidate = Path(str(profiles_path)) if profiles_path else (model_dir / "profiles.json" if model_dir else None)
            if candidate and not candidate.is_absolute() and self.config.resolved_model_dir:
                candidate = self.config.resolved_model_dir / candidate
            if candidate and candidate.exists():
                loaded = json.loads(candidate.read_text(encoding="utf-8"))
                profiles = loaded.get("profiles", loaded)
        profiles = profiles or DEFAULT_PROFILES
        if not isinstance(profiles, list) or not profiles:
            raise ValueError("sbert options.profiles must be a non-empty list")

        self._profiles = profiles
        self._model = SentenceTransformer(
            model_name,
            device=options.get("device"),
            local_files_only=bool(options.get("local_files_only", False)),
            tokenizer_kwargs={"fix_mistral_regex": True},
        )
        prototype_texts: list[str] = []
        self._example_profile_indexes = []
        for profile_index, profile in enumerate(profiles):
            examples = profile.get("examples") or []
            if not examples:
                raise ValueError(f"sbert profile {profile.get('label', profile_index)} has no examples")
            for example in examples:
                prototype_texts.append(str(example))
                self._example_profile_indexes.append(profile_index)
        self._profile_embeddings = self._model.encode(prototype_texts, normalize_embeddings=True)

    def predict(self, text: str) -> AdapterPrediction:
        self.load()
        from sentence_transformers import util

        query_embedding = self._model.encode([text], normalize_embeddings=True)
        scores = util.cos_sim(query_embedding, self._profile_embeddings)[0]
        best_example_index = int(scores.argmax().item())
        best_score = float(scores[best_example_index].item())
        profile = self._profiles[self._example_profile_indexes[best_example_index]]
        raw = {"label": profile["label"], "score": round(best_score, 6)}
        return AdapterPrediction(lighting=normalize_lighting(profile["lighting"]), raw=raw)
