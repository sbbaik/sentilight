from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from compact_llm.model_definition import ModelConfig, RMSNorm, TransformerBlock
from compact_llm.training_data import (
    COLOR_PROFILES,
    EMOTION_PROFILES,
    INTENSITY_MODIFIERS,
    resolve_case_output,
)


CONTROL_FIELDS = ("H", "S", "B", "Dimmer", "CT")


def output_tuple(output: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return tuple(int(output[field]) for field in CONTROL_FIELDS)


def build_tuple_vocab() -> list[dict[str, int]]:
    outputs: list[dict[str, int]] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    colors: list[str | None] = [None] + list(COLOR_PROFILES)
    emotions: list[str | None] = [None] + list(EMOTION_PROFILES)
    intensities = list(INTENSITY_MODIFIERS)
    for color in colors:
        for emotion in emotions:
            for intensity in intensities:
                if color is None and emotion is None:
                    case = {"intensity": intensity}
                else:
                    case = {"base_color": color, "emotion": emotion, "intensity": intensity}
                output = resolve_case_output(case)
                key = output_tuple(output)
                if key not in seen:
                    seen.add(key)
                    outputs.append(output)
    return outputs


def tuple_to_label_map(vocab: list[dict[str, int]]) -> dict[tuple[int, int, int, int, int], int]:
    return {output_tuple(output): index for index, output in enumerate(vocab)}


class SentilightTupleClassifier(nn.Module):
    """CompactLM backbone with a finite-control tuple classification head."""

    def __init__(self, cfg: ModelConfig, tuple_vocab: list[dict[str, int]] | None = None):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tuple_vocab = tuple_vocab or build_tuple_vocab()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.tuple_head = nn.Linear(cfg.d_model, len(self.tuple_vocab), bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def load_lm_backbone_state(self, state_dict: dict[str, Any]) -> None:
        own_state = self.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in own_state and tuple(own_state[key].shape) == tuple(value.shape)
        }
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        unexpected = [key for key in unexpected if key != "tuple_head.weight"]
        if unexpected:
            raise RuntimeError(f"Unexpected classifier state keys: {unexpected}")
        required_prefixes = ("tok_emb.", "blocks.", "norm_f.")
        missing_backbone = [key for key in missing if key.startswith(required_prefixes)]
        if missing_backbone:
            raise RuntimeError(f"Missing LM backbone keys: {missing_backbone[:8]}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, seq_len].")
        if input_ids.size(1) > self.cfg.block_size:
            raise ValueError(f"sequence length {input_ids.size(1)} exceeds block_size {self.cfg.block_size}")
        x = self.tok_emb(input_ids)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        if attention_mask is None:
            pooled = x[:, -1, :]
        else:
            lengths = attention_mask.to(torch.long).sum(dim=1).clamp(min=1)
            batch_index = torch.arange(x.size(0), device=x.device)
            pooled = x[batch_index, lengths - 1, :]
        logits = self.tuple_head(pooled)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return logits, loss

    def checkpoint_payload(self, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "model_state_dict": self.state_dict(),
            "model_config": asdict(self.cfg),
            "tuple_vocab": self.tuple_vocab,
            "tuple_count": len(self.tuple_vocab),
            "metrics": metrics or {},
        }


def extract_lm_state_dict(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint must be a dictionary.")
    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    return {
        key.removeprefix("module.").removeprefix("_orig_mod."): value
        for key, value in state.items()
        if hasattr(value, "shape")
    }
