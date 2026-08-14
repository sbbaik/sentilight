# 51_sentilight_compact_lm.py
"""
Sentilight-CompactLM-v0.1 model definition

위치:
- 이 파일은 50_model/ 아래에 둔다.

역할:
- Sentilight compact LLM의 모델 구조만 정의한다.
- tokenizer 학습, token cache 생성, pretraining 실행은 수행하지 않는다.

모델 방향:
- Decoder-only causal Transformer
- Next-token prediction용 구조
- Qwen / LLaMA / DeepSeek 계열의 핵심 구성 요소를 축소 적용
  - RMSNorm
  - RoPE positional embedding
  - Causal self-attention
  - SwiGLU FFN
  - Tied input/output embedding

Sentilight 목적:
- 일반 한국어 문장 구조 학습
- 감정 표현, 일상 발화 이해 기반 형성
- 이후 SFT 단계에서 감정 문장 -> Tasmota JSON 출력 학습

기본 모델 크기:
- d_model = 384
- n_layer = 8
- n_head = 6
- d_ff = 1024
- block_size = 256

주의:
- 이 모델은 encoder-decoder 구조가 아니라 decoder-only LM이다.
- pretraining 단계에서는 일반 txt 기반 next-token prediction만 수행한다.
- Tasmota 제어 JSON은 이후 SFT 단계에서 학습한다.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    model_name: str = "Sentilight-CompactLM-v0.1"

    # tokenizer에서 실제 vocab size를 읽어와 주입한다.
    vocab_size: int = 24000

    # context length
    block_size: int = 256

    # compact transformer size
    n_layer: int = 8
    n_head: int = 6
    d_model: int = 384
    d_ff: int = 1024

    # regularization
    dropout: float = 0.10

    # RoPE
    rope_base: float = 10000.0

    # initialization
    init_std: float = 0.02

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

        if self.block_size <= 0:
            raise ValueError("block_size must be positive.")

        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive.")

        if self.n_head <= 0:
            raise ValueError("n_head must be positive.")

        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")

        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive.")

        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head.")

        head_dim = self.d_model // self.n_head

        if head_dim % 2 != 0:
            raise ValueError("RoPE requires even head_dim.")

        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0.0, 1.0).")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    LLaMA/Qwen 계열에서 흔히 사용하는 normalization 방식.
    LayerNorm보다 단순하고 compact 모델에 적합하다.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x_norm


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    base: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RoPE cache 생성.

    반환:
    - cos: [1, 1, T, D/2]
    - sin: [1, 1, T, D/2]
    """
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires even head_dim.")

    inv_freq = 1.0 / (
        base ** (
            torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
            / head_dim
        )
    )

    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)

    cos = freqs.cos().to(dtype=dtype)[None, None, :, :]
    sin = freqs.sin().to(dtype=dtype)[None, None, :, :]

    return cos, sin


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    RoPE 적용.

    입력:
    - x: [B, H, T, D]

    출력:
    - rotated x: [B, H, T, D]
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd

    return out


class CausalSelfAttention(nn.Module):
    """
    Decoder-only causal self-attention.

    현재 구현:
    - standard multi-head attention
    - RoPE 적용
    - PyTorch scaled_dot_product_attention 사용
    - causal mask는 is_causal=True로 처리
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        cfg.validate()

        self.n_head = cfg.n_head
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_head
        self.dropout = cfg.dropout
        self.rope_base = cfg.rope_base

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = build_rope_cache(
            seq_len=seq_len,
            head_dim=self.head_dim,
            device=x.device,
            dtype=x.dtype,
            base=self.rope_base,
        )

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        dropout_p = self.dropout if self.training else 0.0

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
        )

        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch_size, seq_len, hidden_dim)

        out = self.o_proj(attn_out)
        out = self.resid_dropout(out)

        return out


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network.

    FFN 구조:
    - silu(gate_proj(x)) * up_proj(x)
    - down_proj(...)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block.

    구조:
    x = x + Attention(RMSNorm(x))
    x = x + FFN(RMSNorm(x))
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)

        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SentilightCompactLM(nn.Module):
    """
    Sentilight-CompactLM-v0.1

    Decoder-only causal language model.

    입력:
    - idx: [B, T]

    출력:
    - logits: [B, T, vocab_size]
    - loss: targets가 주어진 경우 cross entropy loss
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        cfg.validate()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layer)]
        )

        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Tied embedding:
        # 입력 embedding과 출력 projection weight를 공유하여 파라미터 수 절감
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.cfg.init_std,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.cfg.init_std,
            )

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if idx.dim() != 2:
            raise ValueError("idx must have shape [batch, seq_len].")

        _, seq_len = idx.shape

        if seq_len > self.cfg.block_size:
            raise ValueError(
                f"sequence length {seq_len} exceeds block_size {self.cfg.block_size}"
            )

        x = self.tok_emb(idx)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets must have the same shape as idx.")

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        간단한 autoregressive generation helper.

        주의:
        - pretraining 중 샘플 확인용이다.
        - 실제 Android 추론용 최적화 코드는 별도로 작성한다.
        """
        if idx.dim() != 2:
            raise ValueError("idx must have shape [batch, seq_len].")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None and top_k > 0:
                    values, _ = torch.topk(
                        logits,
                        k=min(top_k, logits.size(-1)),
                    )
                    cutoff = values[:, [-1]]
                    logits = torch.where(
                        logits < cutoff,
                        torch.full_like(logits, -float("inf")),
                        logits,
                    )

                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)

            if eos_id is not None:
                if torch.all(next_id.squeeze(-1) == eos_id):
                    break

        return idx


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    return sum(p.numel() for p in model.parameters())


def estimate_parameter_breakdown(model: nn.Module) -> Dict[str, int]:
    """
    모델 구성 요소별 대략적인 parameter 수를 확인하기 위한 helper.
    """
    result: Dict[str, int] = {}

    for name, module in model.named_children():
        result[name] = sum(p.numel() for p in module.parameters())

    result["total"] = count_parameters(model)

    return result


def build_model_from_config_dict(config_dict: Dict[str, object]) -> SentilightCompactLM:
    """
    checkpoint에 저장된 model_config dict에서 모델 복원용.
    """
    cfg = ModelConfig(**config_dict)
    return SentilightCompactLM(cfg)


if __name__ == "__main__":
    # 간단한 구조 확인용.
    # 학습은 여기서 하지 않는다.
    cfg = ModelConfig()
    model = SentilightCompactLM(cfg)

    print("[Sentilight-CompactLM-v0.1]")
    print(cfg.to_dict())

    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)

    print(f"total parameters    : {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")

    print("\n[parameter breakdown]")
    breakdown = estimate_parameter_breakdown(model)
    for key, value in breakdown.items():
        print(f"{key:12s}: {value:,}")
