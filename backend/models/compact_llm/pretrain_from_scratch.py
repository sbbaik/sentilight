from __future__ import annotations

import argparse
import json
import math
import random
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(BASE_DIR.parent))

DEFAULT_TOKENS = BASE_DIR / "datasets" / "sentilight_color_emotion_v1" / "pretrain_tokens.npy"
DEFAULT_TOKENIZER = BASE_DIR / "tokenizer" / "tokenizer.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "from_scratch_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain Sentilight CompactLM from scratch on mixed corpus")
    parser.add_argument("--tokens-npy", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=120)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=40)
    parser.add_argument("--save-interval", type=int, default=400)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=str, default="0,1")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device, torch):
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.empty((batch_size, block_size), dtype=torch.long)
    y = torch.empty((batch_size, block_size), dtype=torch.long)
    for i, start in enumerate(starts.tolist()):
        chunk = np.asarray(data[start : start + block_size + 1], dtype=np.int64)
        x[i] = torch.from_numpy(chunk[:-1])
        y[i] = torch.from_numpy(chunk[1:])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def build_amp_context(torch, primary_device):
    if primary_device.type != "cuda":
        return nullcontext(), False
    if torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16), False
    return torch.autocast(device_type="cuda", dtype=torch.float16), True


def estimate_loss(model, train_data, valid_data, args, primary_device, torch):
    model.eval()
    amp_context, _ = build_amp_context(torch, primary_device)
    results = {}
    with torch.inference_mode():
        for split_name, split_data in [("train", train_data), ("valid", valid_data)]:
            losses = []
            for _ in range(args.eval_iters):
                xb, yb = get_batch(split_data, args.batch_size, args.block_size, primary_device, torch)
                with amp_context:
                    _, loss = model(xb, yb)
                    if getattr(loss, "ndim", 0) > 0:
                        loss = loss.mean()
                losses.append(float(loss.item()))
            results[split_name] = float(np.mean(losses))
    model.train()
    return results


def save_checkpoint(path: Path, model, optimizer, step: int, best_valid_loss: float, metrics: dict[str, Any]) -> None:
    import torch

    core = unwrap_model(model)
    payload = {
        "step": step,
        "best_valid_loss": best_valid_loss,
        "model_state_dict": core.state_dict(),
        "model_config": asdict(core.cfg),
        "metrics": metrics,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    import torch
    from tokenizers import Tokenizer
    from compact_llm.model_definition import ModelConfig, SentilightCompactLM, count_parameters

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for from-scratch pretraining")

    device_ids = [int(value) for value in args.devices.split(",") if value.strip()]
    if not device_ids:
        raise ValueError("at least one CUDA device id is required")
    primary_device = torch.device(f"cuda:{device_ids[0]}")

    token_data = np.load(args.tokens_npy, mmap_mode="r")
    split_index = max(args.block_size + 2, int(len(token_data) * 0.95))
    train_data = np.asarray(token_data[:split_index], dtype=np.uint32)
    valid_data = np.asarray(token_data[split_index:], dtype=np.uint32)
    if len(valid_data) <= args.block_size + 1:
        raise RuntimeError("validation token slice is too small")

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    cfg = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    cfg.validate()
    model = SentilightCompactLM(cfg)
    model.to(primary_device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay)
    amp_context, use_scaler = build_amp_context(torch, primary_device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if use_scaler else None

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "compactlm_pretrain_best.pt"
    last_path = output_dir / "compactlm_pretrain_last.pt"
    log_path = output_dir / "pretrain_log.jsonl"
    run_config_path = output_dir / "pretrain_run_config.json"
    run_config_path.write_text(
        json.dumps(
            {
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "model_config": cfg.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({
        "devices": device_ids,
        "primary_device": str(primary_device),
        "total_parameters": count_parameters(unwrap_model(model)),
        "tokens": int(len(token_data)),
        "train_tokens": int(len(train_data)),
        "valid_tokens": int(len(valid_data)),
    }, ensure_ascii=False))

    best_valid_loss = float("inf")
    for step in range(1, args.steps + 1):
        lr = args.min_learning_rate
        if step < args.warmup_steps:
            lr = args.learning_rate * step / max(1, args.warmup_steps)
        else:
            progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = args.min_learning_rate + coeff * (args.learning_rate - args.min_learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr

        xb, yb = get_batch(train_data, args.batch_size, args.block_size, primary_device, torch)
        optimizer.zero_grad(set_to_none=True)
        with amp_context:
            _, loss = model(xb, yb)
            if getattr(loss, "ndim", 0) > 0:
                loss = loss.mean()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        if step % args.eval_interval == 0 or step == 1 or step == args.steps:
            eval_result = estimate_loss(model, train_data, valid_data, args, primary_device, torch)
            metrics = {
                "step": step,
                "lr": lr,
                "train_loss_batch": float(loss.item()),
                "train_loss_eval": eval_result["train"],
                "valid_loss": eval_result["valid"],
                "valid_ppl": math.exp(min(eval_result["valid"], 20.0)),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            print(json.dumps(metrics, ensure_ascii=False))
            save_checkpoint(last_path, model, optimizer, step, min(best_valid_loss, eval_result["valid"]), metrics)
            if eval_result["valid"] < best_valid_loss:
                best_valid_loss = eval_result["valid"]
                save_checkpoint(best_path, model, optimizer, step, best_valid_loss, metrics)
        elif step % args.save_interval == 0:
            save_checkpoint(last_path, model, optimizer, step, best_valid_loss, {"step": step, "lr": lr, "train_loss_batch": float(loss.item())})


if __name__ == "__main__":
    main()
