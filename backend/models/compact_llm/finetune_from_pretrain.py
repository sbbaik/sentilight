from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_TRAIN_JSONL = BASE_DIR / "datasets" / "sentilight_color_emotion_v1" / "train.jsonl"
DEFAULT_VAL_JSONL = BASE_DIR / "datasets" / "sentilight_color_emotion_v1" / "val.jsonl"
DEFAULT_PRETRAIN_CKPT = BASE_DIR / "from_scratch_runs" / "compactlm_pretrain_best.pt"
DEFAULT_TOKENIZER = BASE_DIR / "tokenizer" / "tokenizer.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "from_scratch_runs" / "finetune"
CONTROL_FIELDS = ("H", "S", "B", "Dimmer", "CT")

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(BASE_DIR.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune CompactLM from scratch-pretrained checkpoint")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--val-jsonl", type=Path, default=DEFAULT_VAL_JSONL)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=DEFAULT_PRETRAIN_CKPT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=str, default="0,1")
    parser.add_argument("--freq-json", type=Path, default=None)
    parser.add_argument("--freq-alpha", type=float, default=0.0)
    parser.add_argument("--freq-max-ratio", type=float, default=50.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tuple_key_from_output(output: dict[str, Any]) -> str:
    return "|".join(str(int(output[field])) for field in CONTROL_FIELDS)


def load_frequency_table(path: Path | None) -> tuple[dict[str, int], dict[str, Any]]:
    if path is None:
        return {}, {"path": None, "sha256": None, "tuples": 0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("frequencies", payload) if isinstance(payload, dict) else {}
    if not isinstance(raw, dict):
        raise ValueError(f"invalid frequency table: {path}")
    frequencies = {str(key): int(value) for key, value in raw.items()}
    return frequencies, {"path": str(path), "sha256": sha256_file(path), "tuples": len(frequencies)}


class SFTDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_seq_len: int):
        from compact_llm.training_data import build_prompt, build_target_text

        self.rows = rows
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.build_prompt = build_prompt
        self.build_target_text = build_target_text

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        prompt = self.build_prompt(str(row["input"]))
        target = self.build_target_text(row["output"])
        tuple_key = tuple_key_from_output(row["output"])
        prompt_ids = self.tokenizer.encode(prompt).ids
        target_ids = self.tokenizer.encode(target).ids
        full_ids = (prompt_ids + target_ids)[: self.max_seq_len + 1]
        if len(full_ids) < 2:
            full_ids = (prompt_ids + target_ids + [0, 0])[:2]
        input_ids = full_ids[:-1]
        labels = full_ids[1:]
        prompt_cut = min(max(len(prompt_ids) - 1, 0), len(labels))
        labels[:prompt_cut] = [-100] * prompt_cut
        attention_mask = [1] * len(input_ids)
        if len(input_ids) < self.max_seq_len:
            pad = self.max_seq_len - len(input_ids)
            input_ids.extend([0] * pad)
            labels.extend([-100] * pad)
            attention_mask.extend([0] * pad)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask, "tuple_key": tuple_key}


def collate_batch(batch: list[dict[str, Any]], torch):
    return {
        "input_ids": torch.tensor([item["input_ids"] for item in batch], dtype=torch.long),
        "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.long),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long),
        "tuple_key": [item["tuple_key"] for item in batch],
    }


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def evaluate(model, loader, primary_device, torch):
    import torch.nn.functional as F

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    autocast_dtype = torch.bfloat16 if primary_device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(primary_device, non_blocking=True)
            labels = batch["labels"].to(primary_device, non_blocking=True)
            with torch.autocast(device_type=primary_device.type, dtype=autocast_dtype, enabled=primary_device.type == "cuda"):
                logits, _ = model(input_ids)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
            valid_tokens = int((labels != -100).sum().item())
            total_loss += float(loss.item()) * max(valid_tokens, 1)
            total_tokens += valid_tokens
    total_tokens = max(total_tokens, 1)
    model.train()
    mean_loss = total_loss / total_tokens
    return {"loss": mean_loss, "ppl": math.exp(min(mean_loss, 20.0))}


def build_batch_weights(
    tuple_keys: list[str],
    frequencies: dict[str, int],
    alpha: float,
    max_ratio: float,
    device: Any,
    torch: Any,
):
    if alpha <= 0.0:
        return torch.ones(len(tuple_keys), dtype=torch.float32, device=device)
    missing = sorted({key for key in tuple_keys if key not in frequencies})
    if missing:
        sample = ", ".join(missing[:5])
        raise KeyError(f"frequency table missing {len(missing)} tuple key(s), e.g. {sample}")
    weights = torch.tensor(
        [float(frequencies[key]) ** (-alpha) for key in tuple_keys],
        dtype=torch.float32,
        device=device,
    )
    if max_ratio > 0 and weights.numel() > 1:
        min_weight = weights.min()
        max_weight = weights.max()
        if float(min_weight.item()) > 0.0 and float((max_weight / min_weight).item()) > max_ratio:
            weights = weights.clamp(max=min_weight * max_ratio)
    mean_weight = weights.mean().clamp_min(1e-12)
    return weights / mean_weight


def weighted_token_cross_entropy(logits, labels, sample_weights, torch):
    import torch.nn.functional as F

    vocab_size = logits.size(-1)
    token_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(labels)
    valid_mask = labels.ne(-100).to(token_loss.dtype)
    weights = sample_weights.to(dtype=token_loss.dtype).unsqueeze(1)
    denominator = (valid_mask * weights).sum().clamp_min(1.0)
    return (token_loss * valid_mask * weights).sum() / denominator


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    best_val_loss: float,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    import torch

    core = unwrap_model(model)
    payload = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": core.state_dict(),
        "model_config": asdict(core.cfg),
        "metrics": metrics,
        "metadata": metadata,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    import torch
    import torch.nn.functional as F
    from tokenizers import Tokenizer
    from torch.utils.data import DataLoader
    from compact_llm.model_definition import ModelConfig, SentilightCompactLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fine-tuning")

    device_ids = [int(value) for value in args.devices.split(",") if value.strip()]
    primary_device = torch.device(f"cuda:{device_ids[0]}")

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    train_rows = load_jsonl(args.train_jsonl)
    val_rows = load_jsonl(args.val_jsonl)
    if args.max_train_rows is not None:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_val_rows is not None:
        val_rows = val_rows[: args.max_val_rows]
    frequencies, frequency_metadata = load_frequency_table(args.freq_json)
    if args.freq_alpha > 0.0 and not frequencies:
        raise ValueError("--freq-json is required when --freq-alpha > 0")
    train_keys = {tuple_key_from_output(row["output"]) for row in train_rows}
    missing_train_keys = sorted(train_keys - set(frequencies)) if args.freq_alpha > 0.0 else []
    if missing_train_keys:
        sample = ", ".join(missing_train_keys[:5])
        raise KeyError(f"frequency table is missing {len(missing_train_keys)} training tuple key(s), e.g. {sample}")

    train_loader = DataLoader(SFTDataset(train_rows, tokenizer, args.max_seq_len), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=lambda batch: collate_batch(batch, torch))
    val_loader = DataLoader(SFTDataset(val_rows, tokenizer, args.max_seq_len), batch_size=args.batch_size, shuffle=False, num_workers=max(1, min(args.num_workers, 2)), pin_memory=True, collate_fn=lambda batch: collate_batch(batch, torch))

    checkpoint = torch.load(args.pretrained_checkpoint, map_location="cpu")
    model_config = ModelConfig(**checkpoint["model_config"])
    model = SentilightCompactLM(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(primary_device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    autocast_dtype = torch.bfloat16 if primary_device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "compactlm_from_scratch_best.pt"
    last_path = output_dir / "compactlm_from_scratch_last.pt"
    summary_path = output_dir / "finetune_summary.json"
    metadata = {
        "train_jsonl": str(args.train_jsonl),
        "train_jsonl_sha256": sha256_file(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl),
        "val_jsonl_sha256": sha256_file(args.val_jsonl),
        "pretrained_checkpoint": str(args.pretrained_checkpoint),
        "pretrained_checkpoint_sha256": sha256_file(args.pretrained_checkpoint),
        "tokenizer_path": str(args.tokenizer_path),
        "tokenizer_sha256": sha256_file(args.tokenizer_path),
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "devices": args.devices,
        "freq_alpha": args.freq_alpha,
        "freq_max_ratio": args.freq_max_ratio,
        "freq_table": frequency_metadata,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
    }

    best_val_loss = float("inf")
    history: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_tokens = 0
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            input_ids = batch["input_ids"].to(primary_device, non_blocking=True)
            labels = batch["labels"].to(primary_device, non_blocking=True)
            with torch.autocast(device_type=primary_device.type, dtype=autocast_dtype, enabled=primary_device.type == "cuda"):
                logits, _ = model(input_ids)
                if args.freq_alpha > 0.0:
                    sample_weights = build_batch_weights(
                        batch["tuple_key"],
                        frequencies,
                        args.freq_alpha,
                        args.freq_max_ratio,
                        primary_device,
                        torch,
                    )
                    loss = weighted_token_cross_entropy(logits, labels, sample_weights, torch)
                else:
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            valid_tokens = int((labels != -100).sum().item())
            running_loss += float(loss.item()) * max(valid_tokens, 1)
            running_tokens += valid_tokens

        train_loss = running_loss / max(running_tokens, 1)
        val_metrics = evaluate(model, val_loader, primary_device, torch)
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_ppl": val_metrics["ppl"],
            "learning_rate": scheduler.get_last_lr()[0],
            "global_step": global_step,
        }
        history.append(metrics)
        save_checkpoint(last_path, model, optimizer, epoch, min(best_val_loss, val_metrics["loss"]), metrics, metadata)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(best_path, model, optimizer, epoch, best_val_loss, metrics, metadata)
        print(json.dumps(metrics, ensure_ascii=False))

    summary = {
        "metadata": metadata,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
