from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
MODELS_DIR = BACKEND_DIR / "models"
for path in (BACKEND_DIR, MODELS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compact_llm.training_data import build_prompt  # noqa: E402
from compact_llm.tuple_classifier import (  # noqa: E402
    SentilightTupleClassifier,
    build_tuple_vocab,
    extract_lm_state_dict,
    output_tuple,
    tuple_to_label_map,
)
from compact_llm.model_definition import ModelConfig, count_parameters  # noqa: E402


DEFAULT_TRAIN_JSONL = Path("backend/models/compact_llm/datasets/scale_grid_subsets/train_100.jsonl")
DEFAULT_VAL_JSONL = Path("backend/models/compact_llm/datasets/sft_full_runs/LATEST/val.jsonl")
DEFAULT_PRETRAIN_CKPT = Path("backend/models/compact_llm/scale_grid_runs/23M/seed_42/pretrain/compactlm_pretrain_best.pt")
DEFAULT_TOKENIZER = Path("backend/models/compact_llm/tokenizer/tokenizer.json")
DEFAULT_OUTPUT_DIR = Path("backend/models/compact_llm/tuple_head_runs/23M/seed_42")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CompactLM 240-way finite tuple classifier head")
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
    parser.add_argument("--devices", default="0")
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
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


def filter_policy_rows(
    rows: list[dict[str, Any]],
    label_map: dict[tuple[int, int, int, int, int], int],
    max_rows: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    kept_by_source: Counter[str] = Counter()
    dropped_by_source: Counter[str] = Counter()
    dropped_examples: list[dict[str, Any]] = []
    for row in rows:
        source = str(row.get("source") or "unknown")
        key = output_tuple(row["output"])
        if key in label_map:
            item = dict(row)
            item["class_id"] = label_map[key]
            kept.append(item)
            kept_by_source[source] += 1
            if max_rows is not None and len(kept) >= max_rows:
                break
        else:
            dropped_by_source[source] += 1
            if len(dropped_examples) < 10:
                dropped_examples.append(
                    {
                        "source": source,
                        "input": str(row.get("input", ""))[:120],
                        "output": row.get("output"),
                    }
                )
    return kept, {
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "dropped_rows": sum(dropped_by_source.values()),
        "kept_by_source": dict(sorted(kept_by_source.items())),
        "dropped_by_source": dict(sorted(dropped_by_source.items())),
        "dropped_examples": dropped_examples,
        "max_rows": max_rows,
    }


class TupleDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_seq_len: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        token_ids = self.tokenizer.encode(build_prompt(str(row["input"]))).ids[-self.max_seq_len :]
        if not token_ids:
            token_ids = [0]
        attention_mask = [1] * len(token_ids)
        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "labels": int(row["class_id"]),
        }


def collate_batch(batch: list[dict[str, Any]], torch: Any) -> dict[str, Any]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    labels: list[int] = []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [0] * pad)
        attention_masks.append(item["attention_mask"] + [0] * pad)
        labels.append(item["labels"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def unwrap_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def evaluate(model: Any, loader: Any, primary_device: Any, torch: Any) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    total_correct = 0
    autocast_dtype = torch.bfloat16 if primary_device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(primary_device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(primary_device, non_blocking=True)
            labels = batch["labels"].to(primary_device, non_blocking=True)
            with torch.autocast(device_type=primary_device.type, dtype=autocast_dtype, enabled=primary_device.type == "cuda"):
                logits, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += float(loss.item()) * labels.numel()
            total_rows += labels.numel()
            total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
    model.train()
    total_rows = max(total_rows, 1)
    return {
        "loss": total_loss / total_rows,
        "accuracy": total_correct / total_rows,
        "rows": float(total_rows),
    }


def save_checkpoint(
    path: Path,
    model: Any,
    optimizer: Any,
    epoch: int,
    best_val_loss: float,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    import torch

    core = unwrap_model(model)
    payload = core.checkpoint_payload(metrics)
    payload.update(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": metadata,
        }
    )
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    import torch
    from tokenizers import Tokenizer
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for tuple classifier training")
    device_ids = [int(value) for value in args.devices.split(",") if value.strip()]
    primary_device = torch.device(f"cuda:{device_ids[0]}")
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    tuple_vocab = build_tuple_vocab()
    label_map = tuple_to_label_map(tuple_vocab)
    if len(tuple_vocab) != 240:
        raise RuntimeError(f"Expected 240 tuple classes, got {len(tuple_vocab)}")

    train_rows, train_filter = filter_policy_rows(load_jsonl(args.train_jsonl), label_map, args.max_train_rows)
    val_rows, val_filter = filter_policy_rows(load_jsonl(args.val_jsonl), label_map, args.max_val_rows)
    if not train_rows:
        raise RuntimeError("No train rows remain after policy tuple filtering")
    if not val_rows:
        raise RuntimeError("No validation rows remain after policy tuple filtering")

    checkpoint = torch.load(args.pretrained_checkpoint, map_location="cpu")
    cfg = ModelConfig(**checkpoint["model_config"])
    model = SentilightTupleClassifier(cfg, tuple_vocab=tuple_vocab)
    model.load_lm_backbone_state(extract_lm_state_dict(checkpoint))
    if args.freeze_backbone:
        for name, parameter in model.named_parameters():
            if not name.startswith("tuple_head."):
                parameter.requires_grad = False
    model.to(primary_device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])

    train_loader = DataLoader(
        TupleDataset(train_rows, tokenizer, args.max_seq_len),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_batch(batch, torch),
    )
    val_loader = DataLoader(
        TupleDataset(val_rows, tokenizer, args.max_seq_len),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, min(args.num_workers, 2)),
        pin_memory=True,
        collate_fn=lambda batch: collate_batch(batch, torch),
    )

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    autocast_dtype = torch.bfloat16 if primary_device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "compactlm_tuple_classifier_best.pt"
    last_path = args.output_dir / "compactlm_tuple_classifier_last.pt"
    summary_path = args.output_dir / "tuple_classifier_summary.json"
    metadata = {
        "train_jsonl": str(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl),
        "pretrained_checkpoint": str(args.pretrained_checkpoint),
        "tokenizer_path": str(args.tokenizer_path),
        "seed": args.seed,
        "tuple_count": len(tuple_vocab),
        "train_filter": train_filter,
        "val_filter": val_filter,
        "freeze_backbone": args.freeze_backbone,
        "parameter_count": count_parameters(unwrap_model(model)),
        "trainable_parameter_count": count_parameters(unwrap_model(model), trainable_only=True),
    }

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        total_correct = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(primary_device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(primary_device, non_blocking=True)
            labels = batch["labels"].to(primary_device, non_blocking=True)
            with torch.autocast(device_type=primary_device.type, dtype=autocast_dtype, enabled=primary_device.type == "cuda"):
                logits, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            total_loss += float(loss.item()) * labels.numel()
            total_rows += labels.numel()
            total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
        val_metrics = evaluate(model, val_loader, primary_device, torch)
        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": total_loss / max(total_rows, 1),
            "train_accuracy": total_correct / max(total_rows, 1),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(metrics)
        save_checkpoint(last_path, model, optimizer, epoch, min(best_val_loss, val_metrics["loss"]), metrics, metadata)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(best_path, model, optimizer, epoch, best_val_loss, metrics, metadata)
        print(json.dumps(metrics, ensure_ascii=False))

    summary = {"metadata": metadata, "history": history, "best_val_loss": best_val_loss, "best_checkpoint": str(best_path), "last_checkpoint": str(last_path)}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
