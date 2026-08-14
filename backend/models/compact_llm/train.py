from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_TRAIN_JSONL = BASE_DIR / "datasets" / "sentilight_color_emotion_v1" / "train.jsonl"
DEFAULT_VAL_JSONL = BASE_DIR / "datasets" / "sentilight_color_emotion_v1" / "val.jsonl"
DEFAULT_CHECKPOINT = BASE_DIR / "checkpoint" / "sentilight_compactlm_final.pt"
DEFAULT_TOKENIZER = BASE_DIR / "tokenizer" / "tokenizer.json"

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(BASE_DIR.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SentiLight CompactLM")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--val-jsonl", type=Path, default=DEFAULT_VAL_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def collate_batch(batch: list[dict[str, Any]], torch):
    return {
        "input_ids": torch.tensor([item["input_ids"] for item in batch], dtype=torch.long),
        "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.long),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long),
    }


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def evaluate(model, loader, device, torch):
    import torch.nn.functional as F

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            valid_tokens = int((labels != -100).sum().item())
            total_loss += float(loss.item()) * max(valid_tokens, 1)
            total_tokens += valid_tokens
    total_tokens = max(total_tokens, 1)
    mean_loss = total_loss / total_tokens
    return {"loss": mean_loss, "ppl": math.exp(min(mean_loss, 20.0))}


def save_checkpoint(
    *,
    path: Path,
    model,
    optimizer,
    epoch: int,
    best_val_loss: float,
    metrics: dict[str, Any],
) -> None:
    import torch

    model_core = unwrap_model(model)
    payload = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model_core.state_dict(),
        "model_config": asdict(model_core.cfg),
        "metrics": metrics,
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

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    if not args.train_jsonl.exists() or not args.val_jsonl.exists():
        raise FileNotFoundError("Prepared dataset files are missing. Run prepare_dataset.py first.")

    device = torch.device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    train_rows = load_jsonl(args.train_jsonl)
    val_rows = load_jsonl(args.val_jsonl)

    train_ds = SFTDataset(train_rows, tokenizer, args.max_seq_len)
    val_ds = SFTDataset(val_rows, tokenizer, args.max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_batch(batch, torch),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, min(args.num_workers, 2)),
        pin_memory=True,
        collate_fn=lambda batch: collate_batch(batch, torch),
    )

    if args.resume_checkpoint.exists():
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")
        model_config = ModelConfig(**checkpoint["model_config"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", checkpoint.get("metrics", {}).get("loss", float("inf"))))
    else:
        raise FileNotFoundError(f"resume checkpoint not found: {args.resume_checkpoint}")

    model = SentilightCompactLM(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_history: list[dict[str, Any]] = []
    global_step = 0
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None

    for epoch_offset in range(1, args.epochs + 1):
        epoch = start_epoch + epoch_offset
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_tokens = 0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
                logits, _ = model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
                loss = loss / args.grad_accum_steps

            loss.backward()
            valid_tokens = int((labels != -100).sum().item())
            running_loss += float(loss.item()) * args.grad_accum_steps * max(valid_tokens, 1)
            running_tokens += valid_tokens

            if step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        train_loss = running_loss / max(running_tokens, 1)
        val_metrics = evaluate(model, val_loader, device, torch)
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_ppl": val_metrics["ppl"],
            "learning_rate": scheduler.get_last_lr()[0],
            "global_step": global_step,
        }
        train_history.append(metrics)

        last_path = args.output_dir / "sentilight_compactlm_last.pt"
        best_path = args.output_dir / "sentilight_compactlm_best.pt"
        legacy_best_path = args.output_dir / "sentilight_compactlm_kote_sft_best.pt"
        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=min(best_val_loss, val_metrics["loss"]),
            metrics=metrics,
        )
        if args.save_every_epoch:
            save_checkpoint(
                path=args.output_dir / f"sentilight_compactlm_epoch_{epoch}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=min(best_val_loss, val_metrics["loss"]),
                metrics=metrics,
            )
        if val_metrics["loss"] <= best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                metrics=metrics,
            )
            save_checkpoint(
                path=legacy_best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                metrics=metrics,
            )

        print(json.dumps(metrics, ensure_ascii=False))

    summary_path = args.output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(train_history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
