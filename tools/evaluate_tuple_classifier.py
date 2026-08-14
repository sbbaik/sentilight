from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
MODELS_DIR = BACKEND_DIR / "models"
for path in (BACKEND_DIR, MODELS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.benchmark_eval import (  # noqa: E402
    COARSE_DIMENSIONS,
    ModelDescriptor,
    _record_prediction,
    build_model_buckets,
    build_per_row_record,
    finalize_bucket,
    load_jsonl,
    normalize_lighting,
)
from compact_llm.model_definition import ModelConfig, count_parameters  # noqa: E402
from compact_llm.training_data import build_prompt  # noqa: E402
from compact_llm.tuple_classifier import SentilightTupleClassifier  # noqa: E402


DEFAULT_DATASET = Path("backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl")
DEFAULT_TOKENIZER = Path("backend/models/compact_llm/tokenizer/tokenizer.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CompactLM tuple classifier checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--segment", choices=("rule", "natural", "all"), default="rule")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--per-row-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    return parser.parse_args()


def select_rows(rows: list[dict[str, Any]], segment: str, max_rows: int | None) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate(rows, start=1):
        source = row.get("source")
        if segment == "rule" and source == "natural_language_baseline":
            continue
        if segment == "natural" and source != "natural_language_baseline":
            continue
        selected.append((index, row))
        if max_rows is not None and len(selected) >= max_rows:
            break
    return selected


def collate_prompts(items: list[tuple[int, dict[str, Any]]], tokenizer: Any, max_seq_len: int, torch: Any) -> dict[str, Any]:
    encoded: list[list[int]] = []
    for _, row in items:
        token_ids = tokenizer.encode(build_prompt(str(row["input"]))).ids[-max_seq_len:]
        encoded.append(token_ids or [0])
    max_len = max(len(ids) for ids in encoded)
    input_ids = [ids + [0] * (max_len - len(ids)) for ids in encoded]
    attention_mask = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in encoded]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def load_classifier(checkpoint_path: Path, device: Any, torch: Any) -> tuple[Any, list[dict[str, int]], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = ModelConfig(**checkpoint["model_config"])
    tuple_vocab = checkpoint["tuple_vocab"]
    model = SentilightTupleClassifier(cfg, tuple_vocab=tuple_vocab)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, tuple_vocab, checkpoint


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from tokenizers import Tokenizer

    device = torch.device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    model, tuple_vocab, checkpoint = load_classifier(args.checkpoint.resolve(), device, torch)
    rows = select_rows(load_jsonl(args.dataset), args.segment, args.max_rows)
    descriptor = ModelDescriptor(
        model_id=args.model_id,
        display_name=args.display_name,
        adapter="custom_pytorch",
        mode="tuple_classifier",
    )
    buckets = build_model_buckets([descriptor])
    args.per_row_jsonl.parent.mkdir(parents=True, exist_ok=True)
    autocast_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    with args.per_row_jsonl.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_items = rows[start : start + args.batch_size]
            batch = collate_prompts(batch_items, tokenizer, args.max_seq_len, torch)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            started_at = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
                logits, _ = model(input_ids, attention_mask=attention_mask)
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            classes = logits.argmax(dim=-1).tolist()
            per_row_latency = elapsed_ms / max(1, len(batch_items))
            for (row_index, row), class_id in zip(batch_items, classes):
                expected = normalize_lighting(row["output"])
                predicted = tuple_vocab[int(class_id)]
                result = {
                    "success": True,
                    "lighting": predicted,
                    "raw": {"class_id": int(class_id), "lighting": predicted},
                    "latency_ms": per_row_latency,
                }
                prediction = _record_prediction(buckets[descriptor.model_id], result, expected, row)
                record = build_per_row_record(
                    mode="tuple_classifier",
                    row_index=row_index,
                    row=row,
                    model=descriptor,
                    expected=expected,
                    prediction=prediction,
                )
                record["class_id"] = int(class_id)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics = finalize_bucket(buckets[descriptor.model_id], len(rows))
    return {
        "dataset": str(args.dataset),
        "segment": args.segment,
        "rows": len(rows),
        "checkpoint": str(args.checkpoint),
        "model_id": args.model_id,
        "display_name": args.display_name,
        "device": args.device,
        "parameter_count": count_parameters(model),
        "tuple_count": len(tuple_vocab),
        "checkpoint_metadata": checkpoint.get("metadata", {}),
        "per_row_jsonl": str(args.per_row_jsonl),
        "coarse_dimensions": list(COARSE_DIMENSIONS),
        "metrics": metrics,
    }


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    latency = metrics["latency"]
    return "\n".join(
        [
            "# CompactLM Tuple Classifier Evaluation",
            "",
            f"- Model: `{report['model_id']}`",
            f"- Checkpoint: `{report['checkpoint']}`",
            f"- Dataset: `{report['dataset']}`",
            f"- Segment: `{report['segment']}`",
            f"- Rows: `{report['rows']}`",
            f"- Parameters: `{report['parameter_count']}`",
            f"- Tuple classes: `{report['tuple_count']}`",
            f"- Per-row JSONL: `{report['per_row_jsonl']}`",
            "",
            "| Success | Strict | Coarse | Mean Latency | P95 Latency |",
            "|---:|---:|---:|---:|---:|",
            (
                f"| {metrics['successes']}/{report['rows']} "
                f"| {metrics['strict_semantic_passes']} ({metrics['strict_semantic_pass_rate']:.2%}) "
                f"| {metrics['coarse_3bin_passes']} ({metrics['coarse_3bin_pass_rate']:.2%}) "
                f"| {latency['mean_ms']} "
                f"| {latency['p95_ms']} |"
            ),
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    report = evaluate(args)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report), end="")


if __name__ == "__main__":
    main()
