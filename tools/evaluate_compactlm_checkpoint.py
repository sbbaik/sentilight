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
from compact_llm.inference import load_model, predict  # noqa: E402
from compact_llm.model_definition import count_parameters  # noqa: E402


DEFAULT_DATASET = Path("backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl")
DEFAULT_MODEL_DIR = Path("backend/models/compact_llm")


class RuntimeConfig:
    def __init__(self, model_dir: Path, checkpoint: Path, tokenizer_dir: str, device: str, max_new_tokens: int) -> None:
        self.resolved_model_dir = str(model_dir)
        self.checkpoint = str(checkpoint)
        self.tokenizer_dir = tokenizer_dir
        self.options = {"device": device, "max_new_tokens": max_new_tokens}


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


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    args.checkpoint = args.checkpoint.resolve()
    args.model_dir = args.model_dir.resolve()
    args.dataset = args.dataset.resolve()
    args.per_row_jsonl = args.per_row_jsonl.resolve()
    args.report_json = args.report_json.resolve()
    args.report_md = args.report_md.resolve()
    rows = select_rows(load_jsonl(args.dataset), args.segment, args.max_rows)
    descriptor = ModelDescriptor(
        model_id=args.model_id,
        display_name=args.display_name,
        adapter="custom_pytorch",
        mode="direct_checkpoint",
    )
    buckets = build_model_buckets([descriptor])
    runtime = load_model(
        RuntimeConfig(
            model_dir=args.model_dir,
            checkpoint=args.checkpoint,
            tokenizer_dir=args.tokenizer_dir,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
    )
    core_model = runtime.model
    parameter_count = count_parameters(core_model)

    args.per_row_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.per_row_jsonl.open("w", encoding="utf-8") as per_row_handle:
        for ordinal, (row_index, row) in enumerate(rows, start=1):
            expected = normalize_lighting(row["output"])
            started_at = time.perf_counter()
            try:
                predicted = predict(str(row["input"]), runtime)
                latency_ms = (time.perf_counter() - started_at) * 1000.0
                result = {
                    "success": True,
                    "lighting": predicted,
                    "raw": predicted,
                    "latency_ms": latency_ms,
                }
            except Exception as exc:  # noqa: BLE001 - record model/eval failures per row
                latency_ms = (time.perf_counter() - started_at) * 1000.0
                result = {
                    "success": False,
                    "lighting": None,
                    "raw": None,
                    "latency_ms": latency_ms,
                    "error": str(exc),
                }

            prediction = _record_prediction(buckets[descriptor.model_id], result, expected, row)
            per_row_handle.write(
                json.dumps(
                    build_per_row_record(
                        mode="direct_checkpoint",
                        row_index=row_index,
                        row=row,
                        model=descriptor,
                        expected=expected,
                        prediction=prediction,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
            if args.progress_every > 0 and ordinal % args.progress_every == 0:
                print(f"[{args.model_id}] progress {ordinal}/{len(rows)}")

    metrics = finalize_bucket(buckets[descriptor.model_id], len(rows))
    return {
        "dataset": str(args.dataset),
        "segment": args.segment,
        "rows": len(rows),
        "checkpoint": str(args.checkpoint),
        "model_dir": str(args.model_dir),
        "model_id": args.model_id,
        "display_name": args.display_name,
        "device": args.device,
        "parameter_count": parameter_count,
        "model_config": runtime.model.cfg.to_dict(),
        "per_row_jsonl": str(args.per_row_jsonl),
        "coarse_dimensions": list(COARSE_DIMENSIONS),
        "metrics": metrics,
    }


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    latency = metrics["latency"]
    return "\n".join(
        [
            "# CompactLM Direct Checkpoint Evaluation",
            "",
            f"- Model: `{report['model_id']}`",
            f"- Checkpoint: `{report['checkpoint']}`",
            f"- Dataset: `{report['dataset']}`",
            f"- Segment: `{report['segment']}`",
            f"- Rows: `{report['rows']}`",
            f"- Parameters: `{report['parameter_count']}`",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CompactLM checkpoint directly against SentiLight benchmark rows")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--tokenizer-dir", default="tokenizer")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--segment", choices=("rule", "natural", "all"), default="rule")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-id", default="compact_llm_direct")
    parser.add_argument("--display-name", default="CompactLM Direct")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--per-row-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_checkpoint(args)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report), end="")


if __name__ == "__main__":
    main()
