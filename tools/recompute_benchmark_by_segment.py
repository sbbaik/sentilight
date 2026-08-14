#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from common.benchmark_eval import (  # noqa: E402
    COARSE_DIMENSIONS,
    coarse_3bin_pass,
    coarse_bins,
    hue_distance,
    normalize_lighting,
    strict_semantic_pass,
)


NATURAL_SOURCE = "natural_language_baseline"
SEGMENT_ORDER = ("rule_based_1000", "natural_1000", "all_2000")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.fmean(values), 2)


def safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def segment_names(source: str | None) -> list[str]:
    names = ["all_2000"]
    if source == NATURAL_SOURCE:
        names.append("natural_1000")
    else:
        names.append("rule_based_1000")
    return names


def empty_bucket(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": record["model_id"],
        "display_name": record.get("display_name") or record["model_id"],
        "adapter": record.get("adapter"),
        "model_mode": record.get("model_mode"),
        "n": 0,
        "successes": 0,
        "errors": 0,
        "strict_passes": 0,
        "coarse_passes": 0,
        "latencies_ms": [],
        "mae_h": [],
        "mae_s": [],
        "mae_b": [],
        "mae_dimmer": [],
        "mae_ct": [],
        "coarse_match_counts": {key: 0 for key in COARSE_DIMENSIONS},
        "coarse_mismatch_counts": {key: 0 for key in COARSE_DIMENSIONS},
    }


def strict_value(record: dict[str, Any], expected: dict[str, int], predicted: dict[str, int]) -> bool:
    if "strict" in record:
        return bool(record["strict"])
    row = {
        "emotion": record.get("emotion"),
        "base_color": record.get("base_color"),
    }
    return strict_semantic_pass(expected, predicted, row)


def coarse_value(record: dict[str, Any], expected: dict[str, int], predicted: dict[str, int]) -> bool:
    if "coarse" in record:
        return bool(record["coarse"])
    return coarse_3bin_pass(expected, predicted)


def update_bucket(bucket: dict[str, Any], record: dict[str, Any]) -> None:
    bucket["n"] += 1
    if not record.get("success"):
        bucket["errors"] += 1
        return

    bucket["successes"] += 1
    latency = record.get("latency_ms")
    if isinstance(latency, (int, float)):
        bucket["latencies_ms"].append(float(latency))

    expected = normalize_lighting(record["reference"])
    predicted = normalize_lighting(record["predicted"])
    bucket["mae_h"].append(float(hue_distance(predicted["H"], expected["H"])))
    bucket["mae_s"].append(abs(predicted["S"] - expected["S"]))
    bucket["mae_b"].append(abs(predicted["B"] - expected["B"]))
    bucket["mae_dimmer"].append(abs(predicted["Dimmer"] - expected["Dimmer"]))
    bucket["mae_ct"].append(abs(predicted["CT"] - expected["CT"]))

    expected_coarse = record.get("coarse_expected") or coarse_bins(expected)
    predicted_coarse = record.get("coarse_predicted") or coarse_bins(predicted)
    for key in COARSE_DIMENSIONS:
        if expected_coarse[key] == predicted_coarse[key]:
            bucket["coarse_match_counts"][key] += 1
        else:
            bucket["coarse_mismatch_counts"][key] += 1

    if strict_value(record, expected, predicted):
        bucket["strict_passes"] += 1
    if coarse_value(record, expected, predicted):
        bucket["coarse_passes"] += 1


def finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    successes = int(bucket["successes"])
    n = int(bucket["n"])
    return {
        "model_id": bucket["model_id"],
        "display_name": bucket["display_name"],
        "adapter": bucket["adapter"],
        "model_mode": bucket["model_mode"],
        "n": n,
        "successes": successes,
        "errors": bucket["errors"],
        "success_rate": safe_rate(successes, n),
        "strict_passes": bucket["strict_passes"],
        "strict_rate": safe_rate(bucket["strict_passes"], n),
        "coarse_passes": bucket["coarse_passes"],
        "coarse_rate": safe_rate(bucket["coarse_passes"], n),
        "latency": {
            "mean_ms": safe_mean(bucket["latencies_ms"]),
            "p95_ms": percentile(bucket["latencies_ms"], 0.95),
        },
        "mae": {
            "H": safe_mean(bucket["mae_h"]),
            "S": safe_mean(bucket["mae_s"]),
            "B": safe_mean(bucket["mae_b"]),
            "Dimmer": safe_mean(bucket["mae_dimmer"]),
            "CT": safe_mean(bucket["mae_ct"]),
        },
        "coarse_match_rates": {
            key: safe_rate(bucket["coarse_match_counts"][key], successes)
            for key in COARSE_DIMENSIONS
        },
    }


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return round(ordered[index], 2)


def validate_records(dataset_rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for record in records:
        row_index = int(record.get("row_index", 0))
        if row_index < 1 or row_index > len(dataset_rows):
            warnings.append(f"row_index out of range: {row_index}")
            continue
        row = dataset_rows[row_index - 1]
        if record.get("source") != row.get("source"):
            warnings.append(
                f"source mismatch at row_index={row_index}: "
                f"record={record.get('source')} dataset={row.get('source')}"
            )
        if record.get("input") != row.get("input"):
            warnings.append(f"input mismatch at row_index={row_index}")
    return warnings[:20]


def summarize(
    *,
    dataset_path: Path,
    per_row_path: Path,
    mode: str,
) -> dict[str, Any]:
    dataset_rows = load_jsonl(dataset_path)
    records = [record for record in load_jsonl(per_row_path) if record.get("mode") == mode]
    if not records:
        raise ValueError(f"No per-row records found for mode={mode!r} in {per_row_path}")

    model_order: list[str] = []
    buckets: dict[str, OrderedDict[str, dict[str, Any]]] = {
        segment: OrderedDict() for segment in SEGMENT_ORDER
    }
    for record in records:
        model_id = str(record["model_id"])
        if model_id not in model_order:
            model_order.append(model_id)
        for segment in segment_names(record.get("source")):
            segment_bucket = buckets[segment]
            if model_id not in segment_bucket:
                segment_bucket[model_id] = empty_bucket(record)
            update_bucket(segment_bucket[model_id], record)

    segments = {
        segment: {
            "models": [finalize_bucket(buckets[segment][model_id]) for model_id in model_order if model_id in buckets[segment]],
        }
        for segment in SEGMENT_ORDER
    }
    return {
        "dataset": str(dataset_path),
        "dataset_rows": len(dataset_rows),
        "per_row": str(per_row_path),
        "mode": mode,
        "records": len(records),
        "validation_warnings": validate_records(dataset_rows, records),
        "segment_definitions": {
            "rule_based_1000": f"source != {NATURAL_SOURCE!r}",
            "natural_1000": f"source == {NATURAL_SOURCE!r}",
            "all_2000": "all rows",
        },
        "segments": segments,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SentiLight Benchmark By Segment",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Per-row predictions: `{report['per_row']}`",
        f"- Mode: `{report['mode']}`",
        f"- Records used: `{report['records']}`",
        "",
    ]
    if report["validation_warnings"]:
        lines.append("## Validation Warnings")
        lines.append("")
        for warning in report["validation_warnings"]:
            lines.append(f"- {warning}")
        lines.append("")

    for segment in SEGMENT_ORDER:
        lines.append(f"## {segment}")
        lines.append("")
        lines.append(
            "| Model ID | N | Success | Strict | Coarse | H Match | S Match | B Match | CT Match | Dimmer Match | Mean Latency | P95 Latency | Hue MAE | S MAE | B MAE | Dimmer MAE | CT MAE |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for model in report["segments"][segment]["models"]:
            latency = model["latency"]
            mae = model["mae"]
            coarse = model["coarse_match_rates"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        model["model_id"],
                        str(model["n"]),
                        f"{model['successes']} ({model['success_rate']:.2%})",
                        f"{model['strict_passes']} ({model['strict_rate']:.2%})",
                        f"{model['coarse_passes']} ({model['coarse_rate']:.2%})",
                        f"{coarse['H']:.2%}",
                        f"{coarse['S']:.2%}",
                        f"{coarse['B']:.2%}",
                        f"{coarse['CT']:.2%}",
                        f"{coarse['Dimmer']:.2%}",
                        str(latency["mean_ms"]),
                        str(latency["p95_ms"]),
                        str(mae["H"]),
                        str(mae["S"]),
                        str(mae["B"]),
                        str(mae["Dimmer"]),
                        str(mae["CT"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute SentiLight benchmark metrics by dataset source segment")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--per-row", type=Path, required=True)
    parser.add_argument("--mode", default="predict_all")
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = summarize(dataset_path=args.dataset, per_row_path=args.per_row, mode=args.mode)
    markdown = render_markdown(report)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
