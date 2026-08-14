from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
MODELS_DIR = BACKEND_DIR / "models"
for path in (BACKEND_DIR, MODELS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compact_llm.tuple_classifier import build_tuple_vocab  # noqa: E402


FIELDS = ("H", "S", "B", "Dimmer", "CT")
MODEL_RE = re.compile(r"compactlm_(?P<scale>\d+M)_f(?P<data>\d+)_s(?P<seed>\d+)")


DEFAULT_TRAIN_JSONL = Path("backend/models/compact_llm/datasets/scale_grid_subsets/train_100.jsonl")
DEFAULT_PER_ROW_DIR = Path("backend/reports/scale_grid/per_row")
DEFAULT_OUTPUT_JSON = Path("backend/reports/tuple_frequency/tuple_frequency_summary.json")
DEFAULT_OUTPUT_MD = Path("backend/reports/tuple_frequency/tuple_frequency_summary.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze accuracy by reference tuple training frequency")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--per-row-dir", type=Path, default=DEFAULT_PER_ROW_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def output_tuple(output: dict[str, Any] | None) -> tuple[int, int, int, int, int] | None:
    if not isinstance(output, dict):
        return None
    try:
        return tuple(int(output[field]) for field in FIELDS)
    except Exception:  # noqa: BLE001 - malformed per-row output is counted as OOV
        return None


def load_train_frequency(
    path: Path,
    policy_vocab: set[tuple[int, int, int, int, int]],
) -> tuple[Counter[tuple[int, int, int, int, int]], int]:
    counter: Counter[tuple[int, int, int, int, int]] = Counter()
    dropped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = output_tuple(row.get("output"))
            if key is not None and key in policy_vocab:
                counter[key] += 1
            else:
                dropped += 1
    return counter, dropped


def rank_training_tuples(
    counter: Counter[tuple[int, int, int, int, int]],
    policy_vocab: set[tuple[int, int, int, int, int]],
) -> dict[tuple[int, int, int, int, int], int]:
    ranked = sorted(((key, counter.get(key, 0)) for key in policy_vocab), key=lambda item: (-item[1], item[0]))
    return {key: rank for rank, (key, _) in enumerate(ranked, start=1)}


def bucket_for(reference: tuple[int, int, int, int, int] | None, ranks: dict[tuple[int, int, int, int, int], int]) -> str:
    if reference is None or reference not in ranks:
        return "unseen"
    rank = ranks[reference]
    if rank <= 50:
        return "top-50"
    if rank <= 120:
        return "rank-51-120"
    if rank <= 240:
        return "rank-121-240"
    return "rank-241+"


def parse_model_id(model_id: str) -> dict[str, Any]:
    match = MODEL_RE.fullmatch(model_id)
    if not match:
        return {"scale": "unknown", "data_pct": -1, "seed": -1}
    return {
        "scale": match.group("scale"),
        "data_pct": int(match.group("data")),
        "seed": int(match.group("seed")),
    }


def safe_rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def summarize_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": statistics.fmean(values), "std": statistics.stdev(values)}


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.2f}%"


def pct_pm(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "n/a"
    return f"{100 * mean:.2f}%±{100 * (std or 0.0):.2f}%"


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    policy_vocab = {output_tuple(output) for output in build_tuple_vocab()}
    policy_vocab.discard(None)
    train_freq, train_dropped = load_train_frequency(args.train_jsonl, policy_vocab)
    ranks = rank_training_tuples(train_freq, policy_vocab)
    runs: list[dict[str, Any]] = []
    grouped_bucket_rates: dict[tuple[str, int, str], dict[str, list[float]]] = defaultdict(lambda: {"strict": [], "coarse": [], "n": []})

    for path in sorted(args.per_row_dir.glob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            continue
        model_id = str(records[0].get("model_id") or path.stem)
        parsed = parse_model_id(model_id)
        bucket_counts: dict[str, Counter[str]] = defaultdict(Counter)
        predicted_unique: set[tuple[int, int, int, int, int]] = set()
        predicted_in_top240 = 0
        successes = 0
        for record in records:
            reference_key = output_tuple(record.get("reference"))
            predicted_key = output_tuple(record.get("predicted"))
            bucket = bucket_for(reference_key, ranks)
            bucket_counts[bucket]["n"] += 1
            bucket_counts[bucket]["strict"] += int(bool(record.get("strict")))
            bucket_counts[bucket]["coarse"] += int(bool(record.get("coarse")))
            successes += int(bool(record.get("success")))
            if predicted_key is not None:
                predicted_unique.add(predicted_key)
                predicted_in_top240 += int(predicted_key in policy_vocab)
        bucket_summary: dict[str, dict[str, Any]] = {}
        for bucket, counter in sorted(bucket_counts.items()):
            n = int(counter["n"])
            bucket_summary[bucket] = {
                "n": n,
                "strict_passes": int(counter["strict"]),
                "strict_rate": safe_rate(counter["strict"], n),
                "coarse_passes": int(counter["coarse"]),
                "coarse_rate": safe_rate(counter["coarse"], n),
            }
            group_key = (parsed["scale"], parsed["data_pct"], bucket)
            grouped_bucket_rates[group_key]["strict"].append(bucket_summary[bucket]["strict_rate"])
            grouped_bucket_rates[group_key]["coarse"].append(bucket_summary[bucket]["coarse_rate"])
            grouped_bucket_rates[group_key]["n"].append(float(n))
        runs.append(
            {
                "model_id": model_id,
                **parsed,
                "rows": len(records),
                "successes": successes,
                "predicted_unique_tuples": len(predicted_unique),
                "predicted_in_policy_vocab": predicted_in_top240,
                "predicted_in_policy_vocab_rate": safe_rate(predicted_in_top240, len(records)),
                "buckets": bucket_summary,
            }
        )

    grouped: list[dict[str, Any]] = []
    for (scale, data_pct, bucket), values in sorted(grouped_bucket_rates.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        strict = summarize_values(values["strict"])
        coarse = summarize_values(values["coarse"])
        grouped.append(
            {
                "scale": scale,
                "data_pct": data_pct,
                "bucket": bucket,
                "runs": len(values["strict"]),
                "mean_n": statistics.fmean(values["n"]) if values["n"] else 0.0,
                "strict": strict,
                "coarse": coarse,
            }
        )

    return {
        "train_jsonl": str(args.train_jsonl),
        "per_row_dir": str(args.per_row_dir),
        "policy_vocab_size": len(policy_vocab),
        "train_policy_unique_tuples": len(train_freq),
        "train_policy_rows": sum(train_freq.values()),
        "train_non_policy_rows": train_dropped,
        "runs": runs,
        "grouped": grouped,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Tuple Frequency Accuracy Analysis",
        "",
        f"- Train JSONL: `{report['train_jsonl']}`",
        f"- Per-row dir: `{report['per_row_dir']}`",
        f"- Policy vocab size: `{report['policy_vocab_size']}`",
        f"- Train policy unique tuples: `{report['train_policy_unique_tuples']}`",
        f"- Train policy rows: `{report['train_policy_rows']}`",
        f"- Train non-policy rows: `{report['train_non_policy_rows']}`",
        "",
        "## 23M/60M 100% Focus",
        "",
        "| Scale | Data % | Bucket | Runs | Mean n | Strict mean±std | Coarse mean±std |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    focus = [
        row
        for row in report["grouped"]
        if row["scale"] in {"23M", "60M"} and row["data_pct"] == 100
    ]
    bucket_order = {"top-50": 0, "rank-51-120": 1, "rank-121-240": 2, "rank-241+": 3, "unseen": 4}
    for row in sorted(focus, key=lambda item: (item["scale"], bucket_order.get(item["bucket"], 99))):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["scale"],
                    str(row["data_pct"]),
                    row["bucket"],
                    str(row["runs"]),
                    f"{row['mean_n']:.1f}",
                    pct_pm(row["strict"]["mean"], row["strict"]["std"]),
                    pct_pm(row["coarse"]["mean"], row["coarse"]["std"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Predicted Tuple Diversity",
            "",
            "| Model | Rows | Predicted unique tuples | In policy vocab |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in sorted(report["runs"], key=lambda item: (item["scale"], item["data_pct"], item["seed"])):
        if row["scale"] in {"23M", "60M"} and row["data_pct"] == 100:
            lines.append(
                f"| {row['model_id']} | {row['rows']} | {row['predicted_unique_tuples']} | {pct(row['predicted_in_policy_vocab_rate'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = analyze(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report), end="")


if __name__ == "__main__":
    main()
