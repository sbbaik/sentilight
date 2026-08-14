from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_PER_ROW = Path("backend/reports/five_model_benchmark_v3_per_row.jsonl")
DEFAULT_REPORT_JSON = Path("results/mcnemar_quant.json")
DEFAULT_REPORT_MD = Path("results/mcnemar_quant.md")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def paired_diff_ci(b: int, c: int, total: int, z: float) -> tuple[float, float, float]:
    if total <= 0:
        return 0.0, 0.0, 0.0
    diff = (c - b) / total
    variance = ((b + c) - ((c - b) ** 2 / total)) / (total**2)
    se = math.sqrt(max(0.0, variance))
    return diff, diff - z * se, diff + z * se


def summarize(
    records: list[dict[str, Any]],
    model_a: str,
    model_b: str,
    segment: str,
    mode: str,
    margin: float,
) -> dict[str, Any]:
    if segment == "rule":
        selected = [
            record
            for record in records
            if record.get("source") != "natural_language_baseline"
            and record.get("mode") == mode
        ]
    elif segment == "natural":
        selected = [
            record
            for record in records
            if record.get("source") == "natural_language_baseline"
            and record.get("mode") == mode
        ]
    elif segment == "all":
        selected = [record for record in records if record.get("mode") == mode]
    else:
        raise ValueError(f"unsupported segment: {segment}")

    by_row: dict[int, dict[str, dict[str, Any]]] = {}
    for record in selected:
        by_row.setdefault(int(record["row_index"]), {})[str(record["model_id"])] = record

    metrics: dict[str, Any] = {}
    for metric in ("strict", "coarse"):
        b = c = both = neither = missing = 0
        for row_models in by_row.values():
            if model_a not in row_models or model_b not in row_models:
                missing += 1
                continue
            a_ok = bool(row_models[model_a].get(metric))
            b_ok = bool(row_models[model_b].get(metric))
            if a_ok and not b_ok:
                b += 1
            elif b_ok and not a_ok:
                c += 1
            elif a_ok and b_ok:
                both += 1
            else:
                neither += 1

        discordant = b + c
        chi2 = ((abs(b - c) - 1) ** 2 / discordant) if discordant else 0.0
        asymptotic_p = math.erfc(math.sqrt(chi2 / 2.0)) if discordant else 1.0
        exact_p = exact_mcnemar_p(b, c)
        diff, ci90_low, ci90_high = paired_diff_ci(b, c, len(by_row), 1.6448536269514722)
        metrics[metric] = {
            "model_a": model_a,
            "model_b": model_b,
            "b_a_only": b,
            "c_b_only": c,
            "discordant": discordant,
            "both_correct": both,
            "both_wrong": neither,
            "missing_pairs": missing,
            "paired_difference": diff,
            "paired_difference_pct": diff * 100.0,
            "mcnemar_chi2_continuity": chi2,
            "mcnemar_p_continuity": asymptotic_p,
            "mcnemar_exact_p": exact_p,
            "ci90_low": ci90_low,
            "ci90_high": ci90_high,
            "ci90_low_pct": ci90_low * 100.0,
            "ci90_high_pct": ci90_high * 100.0,
            "equivalence_margin": margin,
            "equivalence_margin_pct": margin * 100.0,
            "tost_equivalent": ci90_low >= -margin and ci90_high <= margin,
        }

    return {
        "input": {
            "records": len(records),
            "mode": mode,
            "segment": segment,
            "paired_rows": len(by_row),
            "model_a": model_a,
            "model_b": model_b,
        },
        "metrics": metrics,
    }


def render_markdown(report: dict[str, Any]) -> str:
    info = report["input"]
    lines = [
        "# McNemar and TOST for CompactLM Quantization",
        "",
        f"- Segment: `{info['segment']}`",
        f"- Mode: `{info['mode']}`",
        f"- Paired rows: `{info['paired_rows']}`",
        f"- Comparison: `{info['model_a']}` vs `{info['model_b']}`",
        "",
        "| Metric | b: FP32 only | c: Q4 only | Discordant | McNemar p | Exact p | Diff | 90% CI | TOST ±3%p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for metric, row in report["metrics"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    metric,
                    str(row["b_a_only"]),
                    str(row["c_b_only"]),
                    str(row["discordant"]),
                    f"{row['mcnemar_p_continuity']:.6f}",
                    f"{row['mcnemar_exact_p']:.6f}",
                    f"{row['paired_difference_pct']:.2f}%p",
                    f"[{row['ci90_low_pct']:.2f}, {row['ci90_high_pct']:.2f}]%p",
                    "pass" if row["tost_equivalent"] else "fail",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- McNemar tests whether the paired success/failure pattern differs between FP32 and Q4.",
            "- TOST passes when the paired 90% confidence interval is fully inside the ±3%p equivalence margin.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run McNemar and TOST for CompactLM FP32 vs Q4 per-row benchmark records")
    parser.add_argument("--per-row", type=Path, default=DEFAULT_PER_ROW)
    parser.add_argument("--model-a", default="compact_llm")
    parser.add_argument("--model-b", default="compact_llm_q4")
    parser.add_argument("--segment", choices=("rule", "natural", "all"), default="rule")
    parser.add_argument("--mode", default="predict_all")
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    return parser.parse_args()


def main() -> None:
    Path("results").mkdir(parents=True, exist_ok=True)
    args = parse_args()
    records = load_jsonl(args.per_row)
    report = summarize(
        records=records,
        model_a=args.model_a,
        model_b=args.model_b,
        segment=args.segment,
        mode=args.mode,
        margin=args.margin,
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report), end="")


if __name__ == "__main__":
    main()
