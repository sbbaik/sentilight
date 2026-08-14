from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_REPORT_ROOT = Path("backend/reports/scale_grid")
RUN_RE = re.compile(r"^compactlm_(?P<scale>[^_]+)_f(?P<fraction>\d+)_s(?P<seed>\d+)$")


def load_reports(report_root: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in sorted((report_root / "run_summaries").glob("compactlm_*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        match = RUN_RE.match(str(report.get("model_id", "")))
        if not match:
            continue
        report["scale_grid"] = {
            "scale": match.group("scale"),
            "fraction": int(match.group("fraction")),
            "seed": int(match.group("seed")),
            "path": str(path),
        }
        reports.append(report)
    return reports


def mean_std(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": statistics.fmean(values), "std": statistics.stdev(values)}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return (center - half, center + half)


def summarize(reports: list[dict[str, Any]], expected_seeds: set[int]) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        grid = report["scale_grid"]
        groups[(grid["scale"], grid["fraction"])].append(report)

    summaries: list[dict[str, Any]] = []
    for (scale, fraction), items in sorted(groups.items(), key=lambda pair: (pair[0][1], pair[0][0])):
        seeds = {item["scale_grid"]["seed"] for item in items}
        strict_rates = [item["metrics"]["strict_semantic_pass_rate"] for item in items]
        coarse_rates = [item["metrics"]["coarse_3bin_pass_rate"] for item in items]
        latencies = [
            item["metrics"]["latency"]["mean_ms"]
            for item in items
            if item["metrics"]["latency"]["mean_ms"] is not None
        ]
        strict_counts = sum(item["metrics"]["strict_semantic_passes"] for item in items)
        coarse_counts = sum(item["metrics"]["coarse_3bin_passes"] for item in items)
        total_rows = sum(item["rows"] for item in items)
        strict_ci = wilson_interval(strict_counts, total_rows)
        coarse_ci = wilson_interval(coarse_counts, total_rows)
        summaries.append(
            {
                "scale": scale,
                "fraction": fraction,
                "seeds": sorted(seeds),
                "missing_seeds": sorted(expected_seeds - seeds),
                "runs": len(items),
                "parameter_count": items[0].get("parameter_count"),
                "rows_total": total_rows,
                "strict": {
                    **mean_std(strict_rates),
                    "pooled_successes": strict_counts,
                    "pooled_ci95": strict_ci,
                },
                "coarse": {
                    **mean_std(coarse_rates),
                    "pooled_successes": coarse_counts,
                    "pooled_ci95": coarse_ci,
                },
                "mean_latency_ms": mean_std(latencies),
            }
        )
    return {
        "runs_loaded": len(reports),
        "expected_seeds": sorted(expected_seeds),
        "summaries": summaries,
        "prediction_1_decision_rule": {
            "support": "8M->23M improvement < 3.5%p and 23M->60M improvement < 3.5%p",
            "reject_or_reduce": "23M->60M improvement >= 3.5%p with statistical support",
            "lower_plateau": "3M->8M improves substantially but later increments are < 3.5%p",
        },
    }


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}"


def pct_ci(ci: tuple[float, float] | list[float]) -> str:
    return f"[{ci[0] * 100:.2f}, {ci[1] * 100:.2f}]"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CompactLM Scale/Data Grid Summary",
        "",
        f"- Runs loaded: `{report['runs_loaded']}`",
        f"- Expected seeds: `{', '.join(str(seed) for seed in report['expected_seeds'])}`",
        "",
        "| Scale | Data % | Runs | Missing Seeds | Params | Strict mean±std | Strict pooled 95% CI | Coarse mean±std | Coarse pooled 95% CI | Mean latency ms |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summaries"]:
        missing = ",".join(str(seed) for seed in row["missing_seeds"]) or "none"
        strict_std = row["strict"]["std"]
        coarse_std = row["coarse"]["std"]
        latency = row["mean_latency_ms"]["mean"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["scale"],
                    str(row["fraction"]),
                    str(row["runs"]),
                    missing,
                    str(row["parameter_count"]),
                    f"{pct(row['strict']['mean'])}±{pct(strict_std)}",
                    pct_ci(row["strict"]["pooled_ci95"]),
                    f"{pct(row['coarse']['mean'])}±{pct(coarse_std)}",
                    pct_ci(row["coarse"]["pooled_ci95"]),
                    f"{latency:.2f}" if latency is not None else "n/a",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Pre-registered interpretation rule:",
            "",
            "- Support prediction 1 if 8M->23M and 23M->60M improvements are both below 3.5%p.",
            "- Reject or reduce prediction 1 if 23M->60M improves by at least 3.5%p with statistical support.",
            "- Move the estimated plateau downward if only 3M->8M improves substantially.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CompactLM scale/data grid reports")
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_ROOT / "scale_grid_summary.json")
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_ROOT / "scale_grid_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = load_reports(args.report_root)
    report = summarize(reports, set(args.seeds))
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


if __name__ == "__main__":
    main()
