"""Paired pre/post-contamination comparison for the preregistered delta study.

Compares dirty vs clean scale grids cell-by-cell (scale, fraction, seed) on the
frozen rule/natural segments and adjudicates preregistered predictions P1-P3
from results/prereg_contamination_delta.md. P4 (C1/C2 contrasts) is
adjudicated narratively in the report, not here.

Inputs (run_summaries/compactlm_*.json produced by evaluate_compactlm_checkpoint.py):
  backend/reports/scale_grid                -> dirty backbone, rule segment
  backend/reports/scale_grid_clean          -> clean backbone, rule segment
  backend/reports/scale_grid_natural        -> dirty backbone, natural segment
  backend/reports/scale_grid_clean_natural  -> clean backbone, natural segment

Output: JSON to stdout and optional --out path.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_RE = re.compile(r"^compactlm_(?P<scale>[^_]+)_f(?P<fraction>\d+)_s(?P<seed>\d+)$")

SOURCES = {
    "rule": ("backend/reports/scale_grid", "backend/reports/scale_grid_clean"),
    "natural": ("backend/reports/scale_grid_natural", "backend/reports/scale_grid_clean_natural"),
}
SCALE_ORDER = {"3M": 0, "8M": 1, "23M": 2, "60M": 3}


def load_cells(report_root: Path) -> dict[tuple[str, int, int], dict[str, float]]:
    cells: dict[tuple[str, int, int], dict[str, float]] = {}
    for path in sorted((report_root / "run_summaries").glob("compactlm_*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        match = RUN_RE.match(str(report.get("model_id", "")))
        if not match:
            continue
        metrics = report["metrics"]
        key = (match.group("scale"), int(match.group("fraction")), int(match.group("seed")))
        cells[key] = {
            "strict": metrics["strict_semantic_pass_rate"],
            "coarse": metrics["coarse_3bin_pass_rate"],
            "rows": report.get("rows"),
        }
    return cells


def mean_std(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def compare_segment(segment: str) -> dict[str, Any]:
    dirty_root, clean_root = (REPO_ROOT / p for p in SOURCES[segment])
    dirty = load_cells(dirty_root)
    clean = load_cells(clean_root)
    shared = sorted(set(dirty) & set(clean), key=lambda k: (SCALE_ORDER.get(k[0], 9), k[1], k[2]))
    cells = []
    for key in shared:
        scale, fraction, seed = key
        entry = {"scale": scale, "fraction": fraction, "seed": seed}
        for metric in ("strict", "coarse"):
            pre = dirty[key][metric]
            post = clean[key][metric]
            entry[metric] = {
                "dirty": pre,
                "clean": post,
                "delta_pp": (post - pre) * 100.0,
            }
        cells.append(entry)
    summary: dict[str, Any] = {
        "segment": segment,
        "cells_paired": len(shared),
        "cells_missing_dirty": sorted(str(k) for k in set(clean) - set(dirty)),
        "cells_missing_clean": sorted(str(k) for k in set(dirty) - set(clean)),
        "cells": cells,
    }
    for metric in ("strict", "coarse"):
        deltas = [c[metric]["delta_pp"] for c in cells]
        abs_deltas = [abs(d) for d in deltas]
        summary[f"{metric}_delta_pp"] = {
            **mean_std(deltas),
            "min": min(deltas) if deltas else None,
            "max": max(deltas) if deltas else None,
            "mean_abs": statistics.fmean(abs_deltas) if abs_deltas else None,
            "cells_beyond_3pp": [
                f"{c['scale']}_f{c['fraction']}_s{c['seed']}:{c[metric]['delta_pp']:+.1f}pp"
                for c in cells if abs(c[metric]["delta_pp"]) > 3.0
            ],
            "cells_below_minus5pp": [
                f"{c['scale']}_f{c['fraction']}_s{c['seed']}:{c[metric]['delta_pp']:+.1f}pp"
                for c in cells if c[metric]["delta_pp"] < -5.0
            ],
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result: dict[str, Any] = {"segments": {}}
    for segment in ("rule", "natural"):
        try:
            result["segments"][segment] = compare_segment(segment)
        except FileNotFoundError as exc:
            result["segments"][segment] = {"error": f"missing report root: {exc}"}

    rule = result["segments"].get("rule", {})
    natural = result["segments"].get("natural", {})
    verdicts: dict[str, Any] = {}
    if rule.get("cells"):
        beyond = rule["strict_delta_pp"]["cells_beyond_3pp"] + rule["coarse_delta_pp"]["cells_beyond_3pp"]
        verdicts["P1_rule_within_3pp"] = {
            "holds_per_cell": not beyond,
            "violating_cells": beyond,
            "strict_mean_delta_pp": rule["strict_delta_pp"]["mean"],
            "coarse_mean_delta_pp": rule["coarse_delta_pp"]["mean"],
        }
    if natural.get("cells"):
        below = natural["strict_delta_pp"]["cells_below_minus5pp"] + natural["coarse_delta_pp"]["cells_below_minus5pp"]
        verdicts["P2_natural_drop_within_5pp"] = {
            "holds_per_cell": not below,
            "violating_cells": below,
            "strict_mean_delta_pp": natural["strict_delta_pp"]["mean"],
            "coarse_mean_delta_pp": natural["coarse_delta_pp"]["mean"],
        }
    if rule.get("cells") and natural.get("cells"):
        verdicts["P3_abs_natural_gt_abs_rule"] = {
            metric: {
                "natural_mean_abs_pp": natural[f"{metric}_delta_pp"]["mean_abs"],
                "rule_mean_abs_pp": rule[f"{metric}_delta_pp"]["mean_abs"],
                "holds": natural[f"{metric}_delta_pp"]["mean_abs"] > rule[f"{metric}_delta_pp"]["mean_abs"],
            }
            for metric in ("strict", "coarse")
        }
    result["prereg_verdicts"] = verdicts

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
