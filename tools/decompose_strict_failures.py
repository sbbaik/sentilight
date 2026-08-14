"""Which dimensions actually cause strict failures, and is the H check binding?

Strict's hue tolerance (20 degrees) is looser than the policy's own hue-anchor
snap (18 degrees), so any policy-conformant output should clear the H check
automatically. That is an argument from the constants; this measures it.

For every model family x seed on the clean backbone, over the 1000-row rule
segment, this reports:
  - how often each individual strict clause is violated
  - how often a clause is the SOLE reason a row fails (its marginal contribution)
  - strict recomputed with the H clause removed, to size H's true bite
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO / "backend" / "models"))

from common.benchmark_eval import hue_distance  # noqa: E402

SEEDS = [42, 43, 44]
FAMILIES = {
    "A1_gen23": "backend/reports/scale_grid_clean/per_row/compactlm_23M_f100_s{seed}.jsonl",
    "gen60": "backend/reports/scale_grid_clean/per_row/compactlm_60M_f100_s{seed}.jsonl",
    "A2_policy23": "backend/reports/a2_policy_only_clean/per_row/a2_policy_lm_23M_s{seed}.jsonl",
    "B2_tuple23": "backend/reports/tuple_head_clean/per_row/tuple_head_23M_f100_s{seed}.jsonl",
    "B2_tuple60": "backend/reports/tuple_head_60M_clean/per_row/tuple_head_60M_f100_s{seed}.jsonl",
}

# Each clause mirrors benchmark_eval.semantic_pass exactly.
TOL_CLAUSES = ("H", "S", "B", "Dimmer", "CT")


def clause_violations(exp: dict, pred: dict, row: dict) -> set[str]:
    """Every clause this row violates (not just the first)."""
    v = set()
    if hue_distance(pred["H"], exp["H"]) > 20:
        v.add("H")
    if abs(pred["S"] - exp["S"]) > 20:
        v.add("S")
    if abs(pred["B"] - exp["B"]) > 20:
        v.add("B")
    if abs(pred["Dimmer"] - exp["Dimmer"]) > 20:
        v.add("Dimmer")
    if abs(pred["CT"] - exp["CT"]) > 80:
        v.add("CT")
    emotion, base_color = row.get("emotion"), row.get("base_color")
    if emotion == "sadness" and not (pred["B"] <= 40 and pred["CT"] >= 340):
        v.add("emotion:sadness")
    if emotion == "happiness" and not (pred["B"] >= 70 and pred["CT"] <= 290):
        v.add("emotion:happiness")
    if emotion == "anger" and not (pred["S"] >= 75 and pred["B"] >= 60):
        v.add("emotion:anger")
    if emotion == "calm" and not (30 <= pred["S"] <= 70 and 45 <= pred["B"] <= 80):
        v.add("emotion:calm")
    if emotion == "anxiety" and not (pred["B"] <= 45 and pred["CT"] >= 340):
        v.add("emotion:anxiety")
    if base_color is None and emotion is None and not (15 <= pred["S"] <= 45 and 40 <= pred["B"] <= 70):
        v.add("neutral_fallback")
    return v


def main() -> None:
    result: dict = {}
    for fam, pat in FAMILIES.items():
        per_seed = []
        for seed in SEEDS:
            viol = Counter()
            sole = Counter()
            n = passes = passes_no_H = 0
            h_dist_when_failing_H = []
            for line in (REPO / pat.format(seed=seed)).open(encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("success"):
                    n += 1
                    continue
                n += 1
                v = clause_violations(r["reference"], r["predicted"], r)
                if not v:
                    passes += 1
                if not (v - {"H"}):
                    passes_no_H += 1
                for c in v:
                    viol[c] += 1
                if len(v) == 1:
                    sole[next(iter(v))] += 1
                if "H" in v:
                    h_dist_when_failing_H.append(hue_distance(r["predicted"]["H"], r["reference"]["H"]))
            per_seed.append({
                "seed": seed, "rows": n,
                "strict_pct": passes / n * 100.0,
                "strict_without_H_pct": passes_no_H / n * 100.0,
                "violations_pct": {k: viol[k] / n * 100.0 for k in sorted(viol)},
                "sole_cause_pct": {k: sole[k] / n * 100.0 for k in sorted(sole)},
                "median_hue_error_when_H_fails": (
                    st.median(h_dist_when_failing_H) if h_dist_when_failing_H else None),
            })
        keys = sorted({k for s in per_seed for k in s["violations_pct"]})
        result[fam] = {
            "strict_pct": st.fmean(s["strict_pct"] for s in per_seed),
            "strict_without_H_pct": st.fmean(s["strict_without_H_pct"] for s in per_seed),
            "violations_pct": {k: st.fmean(s["violations_pct"].get(k, 0.0) for s in per_seed) for k in keys},
            "sole_cause_pct": {k: st.fmean(s["sole_cause_pct"].get(k, 0.0) for s in per_seed) for k in keys},
            "per_seed": per_seed,
        }

    (REPO / "results").mkdir(parents=True, exist_ok=True)

    (REPO / "results/strict_failure_decomposition.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("mean over seeds 42/43/44, rule segment n=1000, clean backbone\n")
    print(f"{'family':13s} {'strict':>8s} {'no-H':>8s} {'gain':>7s}")
    for fam, r in result.items():
        print(f"{fam:13s} {r['strict_pct']:7.2f}% {r['strict_without_H_pct']:7.2f}% "
              f"{r['strict_without_H_pct']-r['strict_pct']:+6.2f}pp")
    print("\nclause violation rate (% of rows), and % of rows where it is the SOLE cause:")
    allk = sorted({k for r in result.values() for k in r["violations_pct"]})
    print(f"{'clause':22s}" + "".join(f"{f:>14s}" for f in result))
    for k in allk:
        row = f"{k:22s}"
        for fam, r in result.items():
            row += f"{r['violations_pct'].get(k,0):7.1f}/{r['sole_cause_pct'].get(k,0):<6.1f}"
        print(row)


if __name__ == "__main__":
    main()
