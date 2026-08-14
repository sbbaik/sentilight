"""P4 adjudication: substitute clean-backbone generative numbers into C1/C2.

C1 (representation > scale): tuple head 23M (dirty backbone, unchanged) vs
generative 60M f100, with the generative side swapped dirty -> clean.
C2 (2x2 decomposition A1/A2/B2): A1 = generative 23M f100 swapped dirty ->
clean; A2 (policy-only LM) and B2 (tuple head) unchanged (not part of the
preregistered intervention).

Seed-level paired t mirrors backend/reports/family_summary_original_backbone.json methodology.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLAIMS = REPO_ROOT / "backend/reports/family_summary_original_backbone.json"
SEEDS = [42, 43, 44]

# two-sided p for df=2: CDF(t) = 1/2 + t / (2*sqrt(2+t^2))
def t_p_two(t: float, df: int) -> float:
    if df != 2:
        raise NotImplementedError
    return 1.0 - abs(t) / math.sqrt(2.0 + t * t)


def paired_t(baseline: list[float], treatment: list[float]) -> dict:
    diffs = [t - b for b, t in zip(baseline, treatment)]
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    t = mean / (sd / math.sqrt(len(diffs))) if sd > 0 else float("inf")
    return {
        "diffs": diffs,
        "mean_diff": mean,
        "sd_diff": sd,
        "t": t,
        "p_two": t_p_two(abs(t), len(diffs) - 1),
    }


def grid_seed_rates(report_root: str, scale: str, metric: str) -> list[float]:
    key = {"strict": "strict_semantic_pass_rate", "coarse": "coarse_3bin_pass_rate"}[metric]
    out = []
    for seed in SEEDS:
        path = REPO_ROOT / report_root / "run_summaries" / f"compactlm_{scale}_f100_s{seed}.json"
        out.append(json.loads(path.read_text())["metrics"][key])
    return out


def family_seed_rates(claims: dict, name: str, metric: str) -> list[float]:
    fam = next(f for f in claims["families"] if f["name"] == name)
    rows = sorted(fam["rows"], key=lambda r: r["seed"])
    return [r[metric] for r in rows]


def main() -> None:
    claims = json.loads(CLAIMS.read_text())
    result = {}
    for metric in ("strict", "coarse"):
        tuple23 = family_seed_rates(claims, "23M tuple head", metric)
        policy23 = family_seed_rates(claims, "23M policy-only LM", metric)
        gen60_dirty = grid_seed_rates("backend/reports/scale_grid", "60M", metric)
        gen60_clean = grid_seed_rates("backend/reports/scale_grid_clean", "60M", metric)
        gen23_dirty = grid_seed_rates("backend/reports/scale_grid", "23M", metric)
        gen23_clean = grid_seed_rates("backend/reports/scale_grid_clean", "23M", metric)
        result[metric] = {
            "C1_tuple23_vs_gen60": {
                "dirty_backbone": paired_t(gen60_dirty, tuple23),
                "clean_backbone": paired_t(gen60_clean, tuple23),
                "gen60_means": {"dirty": statistics.fmean(gen60_dirty), "clean": statistics.fmean(gen60_clean)},
            },
            "C2_data_effect_A2_minus_A1": {
                "dirty_backbone": paired_t(gen23_dirty, policy23),
                "clean_backbone": paired_t(gen23_clean, policy23),
            },
            "C2_representation_effect_B2_minus_A2": paired_t(policy23, tuple23),
            "C2_tuple23_vs_gen23_A1_to_B2": {
                "dirty_backbone": paired_t(gen23_dirty, tuple23),
                "clean_backbone": paired_t(gen23_clean, tuple23),
            },
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
