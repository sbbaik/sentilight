"""Post-hoc robustness check: C1/C2 under three backbone conditions.

NOT a preregistered analysis. The preregistered contrast check was adjudicated on
a MIXED backbone: the generative side (A1, gen60) was swapped dirty -> clean while A2
(policy-only SFT) and B2 (tuple head) stayed on the contaminated backbone,
because the frozen prereg scoped the intervention to the scale grid only. This
script adds the fully-clean condition so the mixed-backbone limitation can be
quantified rather than merely disclosed.

Conditions:
  dirty       every family on the contaminated backbone (original paper numbers)
  mixed       generative side clean, A2/B2 dirty (what P4 actually reported)
  fully_clean every family on the clean backbone

Sources (all rule segment, 1000 rows, seeds 42/43/44):
  dirty  gen        backend/reports/scale_grid/run_summaries/compactlm_{scale}_f100_s{seed}.json
  clean  gen        backend/reports/scale_grid_clean/...
  dirty  A2         backend/reports/family_summary_original_backbone.json family "23M policy-only LM"
  clean  A2         backend/reports/a2_policy_only_clean/run_summaries/a2_policy_lm_23M_s{seed}.json
  dirty  B2 23M/60M backend/reports/family_summary_original_backbone.json families "{scale} tuple head"
  clean  B2 23M     backend/reports/tuple_head_clean/run_summaries/tuple_head_23M_f100_s{seed}.json
  clean  B2 60M     backend/reports/tuple_head_60M_clean/run_summaries/tuple_head_60M_f100_s{seed}.json

Seed-level paired t (df=2) mirrors backend/reports/family_summary_original_backbone.json.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLAIMS = REPO_ROOT / "backend/reports/family_summary_original_backbone.json"
SEEDS = [42, 43, 44]
METRIC_KEY = {"strict": "strict_semantic_pass_rate", "coarse": "coarse_3bin_pass_rate"}


def t_p_two(t: float, df: int) -> float:
    """Two-sided p for df=2, where CDF(t) = 1/2 + t / (2*sqrt(2+t^2))."""
    if df != 2:
        raise NotImplementedError
    return 1.0 - abs(t) / math.sqrt(2.0 + t * t)


def paired_t(baseline: list[float], treatment: list[float]) -> dict:
    diffs = [t - b for b, t in zip(baseline, treatment)]
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    t = mean / (sd / math.sqrt(len(diffs))) if sd > 0 else float("inf")
    return {
        "mean_diff_pp": mean * 100.0,
        "sd_diff_pp": sd * 100.0,
        "t": t,
        "p_two": t_p_two(abs(t), len(diffs) - 1),
    }


def report_rates(pattern: str, metric: str) -> list[float]:
    out = []
    for seed in SEEDS:
        path = REPO_ROOT / pattern.format(seed=seed)
        out.append(json.loads(path.read_text())["metrics"][METRIC_KEY[metric]])
    return out


def family_rates(claims: dict, name: str, metric: str) -> list[float]:
    fam = next(f for f in claims["families"] if f["name"] == name)
    return [r[metric] for r in sorted(fam["rows"], key=lambda r: r["seed"])]


def main() -> None:
    claims = json.loads(CLAIMS.read_text())
    result: dict = {}
    for metric in ("strict", "coarse"):
        gen23 = {
            "dirty": report_rates("backend/reports/scale_grid/run_summaries/compactlm_23M_f100_s{seed}.json", metric),
            "clean": report_rates("backend/reports/scale_grid_clean/run_summaries/compactlm_23M_f100_s{seed}.json", metric),
        }
        gen60 = {
            "dirty": report_rates("backend/reports/scale_grid/run_summaries/compactlm_60M_f100_s{seed}.json", metric),
            "clean": report_rates("backend/reports/scale_grid_clean/run_summaries/compactlm_60M_f100_s{seed}.json", metric),
        }
        a2 = {
            "dirty": family_rates(claims, "23M policy-only LM", metric),
            "clean": report_rates("backend/reports/a2_policy_only_clean/run_summaries/a2_policy_lm_23M_s{seed}.json", metric),
        }
        b2_23 = {
            "dirty": family_rates(claims, "23M tuple head", metric),
            "clean": report_rates("backend/reports/tuple_head_clean/run_summaries/tuple_head_23M_f100_s{seed}.json", metric),
        }
        b2_60 = {
            "dirty": family_rates(claims, "60M tuple head", metric),
            "clean": report_rates("backend/reports/tuple_head_60M_clean/run_summaries/tuple_head_60M_f100_s{seed}.json", metric),
        }

        # backbone assignment per condition: (generative side, head side)
        conditions = {
            "dirty": ("dirty", "dirty"),
            "mixed": ("clean", "dirty"),
            "fully_clean": ("clean", "clean"),
        }
        per_condition = {}
        for cond, (g, h) in conditions.items():
            per_condition[cond] = {
                "C1_tuple23_vs_gen60": paired_t(gen60[g], b2_23[h]),
                "C2_data_effect_A2_minus_A1": paired_t(gen23[g], a2[h]),
                "C2_representation_effect_B2_minus_A2": paired_t(a2[h], b2_23[h]),
                "C2_composite_B2_minus_A1": paired_t(gen23[g], b2_23[h]),
                "tuple_scale_60_minus_23": paired_t(b2_23[h], b2_60[h]),
            }
        result[metric] = {
            "conditions": per_condition,
            "family_means_pct": {
                "gen23": {k: statistics.fmean(v) * 100 for k, v in gen23.items()},
                "gen60": {k: statistics.fmean(v) * 100 for k, v in gen60.items()},
                "A2_policy23": {k: statistics.fmean(v) * 100 for k, v in a2.items()},
                "B2_tuple23": {k: statistics.fmean(v) * 100 for k, v in b2_23.items()},
                "B2_tuple60": {k: statistics.fmean(v) * 100 for k, v in b2_60.items()},
            },
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    (REPO_ROOT / "results").mkdir(parents=True, exist_ok=True)
    out = REPO_ROOT / "results/c1_c2_fully_clean_results.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
