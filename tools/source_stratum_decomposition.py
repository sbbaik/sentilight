"""Source-stratum decomposition of the row-level contrasts.

The rule segment's 1000 rows come from three strata:
  700  color_emotion_intensity_combinations
  200  conflict_and_color_adjective_cases
  100  ambiguous_fallback_cases

Because the rows are template-generated, the effective sample size is smaller
than the nominal 1000 and a large Stouffer z could in principle be a template
artifact. If every stratum moves in the same direction, that reading is ruled
out. This reports per-stratum paired deltas and exact McNemar per stratum,
pooled across seeds 42/43/44, on the fully-clean backbone (the adjudicated
condition). Exploratory: not part of the preregistered contrast family, uncorrected.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44]
STRATA = [
    "color_emotion_intensity_combinations",
    "conflict_and_color_adjective_cases",
    "ambiguous_fallback_cases",
]

SOURCES = {
    "gen23": "backend/reports/scale_grid_clean/per_row/compactlm_23M_f100_s{seed}.jsonl",
    "gen60": "backend/reports/scale_grid_clean/per_row/compactlm_60M_f100_s{seed}.jsonl",
    "A2": "backend/reports/a2_policy_only_clean/per_row/a2_policy_lm_23M_s{seed}.jsonl",
    "B2_23": "backend/reports/tuple_head_clean/per_row/tuple_head_23M_f100_s{seed}.jsonl",
    "B2_60": "backend/reports/tuple_head_60M_clean/per_row/tuple_head_60M_f100_s{seed}.jsonl",
}

CONTRASTS = [
    ("C1", "B2_tuple23 - gen60", "gen60", "B2_23"),
    ("C2a", "A2 - A1(gen23)", "gen23", "A2"),
    ("C2b", "B2 - A2", "A2", "B2_23"),
    ("C2c", "B2 - A1(gen23)", "gen23", "B2_23"),
]


def binom_sf_le(k: int, n: int) -> float:
    return sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * binom_sf_le(min(b, c), n))


def load(path: Path) -> dict[int, dict]:
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[int(r["row_index"])] = r
    return out


def main() -> None:
    result: dict = {}
    for key, label, base_name, treat_name in CONTRASTS:
        result[key] = {"label": label, "strata": {}}
        for metric in ("strict", "coarse"):
            for stratum in STRATA:
                b_tot = c_tot = n_tot = 0
                per_seed = []
                for seed in SEEDS:
                    base = load(REPO_ROOT / SOURCES[base_name].format(seed=seed))
                    treat = load(REPO_ROOT / SOURCES[treat_name].format(seed=seed))
                    idx = [i for i in sorted(set(base) & set(treat))
                           if base[i]["source"] == stratum]
                    b = sum(1 for i in idx if base[i][metric] and not treat[i][metric])
                    c = sum(1 for i in idx if treat[i][metric] and not base[i][metric])
                    b_tot += b
                    c_tot += c
                    n_tot += len(idx)
                    per_seed.append({"seed": seed, "n": len(idx), "b": b, "c": c,
                                     "delta_pp": (c - b) / len(idx) * 100.0 if idx else 0.0})
                entry = result[key]["strata"].setdefault(stratum, {})
                entry[metric] = {
                    "rows_per_seed": n_tot // len(SEEDS),
                    "b_pooled": b_tot,
                    "c_pooled": c_tot,
                    "delta_pp_pooled": (c_tot - b_tot) / n_tot * 100.0 if n_tot else 0.0,
                    "p_exact_pooled": mcnemar_exact(b_tot, c_tot),
                    "per_seed_delta_pp": [round(s["delta_pp"], 2) for s in per_seed],
                    "all_seeds_same_sign": len({(1 if s["delta_pp"] > 0 else -1 if s["delta_pp"] < 0 else 0)
                                                for s in per_seed}) == 1,
                }

    (REPO_ROOT / "results").mkdir(parents=True, exist_ok=True)
    out = REPO_ROOT / "results/source_stratum_decomposition.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for key, blk in result.items():
        print(f"\n{'='*78}\n{key}: {blk['label']}   (clean backbone, pooled over 3 seeds)")
        print(f"{'stratum':42s} {'n/seed':>7s} {'metric':>7s} {'delta':>8s} {'p_exact':>10s} {'signs':>6s}")
        for stratum, metrics in blk["strata"].items():
            for metric in ("strict", "coarse"):
                m = metrics[metric]
                print(f"{stratum:42s} {m['rows_per_seed']:7d} {metric:>7s} "
                      f"{m['delta_pp_pooled']:+8.2f} {m['p_exact_pooled']:10.2e} "
                      f"{'same' if m['all_seeds_same_sign'] else 'SPLIT':>6s}")


if __name__ == "__main__":
    main()
