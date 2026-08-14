"""Row-level McNemar -> Stouffer across seeds -> Holm over the declared family.

Procedure and contrast family are hash-frozen in
results/prereg_w1_contrast_family.md (SHA-256 f90ce79f...), written
before any result here was computed.

  1. Pair the same 1000 rule rows by row_index; outcome is the per-row boolean
     `strict` / `coarse` pass flag.
  2. Per seed: exact two-sided binomial McNemar on the discordant pairs (b, c).
     No chi-square approximation. b + c == 0 -> p = 1.
  3. Stouffer across seeds 42/43/44: z_i = sign_i * Phi^-1(1 - p_i/2),
     z = sum(z_i)/sqrt(3), combined two-sided p = 2*(1 - Phi(|z|)).
  4. Holm-Bonferroni over the m=8 primary family. The secondary tuple-head
     scale contrast is reported uncorrected and never as confirmatory.

Adjudicated on the fully-clean backbone; dirty/mixed reported for robustness only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44]

# per-row sources: family -> backbone -> path template
SOURCES = {
    "gen23": {
        "dirty": "backend/reports/scale_grid/per_row/compactlm_23M_f100_s{seed}.jsonl",
        "clean": "backend/reports/scale_grid_clean/per_row/compactlm_23M_f100_s{seed}.jsonl",
    },
    "gen60": {
        "dirty": "backend/reports/scale_grid/per_row/compactlm_60M_f100_s{seed}.jsonl",
        "clean": "backend/reports/scale_grid_clean/per_row/compactlm_60M_f100_s{seed}.jsonl",
    },
    "A2": {
        "dirty": "backend/reports/a2_policy_only/per_row/a2_policy_lm_23M_s{seed}.jsonl",
        "clean": "backend/reports/a2_policy_only_clean/per_row/a2_policy_lm_23M_s{seed}.jsonl",
    },
    "B2_23": {
        "dirty": "backend/reports/tuple_head/per_row/tuple_head_23M_f100_s{seed}.jsonl",
        "clean": "backend/reports/tuple_head_clean/per_row/tuple_head_23M_f100_s{seed}.jsonl",
    },
    "B2_60": {
        "dirty": "backend/reports/tuple_head_60M/per_row/tuple_head_60M_f100_s{seed}.jsonl",
        "clean": "backend/reports/tuple_head_60M_clean/per_row/tuple_head_60M_f100_s{seed}.jsonl",
    },
}

# (key, label, baseline, treatment) -- treatment expected to beat baseline
PRIMARY = [
    ("C1", "C1 B2_tuple23 - gen60", "gen60", "B2_23"),
    ("C2a", "C2a data alignment A2 - A1(gen23)", "gen23", "A2"),
    ("C2b", "C2b output structure B2 - A2", "A2", "B2_23"),
    ("C2c", "C2c composite B2 - A1(gen23)", "gen23", "B2_23"),
]
SECONDARY = [
    ("SCALE", "tuple-head scale B2_60 - B2_23", "B2_23", "B2_60"),
]
METRICS = ("strict", "coarse")


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError(p)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def binom_sf_le(k: int, n: int) -> float:
    """P(X <= k) for X ~ Binomial(n, 0.5), exact."""
    total = 0.0
    for i in range(k + 1):
        total += math.comb(n, i)
    return total / (2.0 ** n)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar. b = only-baseline-passes, c = only-treatment."""
    n = b + c
    if n == 0:
        return 1.0
    p = 2.0 * binom_sf_le(min(b, c), n)
    return min(1.0, p)


def load_flags(path: Path) -> dict[int, dict[str, bool]]:
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[int(r["row_index"])] = {"strict": bool(r["strict"]), "coarse": bool(r["coarse"])}
    return out


def contrast(backbone: str, baseline: str, treatment: str, metric: str) -> dict:
    per_seed = []
    for seed in SEEDS:
        base = load_flags(REPO_ROOT / SOURCES[baseline][backbone].format(seed=seed))
        treat = load_flags(REPO_ROOT / SOURCES[treatment][backbone].format(seed=seed))
        shared = sorted(set(base) & set(treat))
        b = sum(1 for i in shared if base[i][metric] and not treat[i][metric])
        c = sum(1 for i in shared if treat[i][metric] and not base[i][metric])
        p = mcnemar_exact(b, c)
        sign = 0 if c == b else (1 if c > b else -1)
        per_seed.append({
            "seed": seed, "rows": len(shared), "b_only_baseline": b,
            "c_only_treatment": c, "p_two": p, "sign": sign,
            "delta_pp": (c - b) / len(shared) * 100.0,
        })
    # Stouffer. For very small exact p, 1 - p/2 rounds to 1.0 in double precision
    # and norm_ppf is undefined; fall back to the standard McNemar normal
    # statistic z = (c - b) / sqrt(b + c), which agrees with the exact test to
    # several digits in exactly the large-discordance regime where p underflows.
    zs = []
    for s in per_seed:
        p = s["p_two"]
        if 1.0 - p / 2.0 >= 1.0:
            n = s["b_only_baseline"] + s["c_only_treatment"]
            z = abs(s["c_only_treatment"] - s["b_only_baseline"]) / math.sqrt(n) if n else 0.0
            s["z_source"] = "mcnemar_normal_fallback"
        else:
            z = norm_ppf(1.0 - p / 2.0)
            s["z_source"] = "exact"
        s["z"] = s["sign"] * z
        zs.append(s["sign"] * z)
    z_comb = sum(zs) / math.sqrt(len(zs))
    p_comb = 2.0 * (1.0 - norm_cdf(abs(z_comb)))
    return {
        "per_seed": per_seed,
        "signs_consistent": len({s["sign"] for s in per_seed}) == 1 and per_seed[0]["sign"] != 0,
        "z_stouffer": z_comb,
        "p_combined": p_comb,
        "mean_delta_pp": sum(s["delta_pp"] for s in per_seed) / len(per_seed),
    }


def holm(pvals: dict[str, float]) -> dict[str, float]:
    m = len(pvals)
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    adj = {}
    running = 0.0
    for i, (k, p) in enumerate(ordered):
        val = min(1.0, (m - i) * p)
        running = max(running, val)  # enforce monotonicity
        adj[k] = running
    return adj


def main() -> None:
    result = {"frozen_declaration_sha256": "f90ce79f9ee324c056cda88013fdeefd1b36a7c03c6ca8979501e2e73a702d55"}
    for backbone in ("dirty", "clean"):
        block = {"primary": {}, "secondary": {}}
        for key, label, base, treat in PRIMARY:
            for metric in METRICS:
                block["primary"][f"{key}_{metric}"] = {
                    "label": f"{label} [{metric}]", **contrast(backbone, base, treat, metric)
                }
        for key, label, base, treat in SECONDARY:
            for metric in METRICS:
                block["secondary"][f"{key}_{metric}"] = {
                    "label": f"{label} [{metric}]", **contrast(backbone, base, treat, metric)
                }
        adj = holm({k: v["p_combined"] for k, v in block["primary"].items()})
        for k, v in block["primary"].items():
            v["p_holm"] = adj[k]
            v["adopted"] = adj[k] < 0.05 and v["signs_consistent"]
        result[backbone] = block

    (REPO_ROOT / "results").mkdir(parents=True, exist_ok=True)
    out = REPO_ROOT / "results/row_level_results.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for backbone in ("dirty", "clean"):
        print(f"\n{'='*72}\nBACKBONE: {backbone}{'  <-- ADJUDICATED' if backbone=='clean' else '  (robustness only)'}")
        print(f"{'contrast':28s} {'meanD':>8s} {'z':>7s} {'p_comb':>10s} {'p_holm':>10s} {'signs':>6s} {'adopt':>6s}")
        for k, v in result[backbone]["primary"].items():
            print(f"{k:28s} {v['mean_delta_pp']:+8.2f} {v['z_stouffer']:+7.2f} "
                  f"{v['p_combined']:10.2e} {v['p_holm']:10.2e} {str(v['signs_consistent']):>6s} "
                  f"{'YES' if v['adopted'] else 'no':>6s}")
        for k, v in result[backbone]["secondary"].items():
            print(f"{k:28s} {v['mean_delta_pp']:+8.2f} {v['z_stouffer']:+7.2f} "
                  f"{v['p_combined']:10.2e} {'(uncorr)':>10s} {str(v['signs_consistent']):>6s} {'n/a':>6s}")


if __name__ == "__main__":
    main()
