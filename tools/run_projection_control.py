"""Nearest-tuple projection control for the generative baselines.

Procedure, projection metric and headline decision rule are hash-frozen in
results/prereg_w2_constrained_decoding.md (SHA-256 ea94b1d3...),
written before any number here was computed.

Projects each generative prediction onto the nearest of the 240 valid policy
tuples under a normalized Euclidean metric (circular hue / 180, S,B,Dimmer / 100,
CT / 347 = COARSE_CT_MAX - COARSE_CT_MIN), equal weights, deterministic tie-break
on the lowest tuple index. Re-scores with the SAME strict/coarse functions the
original evaluation used. Clean backbone only.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "backend" / "models"))

from common.benchmark_eval import (  # noqa: E402
    coarse_3bin_pass,
    hue_distance,
    strict_semantic_pass,
)
from compact_llm.tuple_classifier import build_tuple_vocab  # noqa: E402

SEEDS = [42, 43, 44]
DIMS = ("H", "S", "B", "Dimmer", "CT")
CT_RANGE = 500 - 153  # COARSE_CT_MAX - COARSE_CT_MIN, per the frozen declaration

GEN = {
    "gen23": "backend/reports/scale_grid_clean/per_row/compactlm_23M_f100_s{seed}.jsonl",
    "gen60": "backend/reports/scale_grid_clean/per_row/compactlm_60M_f100_s{seed}.jsonl",
}
B2 = "backend/reports/tuple_head_clean/per_row/tuple_head_23M_f100_s{seed}.jsonl"


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
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
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def binom_sf_le(k: int, n: int) -> float:
    return sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2.0 * binom_sf_le(min(b, c), n))


VOCAB = build_tuple_vocab()


def project(pred: dict) -> tuple[dict, float]:
    """Nearest valid tuple under the frozen normalized Euclidean metric."""
    best_i, best_d = 0, float("inf")
    for i, v in enumerate(VOCAB):
        d = (hue_distance(int(pred["H"]), int(v["H"])) / 180.0) ** 2
        d += (abs(int(pred["S"]) - int(v["S"])) / 100.0) ** 2
        d += (abs(int(pred["B"]) - int(v["B"])) / 100.0) ** 2
        d += (abs(int(pred["Dimmer"]) - int(v["Dimmer"])) / 100.0) ** 2
        d += (abs(int(pred["CT"]) - int(v["CT"])) / CT_RANGE) ** 2
        if d < best_d:              # strict < keeps the lowest index on ties
            best_i, best_d = i, d
    return VOCAB[best_i], math.sqrt(best_d)


def load(path: Path) -> dict[int, dict]:
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[int(r["row_index"])] = r
    return out


def project_rows(rows: dict[int, dict]) -> dict[int, dict]:
    out = {}
    for i, r in rows.items():
        if not r.get("success"):
            out[i] = {"strict": False, "coarse": False, "changed": False, "dist": None}
            continue
        proj, dist = project(r["predicted"])
        expected = r["reference"]
        out[i] = {
            "strict": strict_semantic_pass(expected, proj, r),
            "coarse": coarse_3bin_pass(expected, proj),
            "changed": any(int(proj[k]) != int(r["predicted"][k]) for k in DIMS),
            "dist": dist,
        }
    return out


def stouffer(per_seed: list[dict]) -> tuple[float, float]:
    zs = []
    for s in per_seed:
        p, sign = s["p_two"], s["sign"]
        if 1.0 - p / 2.0 >= 1.0:
            n = s["b"] + s["c"]
            z = abs(s["c"] - s["b"]) / math.sqrt(n) if n else 0.0
        else:
            z = norm_ppf(1.0 - p / 2.0)
        zs.append(sign * z)
    z = sum(zs) / math.sqrt(len(zs))
    return z, 2.0 * (1.0 - norm_cdf(abs(z)))


def contrast(base_flags: list[dict], treat_flags: list[dict], metric: str) -> dict:
    per_seed = []
    for base, treat in zip(base_flags, treat_flags):
        idx = sorted(set(base) & set(treat))
        b = sum(1 for i in idx if base[i][metric] and not treat[i][metric])
        c = sum(1 for i in idx if treat[i][metric] and not base[i][metric])
        per_seed.append({"b": b, "c": c, "p_two": mcnemar_exact(b, c),
                         "sign": 0 if c == b else (1 if c > b else -1),
                         "delta_pp": (c - b) / len(idx) * 100.0})
    z, p = stouffer(per_seed)
    return {"per_seed": per_seed, "z_stouffer": z, "p_combined": p,
            "mean_delta_pp": sum(s["delta_pp"] for s in per_seed) / len(per_seed),
            "signs_consistent": len({s["sign"] for s in per_seed}) == 1 and per_seed[0]["sign"] != 0}


def rate(flags: list[dict], metric: str) -> float:
    return sum(sum(1 for v in f.values() if v[metric]) / len(f) for f in flags) / len(flags) * 100.0


def main() -> None:
    raw = {k: [load(REPO_ROOT / p.format(seed=s)) for s in SEEDS] for k, p in GEN.items()}
    b2 = [load(REPO_ROOT / B2.format(seed=s)) for s in SEEDS]
    b2_flags = [{i: {"strict": r["strict"], "coarse": r["coarse"]} for i, r in f.items()} for f in b2]
    orig = {k: [{i: {"strict": r["strict"], "coarse": r["coarse"]} for i, r in f.items()} for f in v]
            for k, v in raw.items()}
    proj = {k: [project_rows(f) for f in v] for k, v in raw.items()}

    result: dict = {"frozen_declaration_sha256": "ea94b1d3e4274b144a7ada40a9f3a298e9f8a50a01dd6fb2cc5b8f64b83dcdee",
                    "rates_pct": {}, "projection_effect": {}, "gate": {}}

    for name in ("gen23", "gen60"):
        result["rates_pct"][name] = {
            m: {"original": rate(orig[name], m), "projected": rate(proj[name], m)}
            for m in ("strict", "coarse")
        }
        changed = [sum(1 for v in f.values() if v["changed"]) for f in proj[name]]
        dists = [v["dist"] for f in proj[name] for v in f.values() if v["dist"] is not None]
        result["projection_effect"][name] = {
            "rows_changed_per_seed": changed,
            "pct_changed": sum(changed) / (len(changed) * 1000) * 100.0,
            "mean_projection_distance": sum(dists) / len(dists),
            "max_projection_distance": max(dists),
            "self_vs_projected": {m: contrast(orig[name], proj[name], m) for m in ("strict", "coarse")},
        }
    result["rates_pct"]["B2_23"] = {m: {"original": rate(b2_flags, m)} for m in ("strict", "coarse")}

    # Gate family: C1' = B2 - projected gen60, Holm m=2
    gate = {m: contrast(proj["gen60"], b2_flags, m) for m in ("strict", "coarse")}
    ps = sorted(gate.items(), key=lambda kv: kv[1]["p_combined"])
    running = 0.0
    for i, (m, v) in enumerate(ps):
        running = max(running, min(1.0, (2 - i) * v["p_combined"]))
        v["p_holm"] = running
        v["adopted"] = running < 0.05 and v["signs_consistent"]
    result["gate"]["C1_prime"] = gate

    gap_before = rate(b2_flags, "coarse") - rate(orig["gen60"], "coarse")
    gap_after = rate(b2_flags, "coarse") - rate(proj["gen60"], "coarse")
    closure = (gap_before - gap_after) / gap_before
    gb_s = rate(b2_flags, "strict") - rate(orig["gen60"], "strict")
    ga_s = rate(b2_flags, "strict") - rate(proj["gen60"], "strict")
    result["gate"]["closure"] = {
        "coarse": {"gap_before_pp": gap_before, "gap_after_pp": gap_after, "closure": closure},
        "strict_reference_only": {"gap_before_pp": gb_s, "gap_after_pp": ga_s,
                                  "closure": (gb_s - ga_s) / gb_s if gb_s else None},
    }
    branch = ("A_projection_closes_most" if closure >= 0.66
              else "B_head_specific" if closure <= 0.33
              else "C_partial")
    result["gate"]["headline_branch"] = branch

    (REPO_ROOT / "results").mkdir(parents=True, exist_ok=True)
    out = REPO_ROOT / "results/projection_control_results.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== pass rates (%, clean backbone, mean of 3 seeds) ===")
    for name in ("gen23", "gen60"):
        for m in ("strict", "coarse"):
            r = result["rates_pct"][name][m]
            print(f"  {name:6s} {m:7s} original {r['original']:6.2f}  ->  projected {r['projected']:6.2f}"
                  f"  ({r['projected']-r['original']:+.2f}pp)")
    for m in ("strict", "coarse"):
        print(f"  B2_23  {m:7s} {result['rates_pct']['B2_23'][m]['original']:6.2f}")
    print("\n=== projection effect ===")
    for name in ("gen23", "gen60"):
        e = result["projection_effect"][name]
        print(f"  {name}: {e['pct_changed']:.1f}% of rows changed, mean dist {e['mean_projection_distance']:.4f}, max {e['max_projection_distance']:.4f}")
    print("\n=== GATE: C1' = B2_23 - projected gen60 (Holm m=2) ===")
    for m in ("coarse", "strict"):
        v = gate[m]
        print(f"  {m:7s} delta {v['mean_delta_pp']:+6.2f}pp  z {v['z_stouffer']:+7.2f}  "
              f"p_holm {v['p_holm']:.3e}  signs {'same' if v['signs_consistent'] else 'SPLIT'}  "
              f"{'ADOPTED' if v['adopted'] else 'not adopted'}")
    c = result["gate"]["closure"]["coarse"]
    print(f"\n  coarse gap {c['gap_before_pp']:+.2f}pp -> {c['gap_after_pp']:+.2f}pp   CLOSURE = {c['closure']:.3f}")
    print(f"  HEADLINE BRANCH: {branch}")


if __name__ == "__main__":
    main()
