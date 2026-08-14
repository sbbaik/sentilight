"""Recompute every trivial baseline on the SAME 1000-row rule segment.

Earlier baseline figures for this task (uniform random strict 1.11% / coarse 6.17%,
train-majority tuple strict 1.50% / coarse 1.70%, SBERT-centroid strict 2.40% /
coarse 7.55%) were mis-sourced: the SBERT number came from a 2000-row report, a
different denominator from the 1000-row rule segment every claim uses. This
recomputes them all on the claim segment with the scorer of record.

It also adds the baseline that actually matters and was missing: the BEST
CONSTANT predictor. A reviewer asks "what does always guessing the same thing
get you?", and the honest answer is the maximum over all 240 policy tuples, not
the training-set mode.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO / "backend" / "models"))

from common.benchmark_eval import (  # noqa: E402
    coarse_3bin_pass, coarse_bins, load_jsonl, normalize_lighting, strict_semantic_pass,
)
from compact_llm.tuple_classifier import build_tuple_vocab, output_tuple  # noqa: E402

DATASET = REPO / "backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl"
TRAIN = REPO / "backend/models/compact_llm/datasets/scale_grid_subsets/train_100.jsonl"
DIMS = ("H", "S", "B", "CT", "Dimmer")


def rule_rows() -> list[dict]:
    return [r for r in load_jsonl(DATASET) if r.get("source") != "natural_language_baseline"]


def score_constant(rows: list[dict], tup: dict) -> tuple[float, float]:
    s = sum(1 for r in rows if strict_semantic_pass(normalize_lighting(r["output"]), tup, r))
    c = sum(1 for r in rows if coarse_3bin_pass(normalize_lighting(r["output"]), tup))
    return s / len(rows) * 100.0, c / len(rows) * 100.0


def main() -> None:
    rows = rule_rows()
    vocab = build_tuple_vocab()
    n = len(rows)
    out: dict = {"segment": "rule", "rows": n, "baselines": {}}

    # --- uniform random over the 240 policy tuples (exact expectation) ---
    per_tuple = [score_constant(rows, t) for t in vocab]
    out["baselines"]["uniform_random_over_240_tuples"] = {
        "strict_pct": st.fmean(s for s, _ in per_tuple),
        "coarse_pct": st.fmean(c for _, c in per_tuple),
        "method": "exact expectation = mean pass rate over all 240 tuples",
    }

    # --- train-set majority tuple ---
    from collections import Counter
    counts = Counter()
    for r in load_jsonl(TRAIN):
        counts[output_tuple(r["output"])] += 1
    mode_key, mode_n = counts.most_common(1)[0]
    mode_tuple = dict(zip(("H", "S", "B", "Dimmer", "CT"), mode_key))
    s, c = score_constant(rows, mode_tuple)
    out["baselines"]["train_majority_tuple"] = {
        "strict_pct": s, "coarse_pct": c, "tuple": mode_tuple,
        "train_occurrences": mode_n,
    }

    # --- BEST constant tuple (the strongest constant predictor) ---
    bs_i = max(range(len(vocab)), key=lambda i: per_tuple[i][0])
    bc_i = max(range(len(vocab)), key=lambda i: per_tuple[i][1])
    out["baselines"]["best_constant_tuple"] = {
        "strict_pct": per_tuple[bs_i][0], "strict_tuple": vocab[bs_i],
        "coarse_pct": per_tuple[bc_i][1], "coarse_tuple": vocab[bc_i],
        "method": "oracle-selected constant; upper bound on any constant predictor",
    }

    # --- best constant COARSE BIN vector (not necessarily a valid tuple) ---
    refbins = [coarse_bins(normalize_lighting(r["output"])) for r in rows]
    majority_bins = {d: Counter(b[d] for b in refbins).most_common(1)[0][0] for d in DIMS}
    hit = sum(1 for b in refbins if all(b[d] == majority_bins[d] for d in DIMS))
    out["baselines"]["majority_bin_vector"] = {
        "coarse_pct": hit / n * 100.0, "bins": majority_bins,
        "note": "not reachable by any single policy tuple unless it happens to bin this way",
    }

    # --- SBERT restricted to the rule segment ---
    seg = json.loads((REPO / "backend/reports/five_model_benchmark_v3_by_segment.json").read_text())
    sbert = {}
    for name, block in seg["segments"].items():
        for m in block["models"]:
            if m.get("model_id") == "sbert":
                sbert[name] = {
                    "rows": m["successes"],
                    "strict_pct": m["strict_rate"] * 100.0,
                    "coarse_pct": m["coarse_rate"] * 100.0,
                }
    sbert["_note"] = ("the previously quoted 2.40% / 7.55% are the all_2000 numbers; "
                      "the claim segment is rule_based_1000, where SBERT scores strict 4.70% "
                      "and coarse 10.30%. The all_2000 figure is diluted by the natural "
                      "segment, which is contaminated and never used for evaluation.")
    out["baselines"]["sbert_centroid_by_segment"] = sbert

    (REPO / "results").mkdir(parents=True, exist_ok=True)

    (REPO / "results/trivial_baselines_rule1000.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    b = out["baselines"]
    print(f"rule segment, n={n}\n")
    print(f"{'baseline':34s} {'strict':>9s} {'coarse':>9s}")
    print(f"{'uniform random over 240 tuples':34s} {b['uniform_random_over_240_tuples']['strict_pct']:8.2f}% "
          f"{b['uniform_random_over_240_tuples']['coarse_pct']:8.2f}%")
    print(f"{'train-majority tuple':34s} {b['train_majority_tuple']['strict_pct']:8.2f}% "
          f"{b['train_majority_tuple']['coarse_pct']:8.2f}%")
    print(f"{'BEST constant tuple':34s} {b['best_constant_tuple']['strict_pct']:8.2f}% "
          f"{b['best_constant_tuple']['coarse_pct']:8.2f}%")
    print(f"{'majority coarse-bin vector':34s} {'-':>9s} {b['majority_bin_vector']['coarse_pct']:8.2f}%")
    if sbert:
        for k, v in sbert.items():
            if not isinstance(v, dict):
                continue
            print(f"{'SBERT centroid [' + k + ']':34s} {v['strict_pct']:8.2f}% "
                  f"{v['coarse_pct']:8.2f}%  (n={v['rows']})")


if __name__ == "__main__":
    main()
