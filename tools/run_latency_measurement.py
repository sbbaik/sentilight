"""Latency accounting under a single measurement protocol.

The latency numbers previously carried in the storyline are not comparable, for
three reasons discovered while auditing the evaluators:

  1. tools/evaluate_tuple_classifier.py times a BATCH of 64 and divides by 64,
     so the head's "per-row latency" is amortized batched throughput, while
     tools/evaluate_compactlm_checkpoint.py times one sequential row at a time.
  2. Neither evaluator calls torch.cuda.synchronize(), so GPU work is timed
     asynchronously. A single classifier kernel launch returns almost
     immediately, which understates the head; the generative loop forces
     implicit syncs, so it is timed more nearly correctly. The measured ratio is
     therefore inflated by an unknown factor.
  3. CompactLM.generate() has NO KV cache -- it re-runs a full-prefix forward at
     every decode step (its own docstring says it is a sample-checking helper and
     that optimized inference is written separately). The generative number is an
     upper bound on a reference implementation, not a benchmark of an optimized
     decoder.

This harness re-measures under one honest protocol: batch size 1 for every
family, torch.cuda.synchronize() around each timed region, warm-up iterations
discarded, same device, same rows. It also records the actual decode-step count,
which is the implementation-independent quantity behind the architectural claim
(1 forward pass vs N).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "backend" / "models"))

from common.benchmark_eval import load_jsonl  # noqa: E402
from compact_llm.inference import load_model, predict  # noqa: E402
from compact_llm.training_data import build_prompt  # noqa: E402
from compact_llm.tuple_classifier import (  # noqa: E402
    SentilightTupleClassifier,
    build_tuple_vocab,
    output_tuple,
    tuple_to_label_map,
)
from compact_llm.model_definition import ModelConfig  # noqa: E402

DATASET = REPO_ROOT / "backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl"
TOKENIZER = REPO_ROOT / "backend/models/compact_llm/tokenizer/tokenizer.json"
WARMUP = 20


class Cfg:
    def __init__(self, checkpoint, device, max_new_tokens=96):
        self.resolved_model_dir = str(REPO_ROOT / "backend/models/compact_llm")
        self.checkpoint = str(checkpoint)
        self.tokenizer_dir = "tokenizer"
        self.options = {"device": device, "max_new_tokens": max_new_tokens}


def rule_rows(n: int) -> list[dict]:
    rows = [r for r in load_jsonl(DATASET) if r.get("source") != "natural_language_baseline"]
    return rows[:n]


def time_generative(checkpoint: Path, rows: list[dict], device: str) -> dict:
    import torch

    runtime = load_model(Cfg(checkpoint, device))
    torch.cuda.synchronize()
    for r in rows[:WARMUP]:
        predict(str(r["input"]), runtime)
    torch.cuda.synchronize()

    lat, steps = [], []
    eos = runtime.eos_id
    for r in rows:
        prompt_ids = runtime.tokenizer.encode(build_prompt(str(r["input"]))).ids[-runtime.model.cfg.block_size:]
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=runtime.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = runtime.model.generate(ids, max_new_tokens=runtime.max_new_tokens,
                                         temperature=0.0, eos_id=eos)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000.0)
        steps.append(out.shape[1] - len(prompt_ids))
    return {"latency_ms": lat, "decode_steps": steps,
            "prompt_len_mean": st.fmean(len(runtime.tokenizer.encode(build_prompt(str(r["input"]))).ids) for r in rows)}


def time_tuple_head(checkpoint: Path, rows: list[dict], device: str) -> dict:
    import torch
    from tokenizers import Tokenizer

    ck = torch.load(checkpoint, map_location="cpu")
    vocab = build_tuple_vocab()
    cfg = ModelConfig(**ck["model_config"])
    model = SentilightTupleClassifier(cfg, tuple_vocab=vocab)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    tok = Tokenizer.from_file(str(TOKENIZER))

    def encode(text):
        ids = tok.encode(build_prompt(text)).ids[-cfg.block_size:]
        return torch.tensor([ids], dtype=torch.long, device=device)

    torch.cuda.synchronize()
    for r in rows[:WARMUP]:
        with torch.inference_mode():
            model(encode(str(r["input"])))
    torch.cuda.synchronize()

    lat = []
    for r in rows:
        x = encode(str(r["input"]))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model(x)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000.0)
    return {"latency_ms": lat, "decode_steps": [1] * len(lat)}


def summarize(lat: list[float]) -> dict:
    s = sorted(lat)
    n = len(s)
    return {"n": n, "mean_ms": st.fmean(s), "sd_ms": st.stdev(s) if n > 1 else 0.0,
            "p50_ms": s[n // 2], "p95_ms": s[min(n - 1, int(n * 0.95))], "max_ms": s[-1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = rule_rows(args.rows + WARMUP)
    s = args.seed
    targets = [
        ("A1_generative_23M", "generative",
         REPO_ROOT / f"backend/models/compact_llm/scale_grid_runs_clean/23M/seed_{s}/f100/sft/compactlm_from_scratch_best.pt"),
        ("generative_60M", "generative",
         REPO_ROOT / f"backend/models/compact_llm/scale_grid_runs_clean/60M/seed_{s}/f100/sft/compactlm_from_scratch_best.pt"),
        ("B2_tuple_head_23M", "head",
         REPO_ROOT / f"backend/models/compact_llm/tuple_head_runs_clean/23M/seed_{s}/compactlm_tuple_classifier_best.pt"),
        ("B2_tuple_head_60M", "head",
         REPO_ROOT / f"backend/models/compact_llm/tuple_head_runs_clean/60M/seed_{s}/compactlm_tuple_classifier_best.pt"),
    ]

    result = {"protocol": {"batch_size": 1, "cuda_synchronize": True, "warmup_rows": WARMUP,
                           "timed_rows": args.rows, "device": args.device, "seed": s,
                           "kv_cache": False,
                           "note": "generative path has no KV cache; numbers are an upper bound "
                                   "on a reference implementation, not an optimized decoder"},
              "families": {}}
    for name, kind, ckpt in targets:
        if not ckpt.exists():
            print(f"[MISSING] {name}: {ckpt}", flush=True)
            continue
        print(f"timing {name} ...", flush=True)
        out = time_generative(ckpt, rows, args.device) if kind == "generative" \
            else time_tuple_head(ckpt, rows, args.device)
        timed = out["latency_ms"][WARMUP:]
        steps = out["decode_steps"][WARMUP:]
        result["families"][name] = {
            "kind": kind,
            **summarize(timed),
            "decode_steps_mean": st.fmean(steps),
            "decode_steps_p50": sorted(steps)[len(steps) // 2],
            "decode_steps_max": max(steps),
        }
        f = result["families"][name]
        print(f"  mean {f['mean_ms']:.3f} ms  p50 {f['p50_ms']:.3f}  p95 {f['p95_ms']:.3f}  "
              f"steps {f['decode_steps_mean']:.1f}", flush=True)

    fam = result["families"]
    if "generative_60M" in fam and "B2_tuple_head_23M" in fam:
        g, h = fam["generative_60M"], fam["B2_tuple_head_23M"]
        result["ratios"] = {
            "measured_mean_ratio_gen60_over_head23": g["mean_ms"] / h["mean_ms"],
            "measured_p50_ratio": g["p50_ms"] / h["p50_ms"],
            "architectural_forward_passes_gen60": g["decode_steps_mean"],
            "architectural_forward_passes_head": 1,
            "note": "the architectural ratio (N forward passes vs 1) is implementation "
                    "independent; the measured wall-clock ratio is not, and must not be "
                    "quoted as a speedup benchmark",
        }
    out_path = REPO_ROOT / "results/latency_results.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result.get("ratios", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
