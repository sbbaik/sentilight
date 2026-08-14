"""Natural-segment evals for dirty + clean scale grids (prereg P2/P3).

Mirrors tools/run_scale_grid.py eval commands exactly except --segment natural.
Skips any eval whose report JSON already exists, so it is restartable.
"""
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v5")
PY = "/home/sbbaik/miniconda3/envs/sentilight/bin/python"
DATASET = "backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl"
GPUS = [1, 0]

GRIDS = [
    ("backend/models/compact_llm/scale_grid_runs", "backend/reports/scale_grid_natural"),
    ("backend/models/compact_llm/scale_grid_runs_clean", "backend/reports/scale_grid_clean_natural"),
]
SCALES = ["3M", "8M", "23M", "60M"]
SEEDS = [42, 43, 44]
FRACTIONS = [25, 50, 100]


def build_tasks():
    tasks = []
    for run_root, report_root in GRIDS:
        for scale, seed, frac in itertools.product(SCALES, SEEDS, FRACTIONS):
            ckpt = REPO / run_root / scale / f"seed_{seed}" / f"f{frac}" / "sft" / "compactlm_from_scratch_best.pt"
            model_id = f"compactlm_{scale}_f{frac}_s{seed}"
            report_json = REPO / report_root / "run_summaries" / f"{model_id}.json"
            if report_json.exists():
                continue
            if not ckpt.exists():
                print(f"[MISSING CKPT] {ckpt}", flush=True)
                continue
            (REPO / report_root / "run_summaries").mkdir(parents=True, exist_ok=True)
            (REPO / report_root / "per_row").mkdir(parents=True, exist_ok=True)
            cmd = [
                PY, "tools/evaluate_compactlm_checkpoint.py",
                "--checkpoint", str(ckpt.relative_to(REPO)),
                "--dataset", DATASET,
                "--segment", "natural",
                "--device", "cuda:0",
                "--model-id", model_id,
                "--display-name", f"CompactLM {scale} {frac}% seed {seed}",
                "--per-row-jsonl", f"{report_root}/per_row/{model_id}.jsonl",
                "--report-json", f"{report_root}/run_summaries/{model_id}.json",
                "--report-md", f"{report_root}/run_summaries/{model_id}.md",
                "--progress-every", "200",
            ]
            tasks.append((model_id, report_root, cmd))
    return tasks


def main():
    tasks = build_tasks()
    print(f"{len(tasks)} natural-segment evals to run", flush=True)
    pending = list(tasks)
    running = {}  # gpu -> (model_id, report_root, popen, t0)
    done = failed = 0
    while pending or running:
        for gpu in list(running):
            mid, rroot, proc, t0 = running[gpu]
            rc = proc.poll()
            if rc is None:
                continue
            dt = time.time() - t0
            if rc == 0:
                done += 1
                print(f"[done {done}/{len(tasks)}] {rroot}:{mid} ({dt:.0f}s)", flush=True)
            else:
                failed += 1
                print(f"[FAILED rc={rc}] {rroot}:{mid} ({dt:.0f}s)", flush=True)
            del running[gpu]
        for gpu in GPUS:
            if gpu in running or not pending:
                continue
            mid, rroot, cmd = pending.pop(0)
            import os
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log = open(REPO / rroot / "run_summaries" / f"{mid}.eval.log", "a")
            proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
            running[gpu] = (mid, rroot, proc, time.time())
            print(f"[launch gpu={gpu}] {rroot}:{mid}", flush=True)
        time.sleep(5)
    print(f"ALL DONE: {done} ok, {failed} failed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
