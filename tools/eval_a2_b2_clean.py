"""Rule-segment evals for the clean-backbone A2/B2 retrains (post-hoc check).

Mirrors the original A2/B2 eval commands exactly (same dataset, same rule
segment, same model-id scheme) so the only changed variable remains the
pretrain backbone. Restartable: skips runs whose report JSON already exists.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v5")
PY = "/home/sbbaik/miniconda3/envs/sentilight/bin/python"
DATASET = "backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl"
SEEDS = [42, 43, 44]
GPUS = [0, 1]


def build_tasks():
    tasks = []

    # --- A2 clean: generative evaluator ---
    for seed in SEEDS:
        model_id = f"a2_policy_lm_23M_s{seed}"
        root = "backend/reports/a2_policy_only_clean"
        report = REPO / root / "run_summaries" / f"{model_id}.json"
        ckpt = REPO / f"backend/models/compact_llm/a2_policy_only_runs_clean/23M/seed_{seed}/compactlm_from_scratch_best.pt"
        if report.exists():
            continue
        if not ckpt.exists():
            print(f"[MISSING CKPT] {ckpt}", flush=True)
            continue
        (REPO / root / "run_summaries").mkdir(parents=True, exist_ok=True)
        (REPO / root / "per_row").mkdir(parents=True, exist_ok=True)
        cmd = [
            PY, "tools/evaluate_compactlm_checkpoint.py",
            "--checkpoint", str(ckpt.relative_to(REPO)),
            "--dataset", DATASET,
            "--segment", "rule",
            "--device", "cuda:0",
            "--model-id", model_id,
            "--display-name", f"A2 Policy-only Tied LM 23M seed {seed} (clean backbone)",
            "--per-row-jsonl", f"{root}/per_row/{model_id}.jsonl",
            "--report-json", f"{root}/run_summaries/{model_id}.json",
            "--report-md", f"{root}/run_summaries/{model_id}.md",
            "--progress-every", "200",
        ]
        tasks.append((model_id, cmd))

    # --- B2 clean: tuple classifier evaluator ---
    for scale in ("23M", "60M"):
        root = "backend/reports/tuple_head_clean" if scale == "23M" else "backend/reports/tuple_head_60M_clean"
        for seed in SEEDS:
            model_id = f"tuple_head_{scale}_f100_s{seed}"
            report = REPO / root / "run_summaries" / f"{model_id}.json"
            ckpt = REPO / f"backend/models/compact_llm/tuple_head_runs_clean/{scale}/seed_{seed}/compactlm_tuple_classifier_best.pt"
            if report.exists():
                continue
            if not ckpt.exists():
                print(f"[MISSING CKPT] {ckpt}", flush=True)
                continue
            (REPO / root / "run_summaries").mkdir(parents=True, exist_ok=True)
            (REPO / root / "per_row").mkdir(parents=True, exist_ok=True)
            cmd = [
                PY, "tools/evaluate_tuple_classifier.py",
                "--checkpoint", str(ckpt.relative_to(REPO)),
                "--dataset", DATASET,
                "--segment", "rule",
                "--device", "cuda:0",
                "--model-id", model_id,
                "--display-name", f"Tuple Head {scale} 100% seed {seed} (clean backbone)",
                "--per-row-jsonl", f"{root}/per_row/{model_id}.jsonl",
                "--report-json", f"{root}/run_summaries/{model_id}.json",
                "--report-md", f"{root}/run_summaries/{model_id}.md",
            ]
            tasks.append((model_id, cmd))

    return tasks


def main():
    tasks = build_tasks()
    print(f"{len(tasks)} clean-backbone A2/B2 evals", flush=True)
    pending = list(tasks)
    running = {}
    done = failed = 0
    logdir = REPO / "backend/logs/eval_clean_a2b2"
    logdir.mkdir(parents=True, exist_ok=True)
    while pending or running:
        for gpu in list(running):
            name, proc, t0, log = running[gpu]
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            dt = time.time() - t0
            if rc == 0:
                done += 1
                print(f"[done {done}/{len(tasks)}] {name} ({dt:.0f}s)", flush=True)
            else:
                failed += 1
                print(f"[FAILED rc={rc}] {name} ({dt:.0f}s) -> {logdir}/{name}.log", flush=True)
            del running[gpu]
        for gpu in GPUS:
            if gpu in running or not pending:
                continue
            name, cmd = pending.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log = open(logdir / f"{name}.log", "a")
            proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
            running[gpu] = (name, proc, time.time(), log)
            print(f"[launch gpu={gpu}] {name}", flush=True)
        time.sleep(5)
    print(f"ALL A2B2 EVAL DONE: {done} ok, {failed} failed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
