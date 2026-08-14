"""Post-hoc robustness check: retrain A2/B2 on the CLEAN pretrain backbone.

NOT part of the hash-frozen preregistration (prereg_contamination_delta.md scopes
the intervention to tools/run_scale_grid.py grid cells only). P4 was adjudicated
on mixed backbones: the generative side was swapped dirty -> clean while A2
(policy-only SFT) and B2 (240-way tuple head) stayed on the contaminated
backbone. This script closes that gap so C1/C2 can be recomputed fully clean.

Each run replicates its original configuration exactly, changing ONE variable:
--pretrained-checkpoint  scale_grid_runs/... -> scale_grid_runs_clean/...

Original configs (verified from the recorded run summaries):
  A2 23M s{42,43,44}: finetune_from_pretrain.py on policy_only/train_policy_48035
      Parent backbone identified empirically by weight distance (no metadata was
      recorded): dirty scale_grid_runs/23M/seed_N/pretrain, rel_dist 0.17 vs
      ~1.41 for every other seed.
  B2 23M s{42,43,44}: train_tuple_classifier.py on scale_grid_subsets/train_100
      (filtered 84035 -> 48035 policy rows on the fly), val sft_full_runs/LATEST
  B2 60M s{42,43,44}: train_tuple_classifier.py on the pre-filtered
      policy_only/train_policy_48035 + val_policy_2665 (same 48035 kept rows)

Restartable: skips any run whose best checkpoint already exists.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
SEEDS = [42, 43, 44]
GPUS = [0, 1]

CLEAN_PRETRAIN = "backend/models/compact_llm/scale_grid_runs_clean/{scale}/seed_{seed}/pretrain/compactlm_pretrain_best.pt"


def build_tasks():
    tasks = []

    # --- A2: policy-only SFT on clean 23M backbone ---
    for seed in SEEDS:
        out = REPO / f"backend/models/compact_llm/a2_policy_only_runs_clean/23M/seed_{seed}"
        best = out / "compactlm_from_scratch_best.pt"
        if best.exists():
            continue
        ckpt = REPO / CLEAN_PRETRAIN.format(scale="23M", seed=seed)
        if not ckpt.exists():
            print(f"[MISSING PRETRAIN] {ckpt}", flush=True)
            continue
        cmd = [
            PY, "backend/models/compact_llm/finetune_from_pretrain.py",
            "--train-jsonl", "backend/models/compact_llm/datasets/policy_only/train_policy_48035.jsonl",
            "--val-jsonl", "backend/models/compact_llm/datasets/policy_only/val_policy_2665.jsonl",
            "--pretrained-checkpoint", str(ckpt.relative_to(REPO)),
            "--output-dir", str(out.relative_to(REPO)),
            "--seed", str(seed),
            "--devices", "0",
        ]
        tasks.append((f"A2_clean_23M_s{seed}", cmd, best))

    # --- B2: 240-way tuple head on clean backbones ---
    for scale in ("23M", "60M"):
        for seed in SEEDS:
            out = REPO / f"backend/models/compact_llm/tuple_head_runs_clean/{scale}/seed_{seed}"
            best = out / "compactlm_tuple_classifier_best.pt"
            if best.exists():
                continue
            ckpt = REPO / CLEAN_PRETRAIN.format(scale=scale, seed=seed)
            if not ckpt.exists():
                print(f"[MISSING PRETRAIN] {ckpt}", flush=True)
                continue
            if scale == "23M":
                train = "backend/models/compact_llm/datasets/scale_grid_subsets/train_100.jsonl"
                val = "backend/models/compact_llm/datasets/sft_full_runs/LATEST/val.jsonl"
            else:
                train = "backend/models/compact_llm/datasets/policy_only/train_policy_48035.jsonl"
                val = "backend/models/compact_llm/datasets/policy_only/val_policy_2665.jsonl"
            cmd = [
                PY, "tools/train_tuple_classifier.py",
                "--train-jsonl", train,
                "--val-jsonl", val,
                "--pretrained-checkpoint", str(ckpt.relative_to(REPO)),
                "--output-dir", str(out.relative_to(REPO)),
                "--seed", str(seed),
                "--devices", "0",
            ]
            tasks.append((f"B2_clean_{scale}_s{seed}", cmd, best))

    return tasks


def main():
    tasks = build_tasks()
    print(f"{len(tasks)} clean-backbone retraining runs", flush=True)
    pending = list(tasks)
    running = {}
    done = failed = 0
    logdir = REPO / "backend/logs/retrain_clean"
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
            name, cmd, _best = pending.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log = open(logdir / f"{name}.log", "a")
            proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
            running[gpu] = (name, proc, time.time(), log)
            print(f"[launch gpu={gpu}] {name}", flush=True)
        time.sleep(5)
    print(f"ALL RETRAIN DONE: {done} ok, {failed} failed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
