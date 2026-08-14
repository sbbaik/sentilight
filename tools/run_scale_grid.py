from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = Path("backend/models/compact_llm/scale_grid_runs")
DEFAULT_REPORT_ROOT = Path("backend/reports/scale_grid")
DEFAULT_SUBSET_DIR = Path("backend/models/compact_llm/datasets/scale_grid_subsets")
DEFAULT_TOKENS = Path("backend/models/compact_llm/datasets/pretrain_runs/LATEST/pretrain_tokens.npy")
DEFAULT_TOKENIZER = Path("backend/models/compact_llm/tokenizer/tokenizer.json")
DEFAULT_VAL = Path("backend/models/compact_llm/datasets/sft_full_runs/LATEST/val.jsonl")
DEFAULT_DATASET = Path("backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl")
SCALE_CONFIGS = {
    "3M": {"d_model": 96, "n_layer": 6, "n_head": 2, "d_ff": 256},
    "8M": {"d_model": 192, "n_layer": 8, "n_head": 3, "d_ff": 512},
    "23M": {"d_model": 384, "n_layer": 8, "n_head": 6, "d_ff": 1024},
    "60M": {"d_model": 512, "n_layer": 16, "n_head": 8, "d_ff": 1344},
}


@dataclass
class Task:
    name: str
    phase: str
    command: list[str]
    output_dir: Path
    required_paths: list[Path] = field(default_factory=list)
    attempts: int = 0

    @property
    def status_path(self) -> Path:
        return self.output_dir / "status.json"

    @property
    def log_path(self) -> Path:
        return self.output_dir / f"{self.phase}.log"


@dataclass
class RunningTask:
    task: Task
    gpu_id: int
    process: subprocess.Popen[Any]
    log_handle: Any


def write_status(task: Task, status: str, payload: dict[str, Any]) -> None:
    task.output_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "task": task.name,
        "phase": task.phase,
        "status": status,
        **payload,
    }
    task.status_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_status(task: Task) -> dict[str, Any] | None:
    if not task.status_path.exists():
        return None
    return json.loads(task.status_path.read_text(encoding="utf-8"))


def gpu_free_memory_mb() -> dict[int, int]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    result = None
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    for _ in range(3):
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            break
        time.sleep(1)
    if result is None or result.returncode != 0:
        message = result.stderr.strip() if result is not None else "nvidia-smi did not run"
        raise RuntimeError(f"nvidia-smi query failed: {message}")
    free: dict[int, int] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index_s, total_s, used_s = [part.strip() for part in line.split(",")]
        free[int(index_s)] = int(total_s) - int(used_s)
    return free


def runnable_gpu(gpus: list[int], busy: set[int], min_free_mb: int, disable_memory_check: bool) -> int | None:
    if disable_memory_check:
        for gpu_id in gpus:
            if gpu_id not in busy:
                return gpu_id
        return None
    free = gpu_free_memory_mb()
    for gpu_id in gpus:
        if gpu_id in busy:
            continue
        if free.get(gpu_id, 0) >= min_free_mb:
            return gpu_id
    return None


def shell_join(command: list[str]) -> str:
    return " ".join(command)


def launch_task(task: Task, gpu_id: int) -> RunningTask:
    missing = [str(path) for path in task.required_paths if not path.exists()]
    if missing:
        write_status(
            task,
            "failed",
            {
                "reason": "missing required input",
                "missing": missing,
                "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        raise FileNotFoundError(f"{task.name}: missing required input: {missing}")

    task.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_handle = task.log_path.open("a", encoding="utf-8")
    log_handle.write(f"\n# START {time.strftime('%Y-%m-%dT%H:%M:%S%z')} GPU={gpu_id}\n")
    log_handle.write(shell_join(task.command) + "\n")
    log_handle.flush()
    write_status(
        task,
        "running",
        {
            "gpu_id": gpu_id,
            "attempt": task.attempts + 1,
            "command": task.command,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    process = subprocess.Popen(
        task.command,
        cwd=REPO_ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return RunningTask(task=task, gpu_id=gpu_id, process=process, log_handle=log_handle)


def finish_running(running: dict[str, RunningTask], retry_limit: int) -> list[Task]:
    retry: list[Task] = []
    for name, item in list(running.items()):
        return_code = item.process.poll()
        if return_code is None:
            continue
        item.log_handle.write(f"\n# END {time.strftime('%Y-%m-%dT%H:%M:%S%z')} code={return_code}\n")
        item.log_handle.close()
        status = "done" if return_code == 0 else "failed"
        write_status(
            item.task,
            status,
            {
                "gpu_id": item.gpu_id,
                "attempt": item.task.attempts + 1,
                "return_code": return_code,
                "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "log": str(item.task.log_path),
            },
        )
        del running[name]
        if return_code != 0 and item.task.attempts < retry_limit:
            item.task.attempts += 1
            retry.append(item.task)
    return retry


def run_phase(tasks: list[Task], args: argparse.Namespace) -> None:
    if args.dry_run:
        for index, task in enumerate(tasks):
            gpu_id = args.gpus[index % max(1, len(args.gpus))]
            print(f"[dry-run gpu={gpu_id}] {task.name}: CUDA_VISIBLE_DEVICES={gpu_id} {shell_join(task.command)}")
            write_status(
                task,
                "dry_run",
                {
                    "gpu_id": gpu_id,
                    "command": task.command,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
            )
        return

    pending = list(tasks)
    running: dict[str, RunningTask] = {}
    while pending or running:
        pending.extend(finish_running(running, args.retry_failed))
        busy = {item.gpu_id for item in running.values()}
        launched_any = False
        while pending and len(running) < args.max_parallel:
            gpu_id = runnable_gpu(args.gpus, busy, args.min_free_mb, args.disable_memory_check)
            if gpu_id is None:
                break
            task = pending.pop(0)
            existing = read_status(task)
            if existing and existing.get("status") == "done" and not args.force:
                print(f"[skip done] {task.name}")
                continue
            try:
                running[task.name] = launch_task(task, gpu_id)
            except FileNotFoundError as exc:
                print(f"[failed precheck] {exc}")
                if task.attempts < args.retry_failed:
                    task.attempts += 1
                    pending.append(task)
            busy.add(gpu_id)
            launched_any = True
        if not launched_any and running:
            time.sleep(args.poll_seconds)
        elif not launched_any and pending:
            print("[wait] no GPU satisfies memory threshold")
            time.sleep(args.poll_seconds)


def fraction_name(fraction: int) -> str:
    return f"f{fraction}"


def train_path_for_fraction(subset_dir: Path, fraction: int) -> Path:
    return subset_dir / f"train_{fraction}.jsonl"


def build_tasks(args: argparse.Namespace) -> dict[str, list[Task]]:
    py = args.python
    pretrain_tasks: list[Task] = []
    sft_tasks: list[Task] = []
    eval_tasks: list[Task] = []
    for scale_name, cfg in SCALE_CONFIGS.items():
        if args.scales and scale_name not in args.scales:
            continue
        for seed in args.seeds:
            scale_seed_dir = args.run_root / scale_name / f"seed_{seed}"
            pretrain_dir = scale_seed_dir / "pretrain"
            pretrain_ckpt = pretrain_dir / "compactlm_pretrain_best.pt"
            pretrain_tasks.append(
                Task(
                    name=f"pretrain_{scale_name}_s{seed}",
                    phase="pretrain",
                    output_dir=pretrain_dir,
                    required_paths=[args.tokens_npy, args.tokenizer_path],
                    command=[
                        py,
                        "backend/models/compact_llm/pretrain_from_scratch.py",
                        "--tokens-npy",
                        str(args.tokens_npy),
                        "--tokenizer-path",
                        str(args.tokenizer_path),
                        "--output-dir",
                        str(pretrain_dir),
                        "--steps",
                        str(args.pretrain_steps),
                        "--batch-size",
                        str(args.pretrain_batch_size),
                        "--block-size",
                        str(args.block_size),
                        "--learning-rate",
                        str(args.pretrain_lr),
                        "--min-learning-rate",
                        str(args.pretrain_min_lr),
                        "--warmup-steps",
                        str(args.pretrain_warmup_steps),
                        "--seed",
                        str(seed),
                        "--devices",
                        "0",
                        "--d-model",
                        str(cfg["d_model"]),
                        "--n-layer",
                        str(cfg["n_layer"]),
                        "--n-head",
                        str(cfg["n_head"]),
                        "--d-ff",
                        str(cfg["d_ff"]),
                    ],
                )
            )
            for fraction in args.fractions:
                frac_name = fraction_name(fraction)
                train_jsonl = train_path_for_fraction(args.subset_dir, fraction)
                sft_dir = scale_seed_dir / frac_name / "sft"
                sft_ckpt = sft_dir / "compactlm_from_scratch_best.pt"
                model_id = f"compactlm_{scale_name}_{frac_name}_s{seed}"
                report_base = args.report_root / "run_summaries" / f"{model_id}"
                per_row_path = args.report_root / "per_row" / f"{model_id}.jsonl"
                sft_tasks.append(
                    Task(
                        name=f"sft_{model_id}",
                        phase="sft",
                        output_dir=sft_dir,
                        required_paths=[pretrain_ckpt, train_jsonl, args.val_jsonl, args.tokenizer_path],
                        command=[
                            py,
                            "backend/models/compact_llm/finetune_from_pretrain.py",
                            "--train-jsonl",
                            str(train_jsonl),
                            "--val-jsonl",
                            str(args.val_jsonl),
                            "--pretrained-checkpoint",
                            str(pretrain_ckpt),
                            "--tokenizer-path",
                            str(args.tokenizer_path),
                            "--output-dir",
                            str(sft_dir),
                            "--epochs",
                            str(args.sft_epochs),
                            "--batch-size",
                            str(args.sft_batch_size),
                            "--learning-rate",
                            str(args.sft_lr),
                            "--seed",
                            str(seed),
                            "--devices",
                            "0",
                            "--max-seq-len",
                            str(args.block_size),
                        ],
                    )
                )
                if args.sft_max_train_rows is not None:
                    sft_tasks[-1].command.extend(["--max-train-rows", str(args.sft_max_train_rows)])
                if args.sft_max_val_rows is not None:
                    sft_tasks[-1].command.extend(["--max-val-rows", str(args.sft_max_val_rows)])
                eval_tasks.append(
                    Task(
                        name=f"eval_{model_id}",
                        phase="eval",
                        output_dir=report_base,
                        required_paths=[sft_ckpt, args.dataset],
                        command=[
                            py,
                            "tools/evaluate_compactlm_checkpoint.py",
                            "--checkpoint",
                            str(sft_ckpt),
                            "--dataset",
                            str(args.dataset),
                            "--segment",
                            "rule",
                            "--device",
                            "cuda:0",
                            "--model-id",
                            model_id,
                            "--display-name",
                            f"CompactLM {scale_name} {fraction}% seed {seed}",
                            "--per-row-jsonl",
                            str(per_row_path),
                            "--report-json",
                            str(report_base.with_suffix(".json")),
                            "--report-md",
                            str(report_base.with_suffix(".md")),
                            "--progress-every",
                            str(args.eval_progress_every),
                        ],
                    )
                )
                if args.eval_max_rows is not None:
                    eval_tasks[-1].command.extend(["--max-rows", str(args.eval_max_rows)])
    return {"pretrain": pretrain_tasks, "sft": sft_tasks, "eval": eval_tasks}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CompactLM scale/data grid on local GPUs")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR)
    parser.add_argument("--tokens-npy", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--val-jsonl", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--fractions", type=int, nargs="+", default=[25, 50, 100])
    parser.add_argument("--scales", nargs="*", choices=tuple(SCALE_CONFIGS), default=None)
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 0])
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--min-free-mb", type=int, default=24000)
    parser.add_argument("--disable-memory-check", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--retry-failed", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pretrain-steps", type=int, default=2400)
    parser.add_argument("--pretrain-batch-size", type=int, default=64)
    parser.add_argument("--pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--pretrain-min-lr", type=float, default=3e-5)
    parser.add_argument("--pretrain-warmup-steps", type=int, default=120)
    parser.add_argument("--sft-epochs", type=int, default=4)
    parser.add_argument("--sft-batch-size", type=int, default=64)
    parser.add_argument("--sft-lr", type=float, default=1e-4)
    parser.add_argument("--sft-max-train-rows", type=int, default=None)
    parser.add_argument("--sft-max-val-rows", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--eval-max-rows", type=int, default=None)
    parser.add_argument("--eval-progress-every", type=int, default=50)
    parser.add_argument("--phases", nargs="+", choices=("pretrain", "sft", "eval"), default=["pretrain", "sft", "eval"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    tasks_by_phase = build_tasks(args)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scale_configs": SCALE_CONFIGS,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "task_counts": {phase: len(tasks) for phase, tasks in tasks_by_phase.items()},
    }
    (args.report_root / "scale_grid_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for phase in args.phases:
        print(f"[phase] {phase} tasks={len(tasks_by_phase[phase])}")
        run_phase(tasks_by_phase[phase], args)


if __name__ == "__main__":
    main()
