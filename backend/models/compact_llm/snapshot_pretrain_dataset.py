from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compact_llm.training_data import sha256_file


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = BASE_DIR / "datasets" / "sentilight_color_emotion_v1"
DEFAULT_RUNS_DIR = BASE_DIR / "datasets" / "pretrain_runs"

SNAPSHOT_FILES = [
    "pretrain_corpus.txt",
    "pretrain_tokens.npy",
    "pretrain_corpus_preview.txt",
    "mixed_pretrain_corpus_report.json",
    "mixed_pretrain_token_report.json",
    "source_manifest.json",
    "dataset_card.md",
    "metadata.json",
    "labeling_policy.json",
]

FINETUNE_FILES = [
    "train.jsonl",
    "val.jsonl",
    "test.jsonl",
    "required_validation.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snapshot the exact CompactLM pretrain corpus/tokens used for a run.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--name", default=None, help="Optional snapshot directory name")
    parser.add_argument("--latest-pointer", default="LATEST")
    return parser.parse_args()


def file_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name or f"{timestamp}_sentilight_color_emotion_pretrain"
    snapshot_dir = (args.runs_dir / name).resolve()
    if snapshot_dir.exists():
        raise FileExistsError(f"snapshot already exists: {snapshot_dir}")

    snapshot_dir.mkdir(parents=True)
    copied: dict[str, dict[str, Any]] = {}
    for filename in SNAPSHOT_FILES:
        src = dataset_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"required pretrain file not found: {src}")
        dst = snapshot_dir / filename
        shutil.copy2(src, dst)
        copied[filename] = file_entry(dst)

    finetune_inputs: dict[str, dict[str, Any]] = {}
    for filename in FINETUNE_FILES:
        src = dataset_dir / filename
        if src.exists():
            finetune_inputs[filename] = file_entry(src)

    manifest = {
        "snapshot_dir": str(snapshot_dir),
        "dataset_dir": str(dataset_dir),
        "created_at": timestamp,
        "policy": "This snapshot preserves the exact corpus and token file used for CompactLM pretraining. Finetune JSONL files are not copied; their hashes are recorded to prove they were not modified.",
        "copied_files": copied,
        "finetune_jsonl_hashes": finetune_inputs,
    }
    manifest_path = snapshot_dir / "pretrain_snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pointer_path = args.runs_dir / args.latest_pointer
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    if pointer_path.exists() or pointer_path.is_symlink():
        pointer_path.unlink()
    try:
        pointer_path.symlink_to(snapshot_dir, target_is_directory=True)
    except OSError:
        pointer_path.write_text(str(snapshot_dir) + "\n", encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
