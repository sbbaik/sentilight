from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = BASE_DIR / "datasets" / "sentilight_color_emotion_v1"
DEFAULT_SNAPSHOT_ROOT = BASE_DIR / "datasets" / "sft_full_runs"
FILES_TO_COPY = [
    "train.jsonl",
    "val.jsonl",
    "test.jsonl",
    "required_validation.jsonl",
    "metadata.json",
    "source_manifest.json",
    "labeling_policy.json",
    "dataset_card.md",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snapshot the full SentiLight SFT dataset for reuse")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name or f"{timestamp}_sentilight_color_adjective_sft_full"
    snapshot_dir = args.snapshot_root / name
    if snapshot_dir.exists():
        raise FileExistsError(f"snapshot already exists: {snapshot_dir}")
    snapshot_dir.mkdir(parents=True)

    copied: dict[str, dict[str, object]] = {}
    for filename in FILES_TO_COPY:
        src = args.dataset_dir / filename
        if not src.exists():
            raise FileNotFoundError(src)
        dst = snapshot_dir / filename
        shutil.copy2(src, dst)
        copied[filename] = {
            "path": str(dst),
            "size_bytes": dst.stat().st_size,
            "sha256": sha256_file(dst),
            "rows": count_jsonl(dst) if dst.suffix == ".jsonl" else None,
        }

    manifest = {
        "snapshot_dir": str(snapshot_dir),
        "dataset_dir": str(args.dataset_dir),
        "created_at": timestamp,
        "policy": "Full supervised finetuning dataset preserved for CompactLM, SBERT, and Qwen training reuse.",
        "copied_files": copied,
    }
    (snapshot_dir / "snapshot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    latest = args.snapshot_root / "LATEST"
    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            raise RuntimeError(f"LATEST exists and is not a symlink/file: {latest}")
    try:
        latest.symlink_to(snapshot_dir.name)
    except OSError:
        latest.write_text(str(snapshot_dir), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
