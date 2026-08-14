from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compact_llm.training_data import (
    COLOR_PROFILES,
    EMOTION_PROFILES,
    INTENSITY_MODIFIERS,
    build_dataset_card,
    build_labeling_policy,
    build_source_manifest,
    count_rows_by_key,
    load_jsonl,
    prepare_datasets,
    sha256_file,
    write_jsonl,
)


DEFAULT_BASE_TRAIN = Path("data/external/kote/train_kote.jsonl")   # supply the KOTE source split, or pass --base-train-jsonl
DEFAULT_BASE_VAL = Path("data/external/kote/val_kote.jsonl")     # supply the KOTE source split, or pass --base-val-jsonl
DEFAULT_BASE_TEST = Path("data/external/kote/test_kote.jsonl")    # supply the KOTE source split, or pass --base-test-jsonl
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "datasets" / "sentilight_color_emotion_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CompactLM SFT datasets")
    parser.add_argument("--base-train-jsonl", type=Path, default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--base-val-jsonl", type=Path, default=DEFAULT_BASE_VAL)
    parser.add_argument("--base-test-jsonl", type=Path, default=DEFAULT_BASE_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_train = load_jsonl(args.base_train_jsonl)
    base_val = load_jsonl(args.base_val_jsonl)
    train_rows, val_rows, required_rows = prepare_datasets(
        base_train_rows=base_train,
        base_val_rows=base_val,
        seed=args.seed,
    )

    train_path = args.output_dir / "train.jsonl"
    val_path = args.output_dir / "val.jsonl"
    test_path = args.output_dir / "test.jsonl"
    required_path = args.output_dir / "required_validation.jsonl"
    metadata_path = args.output_dir / "metadata.json"
    manifest_path = args.output_dir / "source_manifest.json"
    policy_path = args.output_dir / "labeling_policy.json"
    card_path = args.output_dir / "dataset_card.md"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(test_path, required_rows)
    write_jsonl(required_path, required_rows)

    metadata = {
        "dataset_version": args.output_dir.name,
        "seed": args.seed,
        "base_train_rows": len(base_train),
        "base_val_rows": len(base_val),
        "prepared_train_rows": len(train_rows),
        "prepared_val_rows": len(val_rows),
        "prepared_test_rows": len(required_rows),
        "required_validation_rows": len(required_rows),
        "color_counts": count_rows_by_key(train_rows + val_rows + required_rows, "base_color"),
        "emotion_counts": count_rows_by_key(train_rows + val_rows + required_rows, "emotion"),
        "intensity_counts": count_rows_by_key(train_rows + val_rows + required_rows, "intensity"),
        "source_counts": count_rows_by_key(train_rows + val_rows + required_rows, "source"),
        "color_profiles": sorted(COLOR_PROFILES),
        "emotion_profiles": sorted(EMOTION_PROFILES),
        "intensity_modifiers": sorted(INTENSITY_MODIFIERS),
        "files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
            "required_validation": str(required_path),
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = build_source_manifest([args.base_train_jsonl, args.base_val_jsonl, args.base_test_jsonl])
    manifest["generated_files"] = {
        "train": {"path": str(train_path), "rows": len(train_rows), "sha256": sha256_file(train_path)},
        "val": {"path": str(val_path), "rows": len(val_rows), "sha256": sha256_file(val_path)},
        "test": {"path": str(test_path), "rows": len(required_rows), "sha256": sha256_file(test_path)},
        "required_validation": {"path": str(required_path), "rows": len(required_rows), "sha256": sha256_file(required_path)},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    policy_path.write_text(json.dumps(build_labeling_policy(), ensure_ascii=False, indent=2), encoding="utf-8")
    card_path.write_text(build_dataset_card(metadata, manifest), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
