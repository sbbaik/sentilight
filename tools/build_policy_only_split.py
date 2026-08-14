from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
MODELS_DIR = BACKEND_DIR / "models"
for path in (BACKEND_DIR, MODELS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compact_llm.tuple_classifier import build_tuple_vocab, output_tuple, tuple_to_label_map  # noqa: E402


DEFAULT_TRAIN_IN = Path("backend/models/compact_llm/datasets/scale_grid_subsets/train_100.jsonl")
DEFAULT_VAL_IN = Path("backend/models/compact_llm/datasets/sft_full_runs/LATEST/val.jsonl")
DEFAULT_OUTPUT_DIR = Path("backend/models/compact_llm/datasets/policy_only")
EXPECTED = {
    "train": {"kept": 48_035, "dropped": 36_000},
    "val": {"kept": 2_665, "dropped": 2_000},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build policy-only SFT splits matching the tuple-head filter")
    parser.add_argument("--train-in", type=Path, default=DEFAULT_TRAIN_IN)
    parser.add_argument("--val-in", type=Path, default=DEFAULT_VAL_IN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--skip-count-asserts", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_rows(
    rows: list[dict[str, Any]],
    label_map: dict[tuple[int, int, int, int, int], int],
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    kept_indices: list[int] = []
    kept_by_source: Counter[str] = Counter()
    dropped_by_source: Counter[str] = Counter()
    dropped_examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        source = str(row.get("source") or "unknown")
        key = output_tuple(row["output"])
        if key in label_map:
            kept.append(row)
            kept_indices.append(index)
            kept_by_source[source] += 1
        else:
            dropped_by_source[source] += 1
            if len(dropped_examples) < 10:
                dropped_examples.append(
                    {
                        "row_index": index,
                        "source": source,
                        "input": str(row.get("input", ""))[:120],
                        "output": row.get("output"),
                    }
                )
    return kept, kept_indices, {
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "dropped_rows": len(rows) - len(kept),
        "kept_by_source": dict(sorted(kept_by_source.items())),
        "dropped_by_source": dict(sorted(dropped_by_source.items())),
        "dropped_examples": dropped_examples,
    }


def build_split(name: str, input_path: Path, output_dir: Path, label_map: dict[tuple[int, int, int, int, int], int], skip_asserts: bool) -> dict[str, Any]:
    rows = load_jsonl(input_path)
    kept, indices, summary = filter_rows(rows, label_map)
    out_jsonl = output_dir / f"{name}_policy_{summary['kept_rows']}.jsonl"
    out_indices = output_dir / f"{name}_policy_{summary['kept_rows']}_indices.json"
    write_jsonl(out_jsonl, kept)
    out_indices.write_text(json.dumps(indices, ensure_ascii=False, indent=2), encoding="utf-8")
    expected = EXPECTED[name]
    if not skip_asserts:
        assert summary["kept_rows"] == expected["kept"], f"{name}: expected kept {expected['kept']}, got {summary['kept_rows']}"
        assert summary["dropped_rows"] == expected["dropped"], f"{name}: expected dropped {expected['dropped']}, got {summary['dropped_rows']}"
    return {
        "name": name,
        "input_jsonl": str(input_path),
        "output_jsonl": str(out_jsonl),
        "kept_indices_json": str(out_indices),
        "expected_counts": expected,
        **summary,
    }


def main() -> None:
    args = parse_args()
    tuple_vocab = build_tuple_vocab()
    label_map = tuple_to_label_map(tuple_vocab)
    if len(tuple_vocab) != 240:
        raise RuntimeError(f"Expected 240 policy tuples, got {len(tuple_vocab)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "policy_vocab_size": len(tuple_vocab),
        "filter": "output tuple membership in compact_llm.tuple_classifier.build_tuple_vocab()",
        "train": build_split("train", args.train_in, args.output_dir, label_map, args.skip_count_asserts),
        "val": build_split("val", args.val_in, args.output_dir, label_map, args.skip_count_asserts),
    }
    summary_json = args.summary_json or args.output_dir / "policy_only_split_summary.json"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
