from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_TRAIN = Path("backend/models/compact_llm/datasets/sft_full_runs/LATEST/train.jsonl")
DEFAULT_OUTPUT_DIR = Path("backend/models/compact_llm/datasets/scale_grid_subsets")
DEFAULT_FRACTIONS = (0.25, 0.50)
OUTPUT_KEYS = ("H", "S", "B", "Dimmer", "CT")


def load_jsonl_with_lines(path: Path) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append((stripped, json.loads(stripped)))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def row_digest(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stratify_key(row: dict[str, Any]) -> str:
    output = row.get("output") or {}
    key = {
        "source": row.get("source"),
        "base_color": row.get("base_color"),
        "emotion": row.get("emotion"),
        "intensity": row.get("intensity"),
        "output": [int(output[name]) for name in OUTPUT_KEYS],
    }
    return json.dumps(key, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_subsets(rows: list[dict[str, Any]], fractions: tuple[float, ...], seed: int) -> dict[str, list[dict[str, Any]]]:
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[stratify_key(row)].append(row)

    selected_by_fraction: dict[str, list[dict[str, Any]]] = {
        f"{fraction:.2f}": [] for fraction in fractions
    }
    rng = random.Random(seed)
    for key in sorted(strata):
        items = list(strata[key])
        rng.shuffle(items)
        n = len(items)
        for fraction in fractions:
            take = max(1, round(n * fraction))
            selected_by_fraction[f"{fraction:.2f}"].extend(items[:take])

    return selected_by_fraction


def verify_nested(subsets: dict[str, list[dict[str, Any]]], full_rows: list[dict[str, Any]]) -> dict[str, Any]:
    digests = {
        name: {row_digest(row) for row in rows}
        for name, rows in subsets.items()
    }
    digests["1.00"] = {row_digest(row) for row in full_rows}
    ordered_names = sorted(digests, key=float)
    checks: list[dict[str, Any]] = []
    for lower, upper in zip(ordered_names, ordered_names[1:]):
        lower_set = digests[lower]
        upper_set = digests[upper]
        checks.append(
            {
                "lower": lower,
                "upper": upper,
                "lower_count": len(lower_set),
                "upper_count": len(upper_set),
                "is_subset": lower_set.issubset(upper_set),
                "missing_count": len(lower_set - upper_set),
            }
        )
    return {
        "checks": checks,
        "pass": all(check["is_subset"] for check in checks),
    }


def parse_fraction(value: str) -> float:
    fraction = float(value)
    if not (0.0 < fraction < 1.0):
        raise argparse.ArgumentTypeError("fractions must be between 0 and 1, excluding 1.0")
    return fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create nested stratified SFT subsets for scale/data grid experiments")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fractions", type=parse_fraction, nargs="+", default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_jsonl_with_lines(args.train_jsonl)
    rows = [row for _, row in loaded]
    fractions = tuple(sorted(set(args.fractions)))
    subsets = build_subsets(rows, fractions, args.seed)
    output_files: dict[str, str] = {}
    for name, selected in subsets.items():
        percent = int(round(float(name) * 100))
        path = args.output_dir / f"train_{percent}.jsonl"
        write_jsonl(path, selected)
        output_files[name] = str(path)
    full_path = args.output_dir / "train_100.jsonl"
    write_jsonl(full_path, rows)
    output_files["1.00"] = str(full_path)

    verification = verify_nested(subsets, rows)
    stratum_count = len({stratify_key(row) for row in rows})
    manifest = {
        "source_train_jsonl": str(args.train_jsonl),
        "source_rows": len(rows),
        "seed": args.seed,
        "stratification_key": ["source", "base_color", "emotion", "intensity", "output_tuple"],
        "strata": stratum_count,
        "fractions": {
            name: {
                "path": output_files[name],
                "rows": sum(1 for _ in Path(output_files[name]).open("r", encoding="utf-8")),
            }
            for name in sorted(output_files, key=float)
        },
        "nested_verification": verification,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
