from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(BASE_DIR.parent))

from compact_llm.training_data import (
    AMBIGUOUS_INPUTS,
    COLOR_PROFILES,
    EMOTION_PROFILES,
    INTENSITY_MODIFIERS,
    TARGETED_HARD_CASES,
    build_synthetic_rows,
    load_jsonl,
    sha256_file,
)

DEFAULT_BASE_TRAIN = Path("/home/sbbaik/344_SmartBulb_coding/sentilight_llm_resumable/train_kote.jsonl")
DEFAULT_BASE_VAL = Path("/home/sbbaik/344_SmartBulb_coding/sentilight_llm_resumable/val_kote.jsonl")
DEFAULT_BASE_TEST = Path("/home/sbbaik/codex_work/multibulb_sentilight/Model/fineTune_CompactLLM_KOTE/test_kote.jsonl")
DEFAULT_OUTPUT_DIR = BASE_DIR / "datasets" / "sentilight_color_emotion_v1"
DEFAULT_FINETUNE_DATASET_DIR = DEFAULT_OUTPUT_DIR


COLOR_SEMANTIC_LINES = {
    "red": [
        "빨강은 강렬하고 선명한 붉은 계열의 색이다",
        "빨갛다는 말은 붉은빛이 두드러진다는 뜻이다",
        "새빨간 빛은 강한 에너지와 열기를 떠올리게 한다",
        "은은한 붉은빛은 낮은 밝기의 부드러운 빨강이다",
        "화난 마음의 빨강은 선명하고 강한 빛으로 표현된다",
        "슬픈 마음의 붉은빛은 어둡고 낮은 채도로 표현된다",
    ],
    "yellow": [
        "노랑은 밝고 따뜻한 노란 계열의 색이다",
        "노랗다는 말은 노란빛이 잘 보인다는 뜻이다",
        "밝은 노란색은 기쁨과 활기를 떠올리게 한다",
        "은은한 노란빛은 부드럽고 따뜻한 느낌이다",
        "행복한 노랑은 밝고 포근한 조명으로 표현된다",
        "차분한 노랑은 채도와 밝기를 낮춰 표현된다",
    ],
    "blue": [
        "하늘같이 파란 마음",
        "여름 하늘은 정말 파랗다",
        "맑은 하늘은 파란빛으로 가득하다",
        "푸른 하늘을 보면 차분한 파란색이 떠오른다",
        "청명한 하늘은 선명한 파란색이다",
        "불안한 파랑은 어둡고 차가운 빛으로 표현된다",
    ],
}

HAPPINESS_LINES = [
    "행복하다",
    "기분이 좋고 즐겁다",
    "재미있다",
    "아주 재밌다",
    "즐거운 마음은 밝고 따뜻하다",
]

COLOR_SENTENCE_TEMPLATES = [
    "{label}은 분명한 {label} 느낌이다",
    "{label}은 그 색 자체를 또렷하게 떠올리게 한다",
    "{label}으로 말하면 기본 색감은 {label}이다",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mixed text corpus for CompactLM pretraining from scratch")
    parser.add_argument("--base-train-jsonl", type=Path, default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--base-val-jsonl", type=Path, default=DEFAULT_BASE_VAL)
    parser.add_argument("--base-test-jsonl", type=Path, default=DEFAULT_BASE_TEST)
    parser.add_argument("--finetune-train-jsonl", type=Path, default=DEFAULT_FINETUNE_DATASET_DIR / "train.jsonl")
    parser.add_argument("--finetune-val-jsonl", type=Path, default=DEFAULT_FINETUNE_DATASET_DIR / "val.jsonl")
    parser.add_argument("--finetune-test-jsonl", type=Path, default=DEFAULT_FINETUNE_DATASET_DIR / "test.jsonl")
    parser.add_argument("--finetune-required-jsonl", type=Path, default=DEFAULT_FINETUNE_DATASET_DIR / "required_validation.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def weighted_append(target: list[str], text: str, repeat: int) -> None:
    cleaned = str(text).strip()
    if not cleaned:
        return
    for _ in range(max(1, repeat)):
        target.append(cleaned)


def append_jsonl_inputs(target: list[str], path: Path, repeat: int) -> dict[str, Any]:
    before = len(target)
    rows = 0
    missing_input = 0
    for row in load_jsonl(path):
        rows += 1
        value = row.get("input")
        if value is None:
            missing_input += 1
            continue
        weighted_append(target, str(value), repeat)
    return {
        "path": str(path),
        "rows": rows,
        "missing_input": missing_input,
        "repeat": repeat,
        "lines_added": len(target) - before,
        "sha256": sha256_file(path),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows: list[str] = []
    source_stats: dict[str, Any] = {}

    kote_sources = {
        "base_train": args.base_train_jsonl,
        "base_val": args.base_val_jsonl,
        "base_test": args.base_test_jsonl,
    }
    source_stats["kote_inputs"] = {
        name: append_jsonl_inputs(rows, path, 1)
        for name, path in kote_sources.items()
    }

    finetune_sources = {
        "finetune_train": args.finetune_train_jsonl,
        "finetune_val": args.finetune_val_jsonl,
        "finetune_test": args.finetune_test_jsonl,
        "finetune_required_validation": args.finetune_required_jsonl,
    }
    source_stats["finetune_jsonl_inputs_only"] = {
        name: append_jsonl_inputs(rows, path, 1)
        for name, path in finetune_sources.items()
    }

    synthetic_rows = build_synthetic_rows()
    synthetic_before = len(rows)
    synthetic_by_source: dict[str, int] = {}
    for row in synthetic_rows:
        repeat = {
            "synthetic_emotion": 2,
            "synthetic_color": 4,
            "synthetic_mixed": 5,
            "synthetic_fallback": 2,
            "targeted_hard_case": 18,
            "required_validation": 10,
        }.get(str(row["source"]), 1)
        synthetic_by_source[str(row["source"])] = synthetic_by_source.get(str(row["source"]), 0) + repeat
        weighted_append(rows, row["input"], repeat)
    source_stats["generated_synthetic_inputs"] = {
        "source_row_count": len(synthetic_rows),
        "weighted_lines_added": len(rows) - synthetic_before,
        "weighted_lines_by_source": synthetic_by_source,
    }

    semantic_before = len(rows)
    for lines in COLOR_SEMANTIC_LINES.values():
        for text in lines:
            weighted_append(rows, text, 38)
    for text in HAPPINESS_LINES:
        weighted_append(rows, text, 24)
    for text in AMBIGUOUS_INPUTS:
        weighted_append(rows, text, 16)
    for case in TARGETED_HARD_CASES:
        weighted_append(rows, case["text"], 36)

    for profile in COLOR_PROFILES.values():
        for template in COLOR_SENTENCE_TEMPLATES:
            weighted_append(rows, template.format(label=profile["label"]), 18)
        for phrase in profile["phrases"]:
            weighted_append(rows, phrase, 14)
            for modifier in INTENSITY_MODIFIERS.values():
                if modifier["label"]:
                    weighted_append(rows, f"{modifier['label']} {phrase}", 6)

    for emotion_name, profile in EMOTION_PROFILES.items():
        for phrase in profile["phrases"]:
            weighted_append(rows, phrase, 6 if emotion_name == "happiness" else 4)
    source_stats["handwritten_semantic_inputs"] = {
        "weighted_lines_added": len(rows) - semantic_before,
    }

    rng.shuffle(rows)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "pretrain_corpus.txt"
    report_path = output_dir / "mixed_pretrain_corpus_report.json"
    preview_path = output_dir / "pretrain_corpus_preview.txt"

    corpus_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    preview_path.write_text("\n".join(rows[:200]), encoding="utf-8")
    report = {
        "total_lines": len(rows),
        "unique_lines": len(set(rows)),
        "base_train": str(args.base_train_jsonl),
        "base_val": str(args.base_val_jsonl),
        "base_test": str(args.base_test_jsonl),
        "finetune_train": str(args.finetune_train_jsonl),
        "finetune_val": str(args.finetune_val_jsonl),
        "finetune_test": str(args.finetune_test_jsonl),
        "finetune_required_validation": str(args.finetune_required_jsonl),
        "source_stats": source_stats,
        "pretrain_content_policy": "Only natural/user input text is included. Finetune labels, instructions, and H/S/B/Dimmer/CT outputs are not included.",
        "output_txt": str(corpus_path),
        "sha256": sha256_file(corpus_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = output_dir / "source_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("generated_files", {})["pretrain_corpus"] = {
            "path": str(corpus_path),
            "rows": len(rows),
            "unique_lines": len(set(rows)),
            "sha256": report["sha256"],
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
