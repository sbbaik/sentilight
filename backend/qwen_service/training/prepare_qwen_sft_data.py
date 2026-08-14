from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "당신은 감정 기반 스마트 조명 제어 모델입니다. "
    "반드시 JSON만 출력하세요. 출력 키는 H,S,B,Dimmer,CT만 사용하세요. "
    "명시적 색상 요청이 있으면 hue는 그 색상을 우선하고, 감정과 강도는 채도/밝기/디머/색온도에 반영하세요."
)


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


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"문장: {row['input']}"},
            {"role": "assistant", "content": json.dumps(row["output"], ensure_ascii=False, separators=(",", ":"))},
        ],
        "source": row.get("source"),
        "base_color": row.get("base_color"),
        "emotion": row.get("emotion"),
        "intensity": row.get("intensity"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert full SentiLight SFT JSONL into Qwen chat SFT data")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("models/qwen"))
    parser.add_argument("--output-dir", type=Path, default=Path("qwen_service/training/data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = [convert_row(row) for row in load_jsonl(args.dataset_dir / "train.jsonl")]
    val = [convert_row(row) for row in load_jsonl(args.dataset_dir / "val.jsonl")]
    required = [convert_row(row) for row in load_jsonl(args.dataset_dir / "required_validation.jsonl")]
    write_jsonl(args.output_dir / "train_qwen_chat.jsonl", train)
    write_jsonl(args.output_dir / "val_qwen_chat.jsonl", val)
    write_jsonl(args.output_dir / "required_qwen_chat.jsonl", required)
    metadata = {
        "base_model": str(args.base_model),
        "dataset_dir": str(args.dataset_dir),
        "train_rows": len(train),
        "val_rows": len(val),
        "required_rows": len(required),
        "format": "messages JSONL for Qwen3 chat-template SFT",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
