from __future__ import annotations

import argparse
import hashlib
import json
from array import array
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_TXT = BASE_DIR / "datasets" / "sentilight_color_emotion_v1" / "pretrain_corpus.txt"
DEFAULT_TOKENIZER = BASE_DIR / "tokenizer" / "tokenizer.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "datasets" / "sentilight_color_emotion_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode mixed pretrain corpus with existing CompactLM tokenizer")
    parser.add_argument("--input-txt", type=Path, default=DEFAULT_INPUT_TXT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    token_ids = array("I")
    lines = 0
    encoded = 0
    with args.input_txt.open("r", encoding="utf-8") as handle:
        for raw in handle:
            lines += 1
            text = raw.strip()
            if not text:
                continue
            ids = tokenizer.encode(text).ids
            if not ids:
                continue
            if bos_id is not None:
                token_ids.append(int(bos_id))
            token_ids.extend(int(value) for value in ids)
            if eos_id is not None:
                token_ids.append(int(eos_id))
            encoded += 1

    token_array = np.frombuffer(token_ids, dtype=np.uint32).copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = args.output_dir / "pretrain_tokens.npy"
    report_path = args.output_dir / "mixed_pretrain_token_report.json"
    np.save(npy_path, token_array)
    report = {
        "lines_read": lines,
        "lines_encoded": encoded,
        "total_tokens": int(token_array.size),
        "tokenizer_path": str(args.tokenizer_path),
        "output_npy": str(npy_path),
        "sha256": sha256_file(npy_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = args.output_dir / "source_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("generated_files", {})["pretrain_tokens"] = {
            "path": str(npy_path),
            "total_tokens": int(token_array.size),
            "sha256": report["sha256"],
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
