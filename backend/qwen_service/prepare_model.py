from __future__ import annotations

import argparse
import shutil
from pathlib import Path


TARGET = Path(__file__).resolve().parent / "models" / "sentilight_kote_q4km.gguf"


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy a trained Qwen GGUF into the New runtime")
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    if not source.is_file():
        parser.error(f"GGUF file does not exist: {source}")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, TARGET)
    print(f"Copied {source} to {TARGET}")


if __name__ == "__main__":
    main()
