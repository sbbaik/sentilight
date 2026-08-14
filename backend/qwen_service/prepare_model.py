"""Install a trained Qwen GGUF where the comparator service will find it.

The destination must match what qwen_service/main.py actually loads. That default is
`models/sentilight_qwen3_0_6b_sft_q4km.gguf`, overridable via SENTILIGHT_QWEN_MODEL;
this script resolves the same value so the two cannot drift apart.

Earlier revisions hard-coded `sentilight_kote_q4km.gguf`, a Qwen2.5-derived file the
service never loads, so running this script installed a model that was then ignored.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_TARGET = HERE / "models" / "sentilight_qwen3_0_6b_sft_q4km.gguf"


def resolve_target() -> Path:
    """The path the service will actually read, honouring SENTILIGHT_QWEN_MODEL."""
    override = os.getenv("SENTILIGHT_QWEN_MODEL")
    return Path(override).expanduser() if override else DEFAULT_TARGET


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a trained Qwen GGUF to the path the comparator service loads")
    parser.add_argument("source", type=Path, help="the .gguf file to install")
    parser.add_argument("--target", type=Path, default=None,
                        help="override the destination (defaults to the path the service loads)")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.is_file():
        parser.error(f"GGUF file does not exist: {source}")

    target = (args.target.expanduser() if args.target else resolve_target())
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"Copied {source}\n     -> {target}")
    if target != DEFAULT_TARGET and not os.getenv("SENTILIGHT_QWEN_MODEL"):
        print("WARNING: this is not the path the service loads by default; "
              "set SENTILIGHT_QWEN_MODEL to point at it.")


if __name__ == "__main__":
    main()
