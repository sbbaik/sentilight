from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "from_scratch_runs" / "finetune" / "compactlm_from_scratch_best.pt"
DEFAULT_OUTPUT = BASE_DIR / "checkpoint" / "sentilight_compactlm_final.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CompactLM training checkpoint as FP32 serving checkpoint")
    parser.add_argument("--input-checkpoint", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-checkpoint", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backup-existing", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    if not args.input_checkpoint.exists():
        raise FileNotFoundError(f"input checkpoint not found: {args.input_checkpoint}")

    if args.output_checkpoint.exists() and args.backup_existing:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = args.output_checkpoint.with_name(f"{args.output_checkpoint.stem}.backup_{timestamp}{args.output_checkpoint.suffix}")
        shutil.copy2(args.output_checkpoint, backup_path)
        print(f"backup={backup_path}")

    checkpoint = torch.load(args.input_checkpoint, map_location="cpu")
    serving_payload = {
        "model_state_dict": checkpoint["model_state_dict"],
        "model_config": checkpoint["model_config"],
        "metrics": checkpoint.get("metrics", {}),
        "source_checkpoint": str(args.input_checkpoint),
        "exported_for": "compact_llm_fp32_serving",
    }
    torch.save(serving_payload, args.output_checkpoint)
    print(f"exported={args.output_checkpoint}")


if __name__ == "__main__":
    main()
