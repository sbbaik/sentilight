from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_LLAMA_CPP = Path(os.getenv("LLAMA_CPP_DIR", "third_party/llama.cpp"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export merged Qwen3 HF model to FP16 and Q4_K_M GGUF")
    parser.add_argument("--merged-model", type=Path, required=True)
    parser.add_argument("--fp16-out", type=Path, default=Path("qwen_service/training/outputs/qwen3_0_6b_sentilight_fp16.gguf"))
    parser.add_argument("--q4-out", type=Path, default=Path("qwen_service/models/sentilight_qwen3_0_6b_sft_q4km.gguf"))
    parser.add_argument("--llama-cpp-dir", type=Path, default=DEFAULT_LLAMA_CPP)
    parser.add_argument("--quant-type", type=str, default="Q4_K_M")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert = args.llama_cpp_dir / "convert_hf_to_gguf.py"
    quantize = args.llama_cpp_dir / "build" / "bin" / "llama-quantize"
    if not convert.exists():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found: {convert}")
    if not quantize.exists():
        raise FileNotFoundError(f"llama-quantize not found: {quantize}")
    args.fp16_out.parent.mkdir(parents=True, exist_ok=True)
    args.q4_out.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(quantize.parent) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    subprocess.run([sys.executable, str(convert), str(args.merged_model), "--outfile", str(args.fp16_out), "--outtype", "f16"], check=True, env=env)
    subprocess.run([str(quantize), str(args.fp16_out), str(args.q4_out), args.quant_type], check=True, env=env)
    summary = {"merged_model": str(args.merged_model), "fp16_out": str(args.fp16_out), "q4_out": str(args.q4_out), "quant_type": args.quant_type}
    args.q4_out.with_suffix(args.q4_out.suffix + ".meta.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
