from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compact_llm.training_data import (
    EMOTION_PROFILES,
    REQUIRED_VALIDATION_CASES,
    build_prompt,
    hue_distance,
    resolve_case_output,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = BASE_DIR / "checkpoint" / "sentilight_compactlm_final.pt"
DEFAULT_REPORT = BASE_DIR / "eval_report.json"
CHANNEL_TOLERANCE = {"S": 8, "B": 8, "Dimmer": 8, "CT": 30}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SentiLight CompactLM validation phrases")
    parser.add_argument("--checkpoint", type=str, default="checkpoint/sentilight_compactlm_final.pt")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def expected_output(case: dict[str, Any]) -> dict[str, int]:
    if case.get("emotion") or case.get("base_color") or case.get("intensity"):
        return resolve_case_output(case)
    return dict(EMOTION_PROFILES[case["emotion"]]["target"])


def semantic_pass(case: dict[str, Any], pred: dict[str, Any]) -> tuple[bool, str]:
    expected = expected_output(case)
    try:
        values = {key: int(pred[key]) for key in ("H", "S", "B", "Dimmer", "CT")}
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"invalid output: {exc}"

    hue_ok = hue_distance(values["H"], int(expected["H"])) <= int(case.get("hue_tolerance", 18))
    range_ok = (
        0 <= values["S"] <= 100
        and 0 <= values["B"] <= 100
        and 0 <= values["Dimmer"] <= 100
        and 150 <= values["CT"] <= 500
    )
    if not hue_ok:
        return False, "hue mismatch"
    if not range_ok:
        return False, "value out of range"

    # Explicit color prompts can intentionally override generic emotion hue/temperature
    # expectations, so validate against the resolved color+emotion+intensity policy.
    for key, tolerance in CHANNEL_TOLERANCE.items():
        if abs(values[key] - int(expected[key])) > tolerance:
            return False, f"{key} mismatch"
    return True, "ok"


def main() -> None:
    args = parse_args()
    from compact_llm.inference import load_model, predict

    class Config:
        resolved_model_dir = str(BASE_DIR)
        checkpoint = args.checkpoint
        tokenizer_dir = "tokenizer"
        options = {"device": args.device, "max_new_tokens": 96}

    runtime = load_model(Config)
    report: list[dict[str, Any]] = []
    for case in REQUIRED_VALIDATION_CASES:
        raw_pred = predict(build_prompt(case["text"]), runtime)
        ok, reason = semantic_pass(case, raw_pred)
        report.append(
            {
                "text": case["text"],
                "expected": expected_output(case),
                "predicted": raw_pred,
                "pass": ok,
                "reason": reason,
            }
        )
        print(json.dumps(report[-1], ensure_ascii=False))
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
