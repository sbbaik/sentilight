from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any
from urllib import error, request

from common.config_loader import load_config


DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[3]
    / "Backend_Models_v2"
    / "compact_llm_q4"
    / "data"
    / "test_mixed_1000.jsonl"
)
DEFAULT_DATASET_REPORT_PATH = DEFAULT_DATASET_PATH.with_name("test_mixed_1000_report.json")
COARSE_CT_MIN = 153
COARSE_CT_MAX = 500
COARSE_DIMENSIONS = ("H", "S", "B", "CT", "Dimmer")


@dataclass(frozen=True)
class ModelDescriptor:
    model_id: str
    display_name: str
    adapter: str
    mode: str


class PerRowPredictionWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "PerRowPredictionWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def load_model_descriptors(config_path: str | Path | None = None) -> list[ModelDescriptor]:
    config = load_config(config_path)
    return [
        ModelDescriptor(
            model_id=model.id,
            display_name=model.name,
            adapter=model.adapter,
            mode=model.mode,
        )
        for model in config.enabled_models
    ]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_dataset_report(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_lighting(data: dict[str, Any]) -> dict[str, int]:
    lower = {str(key).lower(): value for key, value in data.items()}
    return {
        "H": int(round(float(lower["h"]))),
        "S": int(round(float(lower["s"]))),
        "B": int(round(float(lower["b"]))),
        "Dimmer": int(round(float(lower["dimmer"]))),
        "CT": int(round(float(lower["ct"]))),
    }


def hue_distance(a: int, b: int) -> int:
    diff = abs(int(a) - int(b)) % 360
    return min(diff, 360 - diff)


def hue_coarse_bin(hue: int) -> str:
    normalized_hue = int(hue) % 360
    if 0 <= normalized_hue < 60 or 300 <= normalized_hue < 360:
        return "RED"
    if 60 <= normalized_hue < 180:
        return "YELLOW"
    return "BLUE"


def linear_coarse_bin(value: int, min_value: int, max_value: int) -> str:
    if min_value >= max_value:
        raise ValueError("min_value must be less than max_value")
    clamped = min(max(int(value), min_value), max_value)
    segment = (max_value - min_value) / 3.0
    if clamped < min_value + segment:
        return "LOW"
    if clamped < min_value + (2 * segment):
        return "MID"
    return "HIGH"


def binary_coarse_bin(value: int, min_value: int, max_value: int) -> str:
    if min_value >= max_value:
        raise ValueError("min_value must be less than max_value")
    clamped = min(max(int(value), min_value), max_value)
    midpoint = min_value + ((max_value - min_value) / 2.0)
    if clamped <= midpoint:
        return "LOW"
    return "HIGH"


def coarse_bins(values: dict[str, int]) -> dict[str, str]:
    return {
        "H": hue_coarse_bin(values["H"]),
        "S": linear_coarse_bin(values["S"], 0, 100),
        "B": linear_coarse_bin(values["B"], 0, 100),
        "CT": binary_coarse_bin(values["CT"], COARSE_CT_MIN, COARSE_CT_MAX),
        "Dimmer": binary_coarse_bin(values["Dimmer"], 0, 100),
    }


def coarse_3bin_pass(expected: dict[str, int], predicted: dict[str, int]) -> bool:
    expected_bins = coarse_bins(expected)
    predicted_bins = coarse_bins(predicted)
    return all(expected_bins[key] == predicted_bins[key] for key in COARSE_DIMENSIONS)


def strict_semantic_pass(expected: dict[str, int], predicted: dict[str, int], row: dict[str, Any]) -> bool:
    return semantic_pass(expected, predicted, row)


def semantic_pass(expected: dict[str, int], predicted: dict[str, int], row: dict[str, Any]) -> bool:
    if hue_distance(predicted["H"], expected["H"]) > 20:
        return False
    if abs(predicted["S"] - expected["S"]) > 20:
        return False
    if abs(predicted["B"] - expected["B"]) > 20:
        return False
    if abs(predicted["Dimmer"] - expected["Dimmer"]) > 20:
        return False
    if abs(predicted["CT"] - expected["CT"]) > 80:
        return False
    emotion = row.get("emotion")
    base_color = row.get("base_color")
    if emotion == "sadness" and not (predicted["B"] <= 40 and predicted["CT"] >= 340):
        return False
    if emotion == "happiness" and not (predicted["B"] >= 70 and predicted["CT"] <= 290):
        return False
    if emotion == "anger" and not (predicted["S"] >= 75 and predicted["B"] >= 60):
        return False
    if emotion == "calm" and not (30 <= predicted["S"] <= 70 and 45 <= predicted["B"] <= 80):
        return False
    if emotion == "anxiety" and not (predicted["B"] <= 45 and predicted["CT"] >= 340):
        return False
    if base_color is None and emotion is None and not (15 <= predicted["S"] <= 45 and 40 <= predicted["B"] <= 70):
        return False
    return True


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.fmean(values), 2)


def round_metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 2)


def summarize_latency(values: list[float]) -> dict[str, float | None]:
    return {
        "mean_ms": round_metric(safe_mean(values)),
        "p50_ms": round_metric(percentile(values, 0.50)) if values else None,
        "p95_ms": round_metric(percentile(values, 0.95)) if values else None,
        "max_ms": round_metric(max(values)) if values else None,
    }


def http_json(method: str, url: str, payload: dict[str, Any] | None, timeout_seconds: float) -> dict[str, Any]:
    encoded = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {} if payload is None else {"Content-Type": "application/json"}
    req = request.Request(url, data=encoded, headers=headers, method=method)
    with request.urlopen(req, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def build_model_buckets(models: list[ModelDescriptor]) -> dict[str, dict[str, Any]]:
    return {
        model.model_id: {
            "model_id": model.model_id,
            "display_name": model.display_name,
            "adapter": model.adapter,
            "mode": model.mode,
            "successes": 0,
            "errors": 0,
            "semantic_passes": 0,
            "strict_semantic_passes": 0,
            "coarse_3bin_passes": 0,
            "latencies_ms": [],
            "mae_h": [],
            "mae_s": [],
            "mae_b": [],
            "mae_dimmer": [],
            "mae_ct": [],
            "coarse_match_counts": {key: 0 for key in COARSE_DIMENSIONS},
            "coarse_mismatch_counts": {key: 0 for key in COARSE_DIMENSIONS},
            "coarse_failure_dimensions": Counter(),
            "error_samples": [],
        }
        for model in models
    }


def finalize_bucket(bucket: dict[str, Any], total_rows: int) -> dict[str, Any]:
    latencies = bucket["latencies_ms"]
    successes = bucket["successes"]
    strict_passes = bucket["strict_semantic_passes"]
    coarse_passes = bucket["coarse_3bin_passes"]
    coarse_match_rates = {
        key: round(bucket["coarse_match_counts"][key] / successes, 4) if successes else 0.0
        for key in COARSE_DIMENSIONS
    }
    failure_breakdown = {
        key: bucket["coarse_failure_dimensions"].get(key, 0) for key in COARSE_DIMENSIONS
    }
    top_failure_dimension = None
    if any(failure_breakdown.values()):
        top_failure_dimension = max(failure_breakdown, key=failure_breakdown.get)
    return {
        "model_id": bucket["model_id"],
        "display_name": bucket["display_name"],
        "adapter": bucket["adapter"],
        "mode": bucket["mode"],
        "successes": successes,
        "errors": bucket["errors"],
        "success_rate": round(successes / total_rows, 4) if total_rows else 0.0,
        "semantic_passes": strict_passes,
        "semantic_pass_rate": round(strict_passes / total_rows, 4) if total_rows else 0.0,
        "strict_semantic_passes": strict_passes,
        "strict_semantic_pass_rate": round(strict_passes / total_rows, 4) if total_rows else 0.0,
        "coarse_3bin_passes": coarse_passes,
        "coarse_3bin_pass_rate": round(coarse_passes / total_rows, 4) if total_rows else 0.0,
        "coarse_pass_delta": coarse_passes - strict_passes,
        "latency": summarize_latency(latencies),
        "mae": {
            "H": round_metric(safe_mean(bucket["mae_h"])),
            "S": round_metric(safe_mean(bucket["mae_s"])),
            "B": round_metric(safe_mean(bucket["mae_b"])),
            "Dimmer": round_metric(safe_mean(bucket["mae_dimmer"])),
            "CT": round_metric(safe_mean(bucket["mae_ct"])),
        },
        "coarse_match_rates": coarse_match_rates,
        "all_5_coarse_match_rate": round(coarse_passes / successes, 4) if successes else 0.0,
        "coarse_failure_breakdown": failure_breakdown,
        "top_coarse_failure_dimension": top_failure_dimension,
        "error_samples": bucket["error_samples"][:10],
    }


def _record_prediction(
    bucket: dict[str, Any],
    result: dict[str, Any],
    expected: dict[str, int],
    row: dict[str, Any],
) -> dict[str, Any]:
    expected_coarse = coarse_bins(expected)
    if not result.get("success"):
        error_message = result.get("error") or "unknown model error"
        bucket["errors"] += 1
        if len(bucket["error_samples"]) < 10:
            bucket["error_samples"].append(
                {
                    "input": row.get("input", "")[:160],
                    "error": error_message,
                }
            )
        return {
            "success": False,
            "predicted": None,
            "raw": result.get("raw"),
            "error": error_message,
            "strict": False,
            "coarse": False,
            "coarse_expected": expected_coarse,
            "coarse_predicted": None,
            "coarse_mismatch_dimensions": list(COARSE_DIMENSIONS),
            "latency_ms": result.get("latency_ms"),
        }

    bucket["successes"] += 1
    latency = result.get("latency_ms")
    if isinstance(latency, (int, float)):
        bucket["latencies_ms"].append(float(latency))
    predicted = normalize_lighting(result["lighting"])
    bucket["mae_h"].append(float(hue_distance(predicted["H"], expected["H"])))
    bucket["mae_s"].append(abs(predicted["S"] - expected["S"]))
    bucket["mae_b"].append(abs(predicted["B"] - expected["B"]))
    bucket["mae_dimmer"].append(abs(predicted["Dimmer"] - expected["Dimmer"]))
    bucket["mae_ct"].append(abs(predicted["CT"] - expected["CT"]))
    predicted_coarse = coarse_bins(predicted)
    mismatched_dimensions: list[str] = []
    for key in COARSE_DIMENSIONS:
        if expected_coarse[key] == predicted_coarse[key]:
            bucket["coarse_match_counts"][key] += 1
        else:
            bucket["coarse_mismatch_counts"][key] += 1
            mismatched_dimensions.append(key)
    if not mismatched_dimensions:
        bucket["coarse_3bin_passes"] += 1
    else:
        bucket["coarse_failure_dimensions"].update(mismatched_dimensions)
    strict = strict_semantic_pass(expected, predicted, row)
    if strict:
        bucket["semantic_passes"] += 1
        bucket["strict_semantic_passes"] += 1
    return {
        "success": True,
        "predicted": predicted,
        "raw": result.get("raw"),
        "error": None,
        "strict": strict,
        "coarse": not mismatched_dimensions,
        "coarse_expected": expected_coarse,
        "coarse_predicted": predicted_coarse,
        "coarse_mismatch_dimensions": mismatched_dimensions,
        "latency_ms": latency,
    }


def build_per_row_record(
    *,
    mode: str,
    row_index: int,
    row: dict[str, Any],
    model: ModelDescriptor,
    expected: dict[str, int],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    return {
        "mode": mode,
        "row_index": row_index,
        "source": row.get("source"),
        "model_id": model.model_id,
        "display_name": model.display_name,
        "adapter": model.adapter,
        "model_mode": model.mode,
        "input": row.get("input"),
        "emotion": row.get("emotion"),
        "base_color": row.get("base_color"),
        "intensity": row.get("intensity"),
        "reference": expected,
        **prediction,
    }


def evaluate_predict_all(
    server_url: str,
    rows: list[dict[str, Any]],
    models: list[ModelDescriptor],
    timeout_seconds: float,
    progress_every: int,
    per_row_writer: PerRowPredictionWriter | None = None,
) -> dict[str, Any]:
    buckets = build_model_buckets(models)
    request_latencies: list[float] = []
    request_failures: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        started_at = time.perf_counter()
        try:
            response = http_json(
                "POST",
                f"{server_url}/predict_all",
                {"text": row["input"]},
                timeout_seconds,
            )
        except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            request_failures.append({"index": index, "input": row["input"][:160], "error": str(exc)})
            continue
        request_latencies.append((time.perf_counter() - started_at) * 1000)
        expected = normalize_lighting(row["output"])
        by_model = {item.get("model_id"): item for item in response.get("results", []) or []}
        for model in models:
            result = by_model.get(model.model_id)
            if result is None:
                result = {
                    "success": False,
                    "raw": None,
                    "latency_ms": None,
                    "error": "missing model result",
                }
            prediction = _record_prediction(buckets[model.model_id], result, expected, row)
            if per_row_writer is not None:
                per_row_writer.write(
                    build_per_row_record(
                        mode="predict_all",
                        row_index=index,
                        row=row,
                        model=model,
                        expected=expected,
                        prediction=prediction,
                    )
                )
        if progress_every > 0 and index % progress_every == 0:
            print(f"[predict_all] progress {index}/{len(rows)}")

    return {
        "mode": "predict_all",
        "request_latency": summarize_latency(request_latencies),
        "request_failures": request_failures[:20],
        "request_failure_count": len(request_failures),
        "per_model": [finalize_bucket(buckets[model.model_id], len(rows)) for model in models],
    }


def evaluate_per_model(
    server_url: str,
    rows: list[dict[str, Any]],
    models: list[ModelDescriptor],
    timeout_seconds: float,
    progress_every: int,
    per_row_writer: PerRowPredictionWriter | None = None,
) -> dict[str, Any]:
    buckets = build_model_buckets(models)

    for model in models:
        for index, row in enumerate(rows, start=1):
            try:
                result = http_json(
                    "POST",
                    f"{server_url}/predict/{model.model_id}",
                    {"text": row["input"]},
                    timeout_seconds,
                )
            except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                result = {
                    "success": False,
                    "raw": None,
                    "latency_ms": None,
                    "error": str(exc),
                }
                expected = normalize_lighting(row["output"])
                prediction = _record_prediction(buckets[model.model_id], result, expected, row)
                if per_row_writer is not None:
                    per_row_writer.write(
                        build_per_row_record(
                            mode="per_model",
                            row_index=index,
                            row=row,
                            model=model,
                            expected=expected,
                            prediction=prediction,
                        )
                    )
                continue
            expected = normalize_lighting(row["output"])
            prediction = _record_prediction(buckets[model.model_id], result, expected, row)
            if per_row_writer is not None:
                per_row_writer.write(
                    build_per_row_record(
                        mode="per_model",
                        row_index=index,
                        row=row,
                        model=model,
                        expected=expected,
                        prediction=prediction,
                    )
                )
            if progress_every > 0 and index % progress_every == 0:
                print(f"[per_model:{model.model_id}] progress {index}/{len(rows)}")

    return {
        "mode": "per_model",
        "per_model": [finalize_bucket(buckets[model.model_id], len(rows)) for model in models],
    }


def evaluate_dataset(
    server_url: str,
    dataset_path: Path,
    dataset_report_path: Path | None,
    config_path: str | Path | None,
    modes: list[str],
    timeout_seconds: float,
    progress_every: int,
    per_row_jsonl_path: str | Path | None = None,
) -> dict[str, Any]:
    models = load_model_descriptors(config_path)
    rows = load_jsonl(dataset_path)
    dataset_report = load_dataset_report(dataset_report_path)

    mode_results: list[dict[str, Any]] = []
    per_row_context = (
        PerRowPredictionWriter(per_row_jsonl_path)
        if per_row_jsonl_path is not None
        else None
    )
    try:
        for mode in modes:
            if mode == "predict_all":
                mode_results.append(
                    evaluate_predict_all(
                        server_url,
                        rows,
                        models,
                        timeout_seconds,
                        progress_every,
                        per_row_context,
                    )
                )
            elif mode == "per_model":
                mode_results.append(
                    evaluate_per_model(
                        server_url,
                        rows,
                        models,
                        timeout_seconds,
                        progress_every,
                        per_row_context,
                    )
                )
            else:
                raise ValueError(f"Unsupported evaluation mode: {mode}")
    finally:
        if per_row_context is not None:
            per_row_context.close()

    return {
        "dataset": {
            "path": str(dataset_path),
            "report_path": str(dataset_report_path) if dataset_report_path else None,
            "size": len(rows),
            "source_counts": dataset_report.get("source_counts", {}),
            "base_test_source": dataset_report.get("base_test_source"),
        },
        "server_url": server_url,
        "model_ids": [model.model_id for model in models],
        "per_row_jsonl": str(per_row_jsonl_path) if per_row_jsonl_path else None,
        "modes": mode_results,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    dataset = report["dataset"]
    lines.append("# Sentilight 5-Model Benchmark Report")
    lines.append("")
    lines.append(f"- Dataset: `{dataset['path']}`")
    lines.append(f"- Dataset size: `{dataset['size']}`")
    if dataset.get("source_counts"):
        lines.append(f"- Dataset mix: `{json.dumps(dataset['source_counts'], ensure_ascii=False)}`")
    lines.append(f"- Server URL: `{report['server_url']}`")
    lines.append(f"- Model IDs: `{', '.join(report['model_ids'])}`")
    lines.append("- Gemini key source: environment variable only; never stored in report output.")
    lines.append("- Coarse bins: `H/S/B=3 bins`, `CT=2 bins`, `Dimmer=2 bins`")
    lines.append(f"- Coarse CT bins: `LOW={COARSE_CT_MIN}-326`, `HIGH=327-{COARSE_CT_MAX}`")
    lines.append("- Coarse Dimmer bins: `LOW=0-50`, `HIGH=51-100`")
    lines.append("")

    for mode in report["modes"]:
        lines.append(f"## {mode['mode']}")
        lines.append("")
        if mode["mode"] == "predict_all":
            latency = mode["request_latency"]
            lines.append(
                f"- Request latency: mean `{latency['mean_ms']}` ms, "
                f"p50 `{latency['p50_ms']}` ms, p95 `{latency['p95_ms']}` ms, "
                f"max `{latency['max_ms']}` ms"
            )
            lines.append(f"- Request failures: `{mode['request_failure_count']}`")
            lines.append("")
        lines.append(
            "| Model ID | Display Name | Adapter | Mode | Success/Total | Strict Pass | Coarse Pass | Delta | H Match | S Match | B Match | CT Match | Dimmer Match | Mean Latency | P95 Latency | Hue MAE | S MAE | B MAE | Dimmer MAE | CT MAE |"
        )
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        total_rows = dataset["size"]
        for row in mode["per_model"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["model_id"],
                        row.get("display_name") or row["model_id"],
                        row["adapter"],
                        row["mode"],
                        f"{row['successes']}/{total_rows}",
                        f"{row['strict_semantic_passes']} ({row['strict_semantic_pass_rate']:.2%})",
                        f"{row['coarse_3bin_passes']} ({row['coarse_3bin_pass_rate']:.2%})",
                        str(row["coarse_pass_delta"]),
                        f"{row['coarse_match_rates']['H']:.2%}",
                        f"{row['coarse_match_rates']['S']:.2%}",
                        f"{row['coarse_match_rates']['B']:.2%}",
                        f"{row['coarse_match_rates']['CT']:.2%}",
                        f"{row['coarse_match_rates']['Dimmer']:.2%}",
                        str(row["latency"]["mean_ms"]),
                        str(row["latency"]["p95_ms"]),
                        str(row["mae"]["H"]),
                        str(row["mae"]["S"]),
                        str(row["mae"]["B"]),
                        str(row["mae"]["Dimmer"]),
                        str(row["mae"]["CT"]),
                    ]
                )
                + " |"
            )
        lines.append("")
        lines.append("| Model ID | All-5 Match | Top Coarse Failure |")
        lines.append("|---|---:|---|")
        for row in mode["per_model"]:
            top_failure = row["top_coarse_failure_dimension"] or "n/a"
            if top_failure != "n/a":
                top_failure = f"{top_failure} ({row['coarse_failure_breakdown'][row['top_coarse_failure_dimension']]})"
            lines.append(f"| {row['model_id']} | {row['all_5_coarse_match_rate']:.2%} | {top_failure} |")
        lines.append("")
        failures: list[str] = []
        if mode["mode"] == "predict_all":
            failures.extend(
                f"- request[{item['index']}] `{item['error']}`"
                for item in mode["request_failures"][:5]
            )
        for row in mode["per_model"]:
            for sample in row["error_samples"][:2]:
                failures.append(f"- {row['model_id']}: `{sample['error']}`")
        if failures:
            lines.append("### Failure Samples")
            lines.append("")
            lines.extend(failures[:10])
            lines.append("")

    return "\n".join(lines).strip() + "\n"
