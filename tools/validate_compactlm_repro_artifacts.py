from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer


DEFAULT_TOKENIZER = Path("backend/models/compact_llm/tokenizer/tokenizer.json")
DEFAULT_CORPUS = Path("backend/models/compact_llm/datasets/pretrain_runs/LATEST/pretrain_corpus.txt")
DEFAULT_TOKENS = Path("backend/models/compact_llm/datasets/pretrain_runs/LATEST/pretrain_tokens.npy")
DEFAULT_REPORT_JSON = Path("results/compactlm_repro_artifacts.json")
DEFAULT_REPORT_MD = Path("results/compactlm_repro_artifacts.md")
SPECIAL_TOKENS = (
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|sentilight|>",
    "<|json|>",
    "<|tasmota|>",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_model_type(tokenizer_path: Path) -> str:
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    model = data.get("model") or {}
    return str(model.get("type") or "unknown")


def encode_corpus(tokenizer: Tokenizer, corpus_path: Path, max_lines: int | None) -> tuple[np.ndarray, dict[str, Any]]:
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    ids: list[int] = []
    line_count = 0
    nonempty_count = 0
    with corpus_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if max_lines is not None and line_count >= max_lines:
                break
            line_count += 1
            text = raw_line.rstrip("\n")
            if not text.strip():
                continue
            nonempty_count += 1
            if bos_id is not None:
                ids.append(int(bos_id))
            ids.extend(int(value) for value in tokenizer.encode(text).ids)
            if eos_id is not None:
                ids.append(int(eos_id))
    return np.asarray(ids, dtype=np.uint32), {
        "lines_scanned": line_count,
        "nonempty_lines": nonempty_count,
        "encoded_tokens": len(ids),
        "used_bos": bos_id is not None,
        "used_eos": eos_id is not None,
        "max_lines": max_lines,
    }


def compare_tokens(regenerated: np.ndarray, saved_path: Path, max_lines: int | None) -> dict[str, Any]:
    saved = np.load(saved_path, mmap_mode="r")
    if max_lines is not None:
        limit = min(len(regenerated), len(saved))
        saved_slice = np.asarray(saved[:limit], dtype=np.uint32)
        regenerated_slice = regenerated[:limit]
        equal_count = int(np.count_nonzero(saved_slice == regenerated_slice))
        return {
            "mode": "prefix",
            "saved_tokens": int(len(saved)),
            "regenerated_tokens": int(len(regenerated)),
            "compared_tokens": int(limit),
            "equal_tokens": equal_count,
            "token_match_rate": equal_count / limit if limit else 0.0,
            "exact_match": bool(limit == len(regenerated) and equal_count == limit),
        }
    saved_array = np.asarray(saved, dtype=np.uint32)
    if len(saved_array) != len(regenerated):
        equal_count = int(np.count_nonzero(saved_array[: min(len(saved_array), len(regenerated))] == regenerated[: min(len(saved_array), len(regenerated))]))
        return {
            "mode": "full",
            "saved_tokens": int(len(saved_array)),
            "regenerated_tokens": int(len(regenerated)),
            "compared_tokens": int(min(len(saved_array), len(regenerated))),
            "equal_tokens": equal_count,
            "token_match_rate": equal_count / min(len(saved_array), len(regenerated)) if min(len(saved_array), len(regenerated)) else 0.0,
            "exact_match": False,
        }
    equal = saved_array == regenerated
    equal_count = int(np.count_nonzero(equal))
    return {
        "mode": "full",
        "saved_tokens": int(len(saved_array)),
        "regenerated_tokens": int(len(regenerated)),
        "compared_tokens": int(len(saved_array)),
        "equal_tokens": equal_count,
        "token_match_rate": equal_count / len(saved_array) if len(saved_array) else 0.0,
        "exact_match": bool(equal_count == len(saved_array)),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    regenerated, corpus_stats = encode_corpus(tokenizer, args.corpus, args.max_lines)
    comparison = compare_tokens(regenerated, args.tokens, args.max_lines)
    token_ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    missing_special_tokens = [token for token, token_id in token_ids.items() if token_id is None]
    artifacts = {
        "tokenizer": {
            "path": str(args.tokenizer),
            "sha256": sha256_file(args.tokenizer),
            "model_type": tokenizer_model_type(args.tokenizer),
            "vocab_size": tokenizer.get_vocab_size(),
            "special_token_ids": token_ids,
            "missing_special_tokens": missing_special_tokens,
        },
        "corpus": {
            "path": str(args.corpus),
            "sha256": sha256_file(args.corpus),
            **corpus_stats,
        },
        "tokens": {
            "path": str(args.tokens),
            "sha256": sha256_file(args.tokens),
            **comparison,
        },
    }
    return {
        "artifacts": artifacts,
        "full_validation": args.max_lines is None,
        "pass": (
            args.max_lines is None
            and
            tokenizer.get_vocab_size() == args.expected_vocab_size
            and not missing_special_tokens
            and comparison["exact_match"]
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    tok = report["artifacts"]["tokenizer"]
    corpus = report["artifacts"]["corpus"]
    tokens = report["artifacts"]["tokens"]
    lines = [
        "# CompactLM Reproducibility Artifact Validation",
        "",
        f"- Overall pass: `{report['pass']}`",
        f"- Full validation: `{report['full_validation']}`",
        f"- Tokenizer type: `{tok['model_type']}`",
        f"- Vocab size: `{tok['vocab_size']}`",
        f"- Missing special tokens: `{', '.join(tok['missing_special_tokens']) if tok['missing_special_tokens'] else 'none'}`",
        f"- Corpus lines scanned: `{corpus['lines_scanned']}`",
        f"- Nonempty corpus lines: `{corpus['nonempty_lines']}`",
        f"- Regenerated tokens: `{tokens['regenerated_tokens']}`",
        f"- Saved tokens: `{tokens['saved_tokens']}`",
        f"- Token exact match: `{tokens['exact_match']}`",
        f"- Token match rate: `{tokens['token_match_rate']:.6%}`",
        "",
        "| Artifact | Path | SHA256 |",
        "|---|---|---|",
        f"| tokenizer | `{tok['path']}` | `{tok['sha256']}` |",
        f"| corpus | `{corpus['path']}` | `{corpus['sha256']}` |",
        f"| tokens | `{tokens['path']}` | `{tokens['sha256']}` |",
        "",
        "Special token IDs:",
        "",
        "| Token | ID |",
        "|---|---:|",
    ]
    for token, token_id in tok["special_token_ids"].items():
        lines.append(f"| `{token}` | {token_id} |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate preserved CompactLM tokenizer/corpus/token reproducibility artifacts")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--expected-vocab-size", type=int, default=24000)
    parser.add_argument("--max-lines", type=int, default=None, help="Optional prefix-only validation for smoke tests")
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    return parser.parse_args()


def main() -> None:
    Path("results").mkdir(parents=True, exist_ok=True)
    args = parse_args()
    report = build_report(args)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report), end="")


if __name__ == "__main__":
    main()
