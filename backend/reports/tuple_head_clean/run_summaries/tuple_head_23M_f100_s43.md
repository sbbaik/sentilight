# CompactLM Tuple Classifier Evaluation

- Model: `tuple_head_23M_f100_s43`
- Checkpoint: `backend/models/compact_llm/tuple_head_runs_clean/23M/seed_43/compactlm_tuple_classifier_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `23470464`
- Tuple classes: `240`
- Per-row JSONL: `backend/reports/tuple_head_clean/per_row/tuple_head_23M_f100_s43.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 521 (52.10%) | 847 (84.70%) | 0.24 | 3.21 |
