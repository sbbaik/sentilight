# CompactLM Tuple Classifier Evaluation

- Model: `tuple_head_23M_f100_s43`
- Checkpoint: `backend/models/compact_llm/tuple_head_runs/23M/seed_43/compactlm_tuple_classifier_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `23470464`
- Tuple classes: `240`
- Per-row JSONL: `backend/reports/tuple_head/per_row/tuple_head_23M_f100_s43.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 537 (53.70%) | 855 (85.50%) | 0.25 | 3.21 |
