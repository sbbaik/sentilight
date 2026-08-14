# CompactLM Tuple Classifier Evaluation

- Model: `tuple_head_60M_f100_s43`
- Checkpoint: `backend/models/compact_llm/tuple_head_runs_clean/60M/seed_43/compactlm_tuple_classifier_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `62235136`
- Tuple classes: `240`
- Per-row JSONL: `backend/reports/tuple_head_60M_clean/per_row/tuple_head_60M_f100_s43.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 518 (51.80%) | 846 (84.60%) | 0.28 | 3.21 |
