# CompactLM Tuple Classifier Evaluation

- Model: `tuple_head_60M_f100_s42`
- Checkpoint: `backend/models/compact_llm/tuple_head_runs/60M/seed_42/compactlm_tuple_classifier_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `62235136`
- Tuple classes: `240`
- Per-row JSONL: `backend/reports/tuple_head_60M/per_row/tuple_head_60M_f100_s42.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 531 (53.10%) | 841 (84.10%) | 0.3 | 3.4 |
