# CompactLM Tuple Classifier Evaluation

- Model: `tuple_head_23M_f100_s44`
- Checkpoint: `backend/models/compact_llm/tuple_head_runs/23M/seed_44/compactlm_tuple_classifier_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `23470464`
- Tuple classes: `240`
- Per-row JSONL: `backend/reports/tuple_head/per_row/tuple_head_23M_f100_s44.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 529 (52.90%) | 850 (85.00%) | 0.27 | 3.52 |
