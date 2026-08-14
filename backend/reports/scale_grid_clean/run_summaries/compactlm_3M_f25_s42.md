# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_3M_f25_s42`
- Checkpoint: `backend/models/compact_llm/scale_grid_runs_clean/3M/seed_42/f25/sft/compactlm_from_scratch_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `2968800`
- Per-row JSONL: `backend/reports/scale_grid_clean/per_row/compactlm_3M_f25_s42.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 46 (4.60%) | 128 (12.80%) | 151.23 | 157.31 |
