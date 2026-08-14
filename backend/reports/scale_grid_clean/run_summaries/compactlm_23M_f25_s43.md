# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_23M_f25_s43`
- Checkpoint: `backend/models/compact_llm/scale_grid_runs_clean/23M/seed_43/f25/sft/compactlm_from_scratch_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `23378304`
- Per-row JSONL: `backend/reports/scale_grid_clean/per_row/compactlm_23M_f25_s43.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 355 (35.50%) | 608 (60.80%) | 198.73 | 202.01 |
