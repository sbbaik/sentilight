# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_8M_f25_s44`
- Checkpoint: `backend/models/compact_llm/scale_grid_runs_clean/8M/seed_44/f25/sft/compactlm_from_scratch_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `8150208`
- Per-row JSONL: `backend/reports/scale_grid_clean/per_row/compactlm_8M_f25_s44.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 994/1000 | 200 (20.00%) | 316 (31.60%) | 193.95 | 202.46 |
