# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_8M_f50_s44`
- Checkpoint: `backend/models/compact_llm/scale_grid_runs_clean/8M/seed_44/f50/sft/compactlm_from_scratch_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `8150208`
- Per-row JSONL: `backend/reports/scale_grid_clean/per_row/compactlm_8M_f50_s44.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 286 (28.60%) | 491 (49.10%) | 197.72 | 205.69 |
