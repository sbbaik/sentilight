# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_8M_f50_s43`
- Checkpoint: `backend/models/compact_llm/scale_grid_runs/8M/seed_43/f50/sft/compactlm_from_scratch_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `natural`
- Rows: `1000`
- Parameters: `8150208`
- Per-row JSONL: `backend/reports/scale_grid_natural/per_row/compactlm_8M_f50_s43.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 95 (9.50%) | 295 (29.50%) | 198.02 | 201.17 |
