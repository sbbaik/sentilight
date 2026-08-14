# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_60M_f50_s43`
- Checkpoint: `backend/models/compact_llm/scale_grid_runs_clean/60M/seed_43/f50/sft/compactlm_from_scratch_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `natural`
- Rows: `1000`
- Parameters: `62112256`
- Per-row JSONL: `backend/reports/scale_grid_clean_natural/per_row/compactlm_60M_f50_s43.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 84 (8.40%) | 314 (31.40%) | 388.29 | 398.74 |
