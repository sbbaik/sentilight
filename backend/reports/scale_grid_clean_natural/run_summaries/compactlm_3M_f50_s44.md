# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_3M_f50_s44`
- Checkpoint: `backend/models/compact_llm/scale_grid_runs_clean/3M/seed_44/f50/sft/compactlm_from_scratch_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `natural`
- Rows: `1000`
- Parameters: `2968800`
- Per-row JSONL: `backend/reports/scale_grid_clean_natural/per_row/compactlm_3M_f50_s44.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 47 (4.70%) | 259 (25.90%) | 150.77 | 153.96 |
