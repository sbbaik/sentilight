# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_23M_f50_s44`
- Checkpoint: `backend/models/compact_llm/scale_grid_runs_clean/23M/seed_44/f50/sft/compactlm_from_scratch_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `23378304`
- Per-row JSONL: `backend/reports/scale_grid_clean/per_row/compactlm_23M_f50_s44.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 333 (33.30%) | 561 (56.10%) | 202.34 | 211.29 |
