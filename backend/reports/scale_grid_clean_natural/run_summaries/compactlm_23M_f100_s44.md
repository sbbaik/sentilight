# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_23M_f100_s44`
- Checkpoint: `backend/models/compact_llm/scale_grid_runs_clean/23M/seed_44/f100/sft/compactlm_from_scratch_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `natural`
- Rows: `1000`
- Parameters: `23378304`
- Per-row JSONL: `backend/reports/scale_grid_clean_natural/per_row/compactlm_23M_f100_s44.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 85 (8.50%) | 339 (33.90%) | 210.21 | 215.99 |
