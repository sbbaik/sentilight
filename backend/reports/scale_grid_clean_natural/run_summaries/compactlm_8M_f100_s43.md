# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_8M_f100_s43`
- Checkpoint: `backend/models/compact_llm/scale_grid_runs_clean/8M/seed_43/f100/sft/compactlm_from_scratch_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `natural`
- Rows: `1000`
- Parameters: `8150208`
- Per-row JSONL: `backend/reports/scale_grid_clean_natural/per_row/compactlm_8M_f100_s43.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 95 (9.50%) | 337 (33.70%) | 196.41 | 204.76 |
