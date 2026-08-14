# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_8M_f100_s42`
- Checkpoint: `backend/models/compact_llm/scale_grid_runs_clean/8M/seed_42/f100/sft/compactlm_from_scratch_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `8150208`
- Per-row JSONL: `backend/reports/scale_grid_clean/per_row/compactlm_8M_f100_s42.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 335 (33.50%) | 547 (54.70%) | 193.16 | 199.62 |
