# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_8M_f100_s43`
- Checkpoint: `backend/models/compact_llm/scale_grid_runs_clean/8M/seed_43/f100/sft/compactlm_from_scratch_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `8150208`
- Per-row JSONL: `backend/reports/scale_grid_clean/per_row/compactlm_8M_f100_s43.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 358 (35.80%) | 567 (56.70%) | 203.99 | 213.38 |
