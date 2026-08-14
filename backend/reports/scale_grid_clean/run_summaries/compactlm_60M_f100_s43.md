# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_60M_f100_s43`
- Checkpoint: `backend/models/compact_llm/scale_grid_runs_clean/60M/seed_43/f100/sft/compactlm_from_scratch_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `62112256`
- Per-row JSONL: `backend/reports/scale_grid_clean/per_row/compactlm_60M_f100_s43.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 391 (39.10%) | 625 (62.50%) | 382.25 | 399.82 |
