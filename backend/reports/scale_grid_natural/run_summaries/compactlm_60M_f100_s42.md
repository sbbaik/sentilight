# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_60M_f100_s42`
- Checkpoint: `backend/models/compact_llm/scale_grid_runs/60M/seed_42/f100/sft/compactlm_from_scratch_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `natural`
- Rows: `1000`
- Parameters: `62112256`
- Per-row JSONL: `backend/reports/scale_grid_natural/per_row/compactlm_60M_f100_s42.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 83 (8.30%) | 329 (32.90%) | 396.29 | 406.78 |
