# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_3M_f100_s42`
- Checkpoint: `<prior-tree-v4>/backend/models/compact_llm/scale_grid_runs/3M/seed_42/f100/sft/compactlm_from_scratch_best.pt`
- Dataset: `<prior-tree-v4>/backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `2968800`
- Per-row JSONL: `<prior-tree-v4>/backend/reports/scale_grid/per_row/compactlm_3M_f100_s42.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 145 (14.50%) | 280 (28.00%) | 144.92 | 157.28 |
