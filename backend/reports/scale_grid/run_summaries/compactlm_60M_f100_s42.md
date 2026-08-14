# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_60M_f100_s42`
- Checkpoint: `<prior-tree-v4>/backend/models/compact_llm/scale_grid_runs/60M/seed_42/f100/sft/compactlm_from_scratch_best.pt`
- Dataset: `<prior-tree-v4>/backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `62112256`
- Per-row JSONL: `<prior-tree-v4>/backend/reports/scale_grid/per_row/compactlm_60M_f100_s42.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 467 (46.70%) | 740 (74.00%) | 393.9 | 422.29 |
