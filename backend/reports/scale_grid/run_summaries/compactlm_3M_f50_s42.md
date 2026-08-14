# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_3M_f50_s42`
- Checkpoint: `<prior-tree-v4>/backend/models/compact_llm/scale_grid_runs/3M/seed_42/f50/sft/compactlm_from_scratch_best.pt`
- Dataset: `<prior-tree-v4>/backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `2968800`
- Per-row JSONL: `<prior-tree-v4>/backend/reports/scale_grid/per_row/compactlm_3M_f50_s42.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 161 (16.10%) | 237 (23.70%) | 152.63 | 161.75 |
