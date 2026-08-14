# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_60M_f25_s42`
- Checkpoint: `<prior-tree-v4>/backend/models/compact_llm/scale_grid_runs/60M/seed_42/f25/sft/compactlm_from_scratch_best.pt`
- Dataset: `<prior-tree-v4>/backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `62112256`
- Per-row JSONL: `<prior-tree-v4>/backend/reports/scale_grid/per_row/compactlm_60M_f25_s42.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 372 (37.20%) | 647 (64.70%) | 390.96 | 416.2 |
