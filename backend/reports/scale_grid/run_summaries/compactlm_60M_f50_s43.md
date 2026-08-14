# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_60M_f50_s43`
- Checkpoint: `<prior-tree-v4>/backend/models/compact_llm/scale_grid_runs/60M/seed_43/f50/sft/compactlm_from_scratch_best.pt`
- Dataset: `<prior-tree-v4>/backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `62112256`
- Per-row JSONL: `<prior-tree-v4>/backend/reports/scale_grid/per_row/compactlm_60M_f50_s43.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 374 (37.40%) | 636 (63.60%) | 390.36 | 429.42 |
