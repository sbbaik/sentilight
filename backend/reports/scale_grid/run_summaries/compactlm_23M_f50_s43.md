# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_23M_f50_s43`
- Checkpoint: `<prior-tree-v4>/backend/models/compact_llm/scale_grid_runs/23M/seed_43/f50/sft/compactlm_from_scratch_best.pt`
- Dataset: `<prior-tree-v4>/backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `23378304`
- Per-row JSONL: `<prior-tree-v4>/backend/reports/scale_grid/per_row/compactlm_23M_f50_s43.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 315 (31.50%) | 575 (57.50%) | 209.54 | 221.27 |
