# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_23M_f25_s44`
- Checkpoint: `<prior-tree-v4>/backend/models/compact_llm/scale_grid_runs/23M/seed_44/f25/sft/compactlm_from_scratch_best.pt`
- Dataset: `<prior-tree-v4>/backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `23378304`
- Per-row JSONL: `<prior-tree-v4>/backend/reports/scale_grid/per_row/compactlm_23M_f25_s44.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 312 (31.20%) | 521 (52.10%) | 202.3 | 215.73 |
