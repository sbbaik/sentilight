# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_3M_f25_s44`
- Checkpoint: `<prior-tree-v4>/backend/models/compact_llm/scale_grid_runs/3M/seed_44/f25/sft/compactlm_from_scratch_best.pt`
- Dataset: `<prior-tree-v4>/backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `2968800`
- Per-row JSONL: `<prior-tree-v4>/backend/reports/scale_grid/per_row/compactlm_3M_f25_s44.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 46 (4.60%) | 191 (19.10%) | 146.2 | 156.36 |
