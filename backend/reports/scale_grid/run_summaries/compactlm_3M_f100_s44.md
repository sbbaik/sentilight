# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_3M_f100_s44`
- Checkpoint: `<prior-tree-v4>/backend/models/compact_llm/scale_grid_runs/3M/seed_44/f100/sft/compactlm_from_scratch_best.pt`
- Dataset: `<prior-tree-v4>/backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `2968800`
- Per-row JSONL: `<prior-tree-v4>/backend/reports/scale_grid/per_row/compactlm_3M_f100_s44.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 126 (12.60%) | 247 (24.70%) | 150.48 | 163.15 |
