# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_3M_f100_s43`
- Checkpoint: `<prior-tree-v4>/backend/models/compact_llm/scale_grid_runs/3M/seed_43/f100/sft/compactlm_from_scratch_best.pt`
- Dataset: `<prior-tree-v4>/backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `2968800`
- Per-row JSONL: `<prior-tree-v4>/backend/reports/scale_grid/per_row/compactlm_3M_f100_s43.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 218 (21.80%) | 308 (30.80%) | 145.49 | 153.96 |
