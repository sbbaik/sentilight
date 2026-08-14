# CompactLM Direct Checkpoint Evaluation

- Model: `a2_policy_lm_23M_s43`
- Checkpoint: `<prior-tree-v4>/backend/models/compact_llm/a2_policy_only_runs/23M/seed_43/compactlm_from_scratch_best.pt`
- Dataset: `<prior-tree-v4>/backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `23378304`
- Per-row JSONL: `<prior-tree-v4>/backend/reports/a2_policy_only/per_row/a2_policy_lm_23M_s43.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 499 (49.90%) | 805 (80.50%) | 212.68 | 222.76 |
