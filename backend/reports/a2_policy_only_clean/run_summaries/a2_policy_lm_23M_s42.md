# CompactLM Direct Checkpoint Evaluation

- Model: `a2_policy_lm_23M_s42`
- Checkpoint: `backend/models/compact_llm/a2_policy_only_runs_clean/23M/seed_42/compactlm_from_scratch_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `23378304`
- Per-row JSONL: `backend/reports/a2_policy_only_clean/per_row/a2_policy_lm_23M_s42.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 507 (50.70%) | 797 (79.70%) | 215.88 | 219.9 |
