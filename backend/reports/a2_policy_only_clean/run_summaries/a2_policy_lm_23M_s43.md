# CompactLM Direct Checkpoint Evaluation

- Model: `a2_policy_lm_23M_s43`
- Checkpoint: `backend/models/compact_llm/a2_policy_only_runs_clean/23M/seed_43/compactlm_from_scratch_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `23378304`
- Per-row JSONL: `backend/reports/a2_policy_only_clean/per_row/a2_policy_lm_23M_s43.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 482 (48.20%) | 802 (80.20%) | 211.78 | 216.2 |
