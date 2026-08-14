# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_23M_f50_s43`
- Checkpoint: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v5/backend/models/compact_llm/scale_grid_runs_clean/23M/seed_43/f50/sft/compactlm_from_scratch_best.pt`
- Dataset: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v5/backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `natural`
- Rows: `1000`
- Parameters: `23378304`
- Per-row JSONL: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v5/backend/reports/scale_grid_clean_natural/per_row/compactlm_23M_f50_s43.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 73 (7.30%) | 300 (30.00%) | 211.53 | 214.8 |
