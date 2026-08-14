# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_3M_f50_s43`
- Checkpoint: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v5/backend/models/compact_llm/scale_grid_runs_clean/3M/seed_43/f50/sft/compactlm_from_scratch_best.pt`
- Dataset: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v5/backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `2968800`
- Per-row JSONL: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v5/backend/reports/scale_grid_clean/per_row/compactlm_3M_f50_s43.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 153 (15.30%) | 222 (22.20%) | 146.29 | 151.44 |
