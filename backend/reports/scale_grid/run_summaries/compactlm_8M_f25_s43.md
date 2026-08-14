# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_8M_f25_s43`
- Checkpoint: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v4/backend/models/compact_llm/scale_grid_runs/8M/seed_43/f25/sft/compactlm_from_scratch_best.pt`
- Dataset: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v4/backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `8150208`
- Per-row JSONL: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v4/backend/reports/scale_grid/per_row/compactlm_8M_f25_s43.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 304 (30.40%) | 473 (47.30%) | 196.67 | 208.86 |
