# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_60M_f25_s42`
- Checkpoint: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v5/backend/models/compact_llm/scale_grid_runs/60M/seed_42/f25/sft/compactlm_from_scratch_best.pt`
- Dataset: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v5/backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `natural`
- Rows: `1000`
- Parameters: `62112256`
- Per-row JSONL: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v5/backend/reports/scale_grid_natural/per_row/compactlm_60M_f25_s42.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 71 (7.10%) | 309 (30.90%) | 389.94 | 405.31 |
