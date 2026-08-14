# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_3M_f25_s44`
- Checkpoint: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v5/backend/models/compact_llm/scale_grid_runs/3M/seed_44/f25/sft/compactlm_from_scratch_best.pt`
- Dataset: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v5/backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `natural`
- Rows: `1000`
- Parameters: `2968800`
- Per-row JSONL: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v5/backend/reports/scale_grid_natural/per_row/compactlm_3M_f25_s44.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 0 (0.00%) | 251 (25.10%) | 144.16 | 150.94 |
