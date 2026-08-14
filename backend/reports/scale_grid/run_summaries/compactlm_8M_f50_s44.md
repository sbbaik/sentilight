# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_8M_f50_s44`
- Checkpoint: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v4/backend/models/compact_llm/scale_grid_runs/8M/seed_44/f50/sft/compactlm_from_scratch_best.pt`
- Dataset: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v4/backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `8150208`
- Per-row JSONL: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v4/backend/reports/scale_grid/per_row/compactlm_8M_f50_s44.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 249 (24.90%) | 457 (45.70%) | 197.87 | 212.11 |
