# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_8M_f100_s42`
- Checkpoint: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v5/backend/models/compact_llm/scale_grid_runs/8M/seed_42/f100/sft/compactlm_from_scratch_best.pt`
- Dataset: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v5/backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `natural`
- Rows: `1000`
- Parameters: `8150208`
- Per-row JSONL: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v5/backend/reports/scale_grid_natural/per_row/compactlm_8M_f100_s42.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 75 (7.50%) | 327 (32.70%) | 204.25 | 210.36 |
