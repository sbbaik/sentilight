# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_23M_f100_s43`
- Checkpoint: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v4/backend/models/compact_llm/scale_grid_runs/23M/seed_43/f100/sft/compactlm_from_scratch_best.pt`
- Dataset: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v4/backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `23378304`
- Per-row JSONL: `/home/sbbaik/codex_work/multibulb_sentilight/New_Android_v4/backend/reports/scale_grid/per_row/compactlm_23M_f100_s43.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 328 (32.80%) | 545 (54.50%) | 199.86 | 211.59 |
