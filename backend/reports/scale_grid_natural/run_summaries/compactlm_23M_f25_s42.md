# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_23M_f25_s42`
- Checkpoint: `backend/models/compact_llm/scale_grid_runs/23M/seed_42/f25/sft/compactlm_from_scratch_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `natural`
- Rows: `1000`
- Parameters: `23378304`
- Per-row JSONL: `backend/reports/scale_grid_natural/per_row/compactlm_23M_f25_s42.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 89 (8.90%) | 301 (30.10%) | 209.45 | 217.88 |
