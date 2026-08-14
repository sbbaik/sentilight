# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_60M_f25_s42`
- Checkpoint: `backend/models/compact_llm/scale_grid_runs_clean/60M/seed_42/f25/sft/compactlm_from_scratch_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `rule`
- Rows: `1000`
- Parameters: `62112256`
- Per-row JSONL: `backend/reports/scale_grid_clean/per_row/compactlm_60M_f25_s42.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 374 (37.40%) | 613 (61.30%) | 380.55 | 394.76 |
