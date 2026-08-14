# CompactLM Direct Checkpoint Evaluation

- Model: `compactlm_8M_f25_s44`
- Checkpoint: `backend/models/compact_llm/scale_grid_runs/8M/seed_44/f25/sft/compactlm_from_scratch_best.pt`
- Dataset: `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl`
- Segment: `natural`
- Rows: `1000`
- Parameters: `8150208`
- Per-row JSONL: `backend/reports/scale_grid_natural/per_row/compactlm_8M_f25_s44.jsonl`

| Success | Strict | Coarse | Mean Latency | P95 Latency |
|---:|---:|---:|---:|---:|
| 1000/1000 | 93 (9.30%) | 289 (28.90%) | 194.37 | 198.45 |
