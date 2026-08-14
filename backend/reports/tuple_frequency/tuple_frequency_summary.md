# Tuple Frequency Accuracy Analysis

- Train JSONL: `backend/models/compact_llm/datasets/scale_grid_subsets/train_100.jsonl`
- Per-row dir: `backend/reports/scale_grid/per_row`
- Policy vocab size: `240`
- Train policy unique tuples: `219`
- Train policy rows: `48035`
- Train non-policy rows: `36000`

## 23M/60M 100% Focus

| Scale | Data % | Bucket | Runs | Mean n | Strict mean±std | Coarse mean±std |
|---|---:|---|---:|---:|---:|---:|
| 23M | 100 | top-50 | 3 | 279.0 | 50.06%±4.20% | 74.55%±5.28% |
| 23M | 100 | rank-51-120 | 3 | 315.0 | 37.57%±4.38% | 68.89%±8.96% |
| 23M | 100 | rank-121-240 | 3 | 406.0 | 30.54%±8.52% | 48.11%±9.53% |
| 60M | 100 | top-50 | 3 | 279.0 | 57.35%±4.93% | 83.03%±5.96% |
| 60M | 100 | rank-51-120 | 3 | 315.0 | 41.90%±1.10% | 78.41%±4.44% |
| 60M | 100 | rank-121-240 | 3 | 406.0 | 34.98%±7.02% | 55.91%±7.10% |

## Predicted Tuple Diversity

| Model | Rows | Predicted unique tuples | In policy vocab |
|---|---:|---:|---:|
| compactlm_23M_f100_s42 | 1000 | 304 | 77.60% |
| compactlm_23M_f100_s43 | 1000 | 350 | 60.80% |
| compactlm_23M_f100_s44 | 1000 | 295 | 76.90% |
| compactlm_60M_f100_s42 | 1000 | 233 | 94.80% |
| compactlm_60M_f100_s43 | 1000 | 272 | 78.40% |
| compactlm_60M_f100_s44 | 1000 | 252 | 90.30% |
