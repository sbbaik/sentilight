# SentiLight Color Emotion Dataset v1

Smart bulb control dataset for CompactLM and future local lighting models.

## Files

- `train.jsonl`: supervised fine-tuning rows
- `val.jsonl`: validation rows
- `test.jsonl`: required color/emotion validation rows
- `required_validation.jsonl`: explicit acceptance cases
- `pretrain_corpus.txt`: mixed pretraining text corpus
- `pretrain_tokens.npy`: tokenized pretraining corpus
- `metadata.json`: generated counts and splits
- `source_manifest.json`: source/generation checksums
- `labeling_policy.json`: color/emotion/intensity rules

## Counts

- train rows: `81125`
- val rows: `4505`
- test rows: `85`
- required validation rows: `85`

## Sources

{
  "train_kote.jsonl": {
    "path": "/home/sbbaik/344_SmartBulb_coding/sentilight_llm_resumable/train_kote.jsonl",
    "exists": true,
    "rows": 36000,
    "bytes": 12415358,
    "sha256": "f37a350e080e2295f321ba2876f0b33ba62ebca33f917824a97ad378651a5e02"
  },
  "val_kote.jsonl": {
    "path": "/home/sbbaik/344_SmartBulb_coding/sentilight_llm_resumable/val_kote.jsonl",
    "exists": true,
    "rows": 2000,
    "bytes": 691389,
    "sha256": "f4214e9293bec84951bd9ace716d4498f274a41a3d16aef7e95041e427297198"
  },
  "test_kote.jsonl": {
    "path": "/home/sbbaik/codex_work/multibulb_sentilight/Model/fineTune_CompactLLM_KOTE/test_kote.jsonl",
    "exists": true,
    "rows": 2000,
    "bytes": 685608,
    "sha256": "09470a4e23774b815eb0fee846d2e788607f12501c9b599d9df2808c35f33fde"
  }
}
