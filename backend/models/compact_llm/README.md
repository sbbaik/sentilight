# Sentilight CompactLM FP32 source model

This directory is the source of truth for CompactLM training, evaluation, and
FP32 serving artifacts inside `New_Android_v3/backend/models`.

`compact_llm_q4/` must not own pretraining or finetuning logic. It should only
consume a validated FP32 checkpoint from this directory and produce lightweight
deployment artifacts.

## Dataset

The reusable smart bulb control dataset is written to:

```text
compact_llm/datasets/sentilight_color_emotion_v1/
  train.jsonl
  val.jsonl
  test.jsonl
  required_validation.jsonl
  pretrain_corpus.txt
  pretrain_tokens.npy
  metadata.json
  source_manifest.json
  labeling_policy.json
  dataset_card.md
```

The dataset preserves the final training inputs for reuse by other models.
Color words such as red/yellow/blue act as hue anchors, while emotion and
intensity expressions vary saturation, brightness, dimmer, and color
temperature.

## Build Dataset

Run these commands from `New_Android_v3/backend/models`:

```bash
conda run -n sentilight python compact_llm/prepare_dataset.py
conda run -n sentilight python compact_llm/snapshot_sft_dataset.py
conda run -n sentilight python compact_llm/build_mixed_pretrain_corpus.py
conda run -n sentilight python compact_llm/build_mixed_pretrain_tokens.py
conda run -n sentilight python compact_llm/snapshot_pretrain_dataset.py
```

The final dataset is already preserved under `compact_llm/datasets/`. Rebuild
it only when intentionally regenerating data from the upstream KOTE sources.

## Pretrain

Pretraining uses `pretrain_corpus.txt` / `pretrain_tokens.npy` to teach Korean
emotion, color, and smart bulb control domain language.

```bash
conda run -n sentilight python compact_llm/pretrain_from_scratch.py \
  --tokens-npy compact_llm/datasets/pretrain_runs/LATEST/pretrain_tokens.npy \
  --devices 0 \
  --steps 2400 \
  --batch-size 64 \
  --block-size 256
```

## Finetune

Finetuning uses supervised rows mapping user utterances to
`H/S/B/Dimmer/CT` JSON.

```bash
conda run -n sentilight python compact_llm/finetune_from_pretrain.py \
  --pretrained-checkpoint compact_llm/from_scratch_runs/compactlm_pretrain_best.pt \
  --train-jsonl compact_llm/datasets/sentilight_color_emotion_v1/train.jsonl \
  --val-jsonl compact_llm/datasets/sentilight_color_emotion_v1/val.jsonl \
  --devices 0 \
  --epochs 4 \
  --batch-size 64
```

## Evaluate And Export

```bash
conda run -n sentilight python -m unittest discover -s compact_llm/tests -p 'test_*.py'

conda run -n sentilight python compact_llm/evaluate.py \
  --device cuda:0 \
  --checkpoint from_scratch_runs/finetune/compactlm_from_scratch_best.pt \
  --report-path compact_llm/eval_report_color_emotion_v1.json

conda run -n sentilight python compact_llm/export_final_checkpoint.py \
  --input-checkpoint compact_llm/from_scratch_runs/finetune/compactlm_from_scratch_best.pt
```

The exported serving checkpoint is:

```text
compact_llm/checkpoint/sentilight_compactlm_final.pt
```
