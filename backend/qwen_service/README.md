# Qwen comparator service

A standalone FastAPI process that serves the Qwen3-0.6B comparator on port `8103`.

The model is Qwen3-0.6B, LoRA-supervised-fine-tuned and quantized to GGUF Q4_K_M. The
service loads `models/sentilight_qwen3_0_6b_sft_q4km.gguf` by default
(`SENTILIGHT_QWEN_MODEL` overrides it). Weights are distributed separately from this
repository; the training code that produces them is under `training/`.

> `prepare_model.py` still targets an older `sentilight_kote_q4km.gguf` derived from
> Qwen2.5, which this service does **not** load. Use `SENTILIGHT_QWEN_MODEL` or place the
> Qwen3 GGUF at the default path instead.

Install and run:

```bash
cd backend
python -m pip install -r qwen_service/requirements.txt
python -m qwen_service.main
```

Validate:

```bash
curl "http://127.0.0.1:8103/health?load_model=true"
curl -X POST http://127.0.0.1:8103/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"오늘은 마음이 편안해"}'
```

The request text is Korean because the models are trained on Korean input; the example
above means "I feel at ease today."

The main backend service reaches this endpoint through its `remote` adapter.
