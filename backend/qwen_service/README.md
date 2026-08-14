# Qwen GGUF service

This standalone FastAPI process serves the fourth lighting model on port `8103`.

Prepare the existing trained GGUF model:

```bash
cd New
python backend/qwen_service/prepare_model.py \
  ../App/sentilight_v4_local_Qwen2.5/model_qwen_kote/output/sentilight_kote_q4km.gguf
```

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

The main backend service uses this endpoint through its `remote` adapter.
