# Sentilight Validation Status

Last updated: 2026-06-20

## Verified

- Backend model registry executes four configured models concurrently and preserves model order.
- Frontend dispatches five successful model results to five bulbs concurrently.
- A model failure or bulb failure is isolated and reported without cancelling other work.
- Duplicate enabled model IDs and duplicate bulb IP assignments are rejected.
- Lighting output is normalized and clamped before building a Tasmota command.
- FastAPI and frontend client communicate successfully in a real local-process dry-run.
- The trained CompactLM SFT checkpoint loads from the deployment folder and generates valid lighting JSON.
- CompactLM CPU measurement on the current machine: about 0.69 s load and 1.41 s for one sample.
- Multilingual SBERT runs fully offline from `models/sbert_emotion/model`.
- SBERT correctly selected joy, calm, anger, and sadness profiles for four Korean validation sentences.
- SBERT CPU measurement after local download: about 2.5 s load and 0.005 s per sample.
- The trained Qwen KOTE GGUF loads through `llama-cpp-python` and returns valid lighting JSON using its original fine-tuning prompt.
- Qwen CPU measurement on the current machine: about 0.19 s load and 0.47-0.74 s per sample.
- A real two-process HTTP integration run connected the main FastAPI server to the Qwen FastAPI service. CompactLM, SBERT, and Qwen succeeded concurrently while an intentionally invalid Gemini key failed independently.
- The frontend real-send path is tested with four concurrent mocked Tasmota HTTP responses.
- Tasmota light settings are sent as one URL-encoded `Backlog0` request per bulb, following the official multi-command format.

## Automated Checks

Run all split-runtime tests:

```bash
cd New
pytest -q
```

Run each side directly for detailed counts:

```bash
cd New/backend && pytest -q
cd New/frontend && pytest -q
```

Run production preflight:

```bash
cd New/backend
curl "http://127.0.0.1:8100/preflight?expected_model_count=4&load_models=true"
```

Run repeated model-service measurements:

```bash
cd New/frontend
python scripts/benchmark_predict_all.py --requests 20
```

Run the 1000-row five-model benchmark from the backend host:

```bash
cd New_Android_v2/backend_compactlm_q8
source ~/.bashrc
python scripts/benchmark_five_models.py \
  --mode predict_all \
  --mode per_model \
  --report-json reports/five_model_benchmark.json \
  --report-md reports/five_model_benchmark.md
```

## Hardware Acceptance Gate

The following external values must be supplied before final hardware acceptance:

1. Set a valid `GEMINI_API_KEY` on the backend server.
2. Start the Qwen/other fourth-model endpoint configured as `qwen2.5_0.5B`.
3. Replace the backend server IP and all five Tasmota IPs in `frontend/config/frontend.yaml`.
4. Confirm `/preflight?expected_model_count=5&load_models=true` returns `ok: true`.
5. Run `verify_four_bulbs` in dry-run mode and inspect all five generated URLs.
6. Run the same command with `--send-real` and visually confirm that all five bulbs change.
7. Run the benchmark and record mean, p95, maximum latency, failure count, semantic pass, and MAE by model.

Linux/macOS:

```bash
cd New/frontend
./scripts/verify_four_bulbs.sh "오늘은 마음이 편안해" --send-real
```

Windows PowerShell:

```powershell
cd New\frontend
.\scripts\verify_four_bulbs.ps1 -Text "오늘은 마음이 편안해" -SendReal
```

Hardware acceptance is not complete until all seven gates pass with the real API key, remote model process, and physical bulbs.

Use `frontend/scripts/discover_tasmota.py --subnet <LAN/CIDR>` when the bulb IPs are unknown.
