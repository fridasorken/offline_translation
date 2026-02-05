# Offline Translation Eval API (Iteration 1)

Minimal local HTTP API for single-input translation with latency reporting.

## Requirements

- Python 3.10+
- `uv`

Install dependencies:

```bash
uv sync
```

## Configure models

Edit `models.json` to point at local model folders (no downloads at runtime).

```json
{
  "translation-model-1": {
    "adapter": "transformers",
    "model_path": "./models/opus-mt-no-en",
    "supported_pairs": [["no", "en"]],
    "num_beams": 4,
    "max_new_tokens": 256,
    "device": "cpu"
  }
}
```

Optional config keys (adapter-specific):
- `num_beams` (int)
- `max_new_tokens` (int)
- `device` ("cpu" or "cuda")
- `forced_bos_token_id` (int)

If you want to use a different config path, set:

```bash
export MODELS_CONFIG_PATH=./models.json
```

## Run

```bash
uv run uvicorn app.main:app
```

API docs are available at `/docs`.
## Example request

```json
{
  "src_lang": "no",
  "tgt_lang": "en",
  "source": "Hold posisjon ved broen. Ingen fiende i sikte.",
  "model_id": "translation-model-1",
  "reference": "Hold position at the bridge. No enemy in sight.",
  "metrics": ["bleu", "chrf", "ter"]
}
```

Response:

```json
{
  "model_id": "translation-model-1",
  "translated_value": "Hold position at the bridge. No enemy in sight.",
  "latency_ms": 147,
  "metrics": {
    "bleu": 100.0,
    "chrf": 100.0,
    "ter": 0.0
  }
}
```

Notes:
- Metrics are only computed when `reference` or `references` are provided.
- If `metrics` is omitted, the API computes all default metrics (`bleu`, `chrf`, `ter`).

## Quick test with a Hugging Face model download

If you want to pull a model directly from Hugging Face once, set `model_path` to a repo id
and allow remote files for that entry:

```json
{
  "translation-model-1": {
    "adapter": "transformers",
    "model_path": "Helsinki-NLP/opus-mt-no-en",
    "supported_pairs": [["no", "en"]],
    "num_beams": 4,
    "max_new_tokens": 256,
    "device": "cpu",
    "local_files_only": false
  }
}
```

The first run will download to the Hugging Face cache. After that, you can switch
`local_files_only` back to `true` to enforce offline loading.

## Repo description

This repo provides a minimal FastAPI service for offline translation evaluation with
pluggable model adapters, a config-driven model registry, and latency reporting. It is
designed to stay small in v1 and scale to batching/metrics in later iterations.
