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

Edit `models.json` to define model adapters and paths.

```json
{
  "translation-model-1": {
    "adapter": "opusmt",
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
- `local_files_only` (bool)
- `ct2_model_path` (optional path to a converted CT2 model directory)
- `ct2_cache_dir` (optional cache root for converted CT2 models)
- `tokenizer_path` (optional tokenizer source, defaults to `model_path`)
- `quantization` (default: `float32`)
- `compute_type` (default: `default`)
- `inter_threads` (default: `1`)
- `num_threads` (maps to CT2 `intra_threads`)

All runtime adapters are now CT2-backed:
- `opusmt`
- `transformers` (used for M2M100)
- `nllb`

If you want to use a different config path, set:

```bash
export MODELS_CONFIG_PATH=./models.json
```

## Fine-tuned OPUS models from Hugging Face

The eval API can pull your fine-tuned OPUS models directly from Hugging Face on first use.
No separate download script is required as long as the model entry has:
- `model_path` set to a Hugging Face repo id
- `local_files_only` set to `false`
- `adapter` set to `opusmt`

These model IDs are configured in `models.json`:
- `opus-mt-tc-big-en-de-military-v1` -> `MariusBerg/opus-tc-big-en-de-military-v1`
- `opus-mt-tc-big-en-nob-military` -> `MariusBerg/opus-tc-big-en-nob-military`
- `opus-mt-tc-big-en-pt-military` -> `MariusBerg/opus-tc-big-en-pt-military`
- `m2m-100-1.2b` -> `facebook/m2m100_1.2B`
- `m2m-100-418m` -> `facebook/m2m100_418M`
- `nllb-200-distilled-600m` -> `facebook/nllb-200-distilled-600M`

The first request against each model downloads tokenizer/model files and converts the
checkpoint to CTranslate2 under `src/eval_api/models/ct2/` (or your configured CT2 cache dir).
Later requests load the converted CT2 model from cache.

When adding a model that has specific configuration requirements, extend the adapter logic in `app/adapters/transformers.py` and register the adapter name in `app/registry.py`.

## Run locally

```bash
uv run uvicorn app.main:app
```

API docs are available at `/docs`.

## Run with Docker

First, make sure Docker is installed locally:

```bash
docker -v
```

Then build the Docker image and start the container using Docker Compose in detached mode:

```bash
docker compose up -d
```

If you want to use gated Hugging Face models such as `Unbabel/wmt22-cometkiwi-da`,
set a local Hugging Face read token before starting Docker:

```bash
export HF_TOKEN=hf_your_read_token
docker compose up -d
```

## Example request

```json
{
  "src_lang": "no",
  "tgt_lang": "en",
  "source": "Hold posisjon ved broen. Ingen fiende i sikte.",
  "model_id": "translation-model-1"
}
```

Response:

```json
{
  "model_id": "translation-model-1",
  "translated_value": "Hold position at the bridge. No enemy in sight.",
  "latency_ms": 147
}
```

## Evaluate endpoint

`/evaluate` translates a batch of inputs and scores the outputs against references.

Request:

```json
{
  "model_id": "translation-model-1",
  "src_lang": "no",
  "tgt_lang": "en",
  "metrics": ["bleu", "chrf", "ter", "comet"],
  "items": [
    {
      "item_id": "example-1",
      "source": "Hold posisjon ved broen. Ingen fiende i sikte.",
      "reference": "Hold position at the bridge. No enemy in sight."
    }
  ]
}
```

Response:

```json
{
  "model_id": "translation-model-1",
  "src_lang": "no",
  "tgt_lang": "en",
  "results": [
    {
      "item_id": "example-1",
      "source": "Hold posisjon ved broen. Ingen fiende i sikte.",
      "reference": "Hold position at the bridge. No enemy in sight.",
      "translated_value": "Hold position at the bridge. No enemy in sight.",
      "latency_ms": 147,
      "pure_inference_latency_ms": 132,
      "cpu_percent_per_core": 42.1,
      "ram_mean_mb": 1250.4,
      "ram_peak_mb": 1275.9,
      "metrics": {
        "bleu": 100.0,
        "chrf": 100.0,
        "ter": 0.0,
        "comet": 1.0
      }
    }
  ],
  "aggregates": {
    "bleu_mean": 100.0,
    "bleu_median": 100.0,
    "bleu_stdev": 0.0,
    "chrf_mean": 100.0,
    "chrf_median": 100.0,
    "chrf_stdev": 0.0,
    "ter_mean": 0.0,
    "ter_median": 0.0,
    "ter_stdev": 0.0,
    "comet_mean": 1.0,
    "comet_median": 1.0,
    "comet_stdev": 0.0,
    "cpu_percent_per_core_mean": 42.1,
    "cpu_percent_per_core_median": 42.1,
    "cpu_percent_per_core_stdev": 0.0,
    "ram_mean_mb_mean": 1250.4,
    "ram_mean_mb_median": 1250.4,
    "ram_mean_mb_stdev": 0.0,
    "ram_peak_mb_mean": 1275.9,
    "ram_peak_mb_median": 1275.9,
    "ram_peak_mb_stdev": 0.0
  },
  "average_latency_ms": 147.0,
  "average_pure_inference_latency_ms": 132.0,
  "baseline_rss_mb": 1234.5
}
```

Notes:
- If `metrics` is omitted, the API computes `bleu`, `chrf`, `ter`, and `comet`.
- `cometkiwi` can be requested explicitly for reference-free scoring.
- `latency_ms` is translation-only. The full `/evaluate` request takes longer when metrics are enabled.

## Metrics reported and how we compute them

Translation metrics:
- `bleu`, `chrf`, `ter` are sentence-level scores from `sacrebleu` (computed on the model output vs. reference).
- `comet` uses COMET reference-based scoring on the model output vs. reference.
- `cometkiwi` uses COMET reference-free scoring on the model output (no reference required).

Resource metrics (translation-only):
- `latency_ms` is profiled wall-clock time in the isolated worker (translation wrapped with memory sampling, no metric computation).
- `pure_inference_latency_ms` is the direct timer around `adapter.translate(...)` for the same item.
- `cpu_percent_per_core` is computed from CPU time deltas: `(user + system CPU seconds) / wall_seconds / logical_cores * 100`.
- `ram_mean_mb` and `ram_peak_mb` are the mean/peak of sampled RSS during translation, minus the pre-translation baseline RSS.
- `baseline_rss_mb` is the RSS right after the model is loaded in the worker (idle footprint).
- `user_cpu_ms` is the time spent by the CPU executing the code (doing inference). On a multicore system, this is the sum of the times spent across all cores, and can therefore exceed wall time.
- `system_cpu_ms` is the time spent by the CPU in kernel mode, e.g. for I/O operations.
- `input_tokens` and `output_tokens`  are token counts from the tokenizer.
- `total_tokens_per_second` is `(input_tokens + output_tokens) / wall time`, i.e. a measurement of the overall throughput. 
- `output_tokens_per_second` is `output_tokens / wall time`, which is the text generation speed.
- `ctx_switches_involuntary` is the number of involuntary context switches during translation.

Aggregates:
- For each translation quality and resource metric above, we report `mean`, `median`, and `stdev` across the batch.
- For `ctx_switches_involuntary` we report the sum and max across the batch.
- For `latency_ms` and `pure_inference_latency_ms`, we report average and tail latency (`p50`, `p95` and `p99`) across the batch, plus `min` and `max`.

## COMET configuration

The COMET models are resolved at runtime. You can override them via env vars:

```bash
export COMET_MODEL_NAME="Unbabel/wmt22-comet-da"
export COMET_KIWI_MODEL_NAME="Unbabel/wmt22-cometkiwi-da"
export COMET_BATCH_SIZE=8
export COMET_GPUS=0
export COMET_NUM_WORKERS=1
```

`Unbabel/wmt22-cometkiwi-da` is a gated Hugging Face model. To use `cometkiwi`:
- request/accept access on the Hugging Face model page
- create a Hugging Face user access token with `Read` scope
- expose it to the backend as `HF_TOKEN`

Without that token, `cometkiwi` requests will fail with a Hugging Face `401` / gated repo error.

## Resource profiling

Resource profiling is always on for `/evaluate`, but you can tune it via env vars:

```bash
export EVAL_MEM_INTERVAL=0.1
export EVAL_MEM_BACKEND=psutil
export EVAL_WARMUP_ITEMS=1
```

Translation is run in a separate process so COMET/metrics do not affect CPU/RAM numbers.

If `COMET_MODEL_NAME` or `COMET_KIWI_MODEL_NAME` points at a local checkpoint path, that file will be loaded directly.

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
