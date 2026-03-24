# Product CLI Prototype

Minimal Opus-only product prototype.

## What it does

- loads one Opus model at startup based on `PRODUCT_SOURCE_LANG` + `PRODUCT_TARGET_LANG`
- reuses cached CT2 model if present
- otherwise downloads from Hugging Face and converts once
- runs an interactive stdin loop for translation

## REST API (Compose)

The gateway will only route to running translation services. Requests to language pairs that aren't available will fail at the gateway.

To load English-Norwegian and translate:

```bash
docker compose up -d translate-en-nob translate-nob-en gateway
curl -X POST http://localhost:8080/translate/en/nob \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

## Supported language pairs

- `en -> nob` (fine-tuned)
- `en -> de` (fine-tuned)
- `en -> pt` (fine-tuned)
- `nob -> en` (base Opus `tc-big-gmq-en`)
- `nno -> en` (base Opus `tc-big-gmq-en`)
- `de -> en` (base Opus)
- `pt -> en` (base Opus `ROMANCE-en`)

## Build

```bash
docker build -t translation-product-cli ./product
```

## Run interactively

```bash
docker run --rm -it \
  -e PRODUCT_SOURCE_LANG=en \
  -e PRODUCT_TARGET_LANG=nob \
  -e PRODUCT_MODEL_QUANTIZATION=int8 \
  -v translation_product_cache:/models \
  translation-product-cli
```

Then type sentences into the prompt.

## Example

```bash
docker run --rm -it \
  -e PRODUCT_SOURCE_LANG=en \
  -e PRODUCT_TARGET_LANG=de \
  -e PRODUCT_MODEL_QUANTIZATION=int8 \
  -v translation_product_cache:/models \
  translation-product-cli
```

## Reverse direction example

```bash
docker run --rm -it \
  -e PRODUCT_SOURCE_LANG=de \
  -e PRODUCT_TARGET_LANG=en \
  -e PRODUCT_MODEL_QUANTIZATION=int8 \
  -v translation_product_cache:/models \
  translation-product-cli
```

## Notes

- first startup can take a while because the model may need to be downloaded and converted
- later runs reuse the mounted `/models` cache volume
- `pt -> en` currently uses `Helsinki-NLP/opus-mt-ROMANCE-en` because a dedicated `pt -> en`
  or `tc-big-pt -> en` Opus checkpoint was not available
- use `Ctrl+D` or type `exit` to quit

## Developement:

- Run `pre-commit install` to get correct pre committer (Ruff)
