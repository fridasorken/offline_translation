# Product CLI Prototype

Minimal Opus-only product prototype.

## What it does

- loads one fine-tuned Opus model at startup based on `PRODUCT_TARGET_LANG`
- reuses cached CT2 model if present
- otherwise downloads from Hugging Face and converts once
- runs an interactive stdin loop for translation

## Supported target languages

- `nob`
- `de`
- `pt`

## Build

```bash
docker build -t translation-product-cli ./product
```

## Run interactively

```bash
docker run --rm -it \
  -e PRODUCT_TARGET_LANG=nob \
  -e PRODUCT_MODEL_QUANTIZATION=int8 \
  -v translation_product_cache:/models \
  translation-product-cli
```

Then type sentences into the prompt.

## Example

```bash
docker run --rm -it \
  -e PRODUCT_TARGET_LANG=de \
  -e PRODUCT_MODEL_QUANTIZATION=int8 \
  -v translation_product_cache:/models \
  translation-product-cli
```

## Notes

- first startup can take a while because the model may need to be downloaded and converted
- later runs reuse the mounted `/models` cache volume
- this prototype currently supports only English source text
- use `Ctrl+D` or type `exit` to quit

## Developement:

- Run `pre-commit install` to get correct pre committer (Ruff)
