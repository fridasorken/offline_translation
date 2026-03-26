# Product API

Customer flow stays the same:

- call `/initialize` to load the requested language pair into memory
- call `/translate` after initialization

The Docker build now downloads and converts all configured model artifacts into the image,
so runtime initialization can load from local disk instead of downloading from Hugging Face.

## Build

```bash
docker build -t product-api ./product
```

## Run

```bash
docker run --rm -p 8000:8000 product-api
```

## Example

```bash
curl -X POST http://localhost:8000/initialize \
  -H "Content-Type: application/json" \
  -d '{"language":"nob"}'

curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"sender":true,"text":"Hello world"}'
```
