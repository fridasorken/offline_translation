# Product API

Customer flow stays the same:

- call `/initialize` to load the requested language pair into memory
- call `/translate` after initialization

The Docker build now downloads and converts all configured model artifacts into the image,
so runtime initialization can load from local disk instead of downloading from Hugging Face.

Implementation entrypoints:
- API app: [`app.main`](app/main.py)
- Container start command: [main.py](main.py)
- Image build config: [Dockerfile](Dockerfile)
- Compose service config: [docker-compose.yml](docker-compose.yml)

## Prerequisites

- Docker + Docker Compose plugin installed.
- Port `8000` available on host.

Check installation:

```bash
docker --version
docker compose version
```

## Run the API

From repository root:

```bash
cd product
docker compose up --build
```

What this does:
- Builds image from [product/Dockerfile](product/Dockerfile), which includes downloading the translation models from HuggingFace
- Starts `product-api` service from [product/docker-compose.yml](product/docker-compose.yml)
- Exposes API at http://localhost:8000

## Language Initialization

Example with Norwegian:

```bash
curl -X POST http://localhost:8000/initialize \
  -H "Content-Type: application/json" \
  -d '{"language":"nob"}'
```

Expected response:

```bash
{"status":"ok","language":"nob"}
```

## Translation

Example where the host is the sender (i.e. translation happens from configured language to english):

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"sender":true,"text":"Hallo verden!"}'
```

Expected response:

```bash
{"translation":"Hello World!"}
```

## Stop and clean-up

When finished, stop and remove the docker container using

```bash
docker compose down
```