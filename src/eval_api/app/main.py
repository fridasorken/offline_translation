import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .registry import ModelRegistry
from .schemas import TranslateRequest, TranslateResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = ModelRegistry()
    try:
        registry.load()
    except Exception:
        logger.exception("Failed to load model registry")
        raise
    app.state.registry = registry
    yield


app = FastAPI(title="Offline Translation Eval API", version="0.1.0", lifespan=lifespan)


@app.post("/translate", response_model=TranslateResponse)
def translate(request: TranslateRequest) -> TranslateResponse:
    registry: ModelRegistry = app.state.registry

    try:
        registry.get_config(request.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="model_id not found")

    if not registry.is_supported_pair(request.model_id, request.src_lang, request.tgt_lang):
        raise HTTPException(status_code=400, detail="unsupported language pair")

    adapter = registry.get_adapter(request.model_id)

    start = time.perf_counter()
    try:
        translated = adapter.translate(request.src_lang, request.tgt_lang, request.source)
    except Exception:
        logger.exception("Translation failed for model_id=%s", request.model_id)
        raise HTTPException(status_code=500, detail="translation failed")
    latency_ms = int((time.perf_counter() - start) * 1000)

    logger.info(
        "translate model_id=%s src=%s tgt=%s latency_ms=%d",
        request.model_id,
        request.src_lang,
        request.tgt_lang,
        latency_ms,
    )

    return TranslateResponse(
        model_id=request.model_id,
        translated_value=translated,
        latency_ms=latency_ms,
    )
