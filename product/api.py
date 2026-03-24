from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from config import load_product_config
from inference import OpusTranslator

logger = logging.getLogger(__name__)

translator: OpusTranslator | None = None
translate_lock = asyncio.Lock()


class TranslateRequest(BaseModel):
    text: str


class TranslateResponse(BaseModel):
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    model_id: str
    latency_ms: int


class InfoResponse(BaseModel):
    source_lang: str
    target_lang: str
    model_id: str
    quantization: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global translator
    config = load_product_config()
    logger.info(
        "Loading translator: %s -> %s (model=%s)",
        config.source_lang,
        config.target_lang,
        config.model_id,
    )
    translator = OpusTranslator(config)
    translator.warmup()
    logger.info("Translator ready")
    yield
    translator = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ready" if translator else "loading"}


@app.get("/info", response_model=InfoResponse)
async def info():
    config = translator.config
    return InfoResponse(
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        model_id=config.model_id,
        quantization=config.quantization,
    )


@app.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    start = time.perf_counter()
    async with translate_lock:
        translated = translator.translate(request.text)
    latency_ms = int((time.perf_counter() - start) * 1000)

    return TranslateResponse(
        source_text=request.text,
        translated_text=translated,
        source_lang=translator.config.source_lang,
        target_lang=translator.config.target_lang,
        model_id=translator.config.model_id,
        latency_ms=latency_ms,
    )
