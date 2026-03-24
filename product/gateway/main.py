from __future__ import annotations

import logging

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Translation Gateway")

SUPPORTED_PAIRS = {
    ("en", "nob"),
    ("en", "de"),
    ("en", "pt"),
    ("nob", "en"),
    ("nno", "en"),
    ("de", "en"),
    ("pt", "en"),
}


class TranslateRequest(BaseModel):
    text: str


class TranslateResponse(BaseModel):
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    model_id: str
    latency_ms: int


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


@app.get("/health")
async def health():
    return {"status": "ready"}


@app.post("/translate/{source_lang}/{target_lang}", response_model=TranslateResponse)
async def translate(source_lang: str, target_lang: str, request: TranslateRequest):
    pair = (source_lang, target_lang)
    if pair not in SUPPORTED_PAIRS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language pair: {source_lang}->{target_lang}",
        )

    backend_service = f"translate-{source_lang}-{target_lang}"
    backend_url = f"http://{backend_service}:8080/translate"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(backend_url, json={"text": request.text})
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error("Backend request failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Translation service unavailable: {e}")
