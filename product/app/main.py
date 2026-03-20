from __future__ import annotations

import logging
import time

# from config import PRELOAD_ON_STARTUP, RUN_MODE, SINGLE_TEXT
# from inference import OpusTranslator
# from app.config import load_product_config
# from app.schemas import InitializeRequest, TranslateRequest, TranslateResponse
# from app.services.initialization_service import load_translator

from fastapi import FastAPI, HTTPException

from app.schemas import InitializeRequest, TranslateRequest, TranslateResponse
from app.services.initialization_service import load_translator
from app.services.inference_service import translate_message


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("product")

app = FastAPI(title="Offline Translation", version="0.1.0")


@app.post("/initialize")
def initialize(payload: InitializeRequest) -> dict[str, str]:
    language = payload.language.strip().lower()
    if not language:
        raise HTTPException(status_code=400, detail="language must be non-empty")

    try:
        app.state.translator = load_translator(language)
        app.state.input_language = language
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("initialize failed for language=%s", language)
        raise HTTPException(status_code=500, detail="Initialization failed") from exc

    return {"status": "ok", "language": language}


@app.post("/translate", response_model=TranslateResponse)
def translate(payload: TranslateRequest) -> TranslateResponse:
    if not hasattr(app.state, "translator") or not hasattr(app.state, "input_language"):
        raise HTTPException(status_code=409, detail="Call /initialize first")

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")

    if payload.sender:
        # check acronyms
        print("hei")

    if app.state.input_language == "en":
        return TranslateResponse(translation=text)

    try:
        start = time.perf_counter()
        translated_text = translate_message(app.state.translator, payload.sender, text)
        latency_ms = int((time.perf_counter() - start) * 1000)
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=503, detail=f"Translation backend unavailable: {exc}") from exc
    except Exception as exc:
        logger.exception("translate failed sender=%s input_lang=%s", payload.sender, app.state.input_language)
        raise HTTPException(status_code=500, detail="Translation failed") from exc

    logger.info(
        "translate sender=%s input_lang=%s latency_ms=%d",
        payload.sender,
        app.state.input_language,
        latency_ms,
    )

    return TranslateResponse(translation=translated_text)
