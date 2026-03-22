from __future__ import annotations

import logging
import time

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.config import ACRONYMS_PATH
from app.schemas import InitializeRequest, TranslateRequest, TranslateResponse
from app.services.acronym_service import build_acronym_map, parse_acronyms
from app.services.inference_service import translate_message
from app.services.initialization_service import load_translator

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("product")

app = FastAPI(title="Offline Translation", version="0.1.0")


@app.post("/initialize")
def initialize(payload: InitializeRequest) -> dict[str, str]:
    language = payload.language.lower()

    try:
        if language != "en":
            app.state.translator = load_translator(language)
        app.state.input_language = language

        try:
            df = pd.read_excel(ACRONYMS_PATH, sheet_name=language)
        except ValueError:
            raise ValueError(
                f"No acronym sheet found for language '{language}' in {ACRONYMS_PATH.name}"
            )
        app.state.acronyms = build_acronym_map(dict(zip(df["acronym"], df["expansion"])))

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
    if not hasattr(app.state, "input_language"):
        raise HTTPException(status_code=409, detail="Call /initialize first")

    text = payload.text.strip()

    if payload.sender:
        text = parse_acronyms(text, app.state.acronyms)

    if app.state.input_language == "en":
        return TranslateResponse(translation=text)

    try:
        start = time.perf_counter()
        translated_text = translate_message(app.state.translator, payload.sender, text)
        latency_ms = int((time.perf_counter() - start) * 1000)
    except (RuntimeError, OSError) as exc:
        raise HTTPException(
            status_code=503, detail=f"Translation backend unavailable: {exc}"
        ) from exc
    except Exception as exc:
        logger.exception(
            "translate failed sender=%s input_lang=%s", payload.sender, app.state.input_language
        )
        raise HTTPException(status_code=500, detail="Translation failed") from exc

    logger.info(
        "translate sender=%s input_lang=%s latency_ms=%d",
        payload.sender,
        app.state.input_language,
        latency_ms,
    )

    return TranslateResponse(translation=translated_text)
