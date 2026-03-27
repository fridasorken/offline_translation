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


@app.post("/initialize", summary="Initialize a language to load models into memory")
def initialize(payload: InitializeRequest) -> dict[str, str]:
    """
    Initialize the application for a given language.

    This endpoint sets up the translation runtime and loads associated acronym
    mappings for the specified language.

    If the given language is English, no translation runtime is created.

    Acronym mappings are loaded from an Excel sheet matching the language code.
    If no sheet is found, an empty mapping is used.

    Parameters
    ----------
    payload : InitializeRequest
        Request payload containing the target language code.

    Returns
    -------
    dict of str to str
        A dictionary containing the initialization status and language code.

    Raises
    ------
    HTTPException
        400: Invalid language or configuration error.

    HTTPException
        503: Required file with acronyms not found.

    HTTPException
        500: Unexpected internal error in initialization.
    """

    language = payload.language.lower()

    try:
        if language != "en":
            app.state.translator = load_translator(language)
        app.state.input_language = language

        try:
            df = pd.read_excel(ACRONYMS_PATH, sheet_name=language)
            app.state.acronyms = build_acronym_map(dict(zip(df["acronym"], df["expansion"])))
        except ValueError:
            logger.warning(
                "No acronym sheet found for language=%s in %s, using empty acronym map",
                language,
                ACRONYMS_PATH.name,
            )
            app.state.acronyms = {}

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("initialize failed for language=%s", language)
        raise HTTPException(status_code=500, detail="Initialization failed") from exc

    return {"status": "ok", "language": language}


@app.post(
    "/translate",
    response_model=TranslateResponse,
    summary="Translate text based on loaded languages",
)
def translate(payload: TranslateRequest) -> TranslateResponse:
    """
    Translate text based on the configured language and direction of the message.

    This endpoint translates text either to or from English depending on the request
    direction. It requires previously performed initialization with the `\initialize`
    endpoint.

    If the configured language is English, no translation is performed and the
    input text is returned.

    For outgoing messages, mapped acronyms are expanded before translation to work
    with translation models' capabilities.

    Parameters
    ----------
    payload : TranslateRequest
        Request payload containing the text to translate and whether the message is
        outgoing or incoming.

    Returns
    -------
    TranslateResponse
        Response payload containing the translated text.

    Raises
    ------
    HTTPException
        409: Application not previously initialized.

    HTTPException
        503: Translation backend is unavailable.

    HTTPException
        500: Unexpected internal error during translation.
    """

    if not hasattr(app.state, "input_language"):
        raise HTTPException(status_code=409, detail="Call /initialize first")

    text = payload.text.strip()

    if payload.is_outgoing:
        text = parse_acronyms(text, app.state.acronyms)

    if app.state.input_language == "en":
        return TranslateResponse(translation=text)

    try:
        start = time.perf_counter()
        translated_text = translate_message(app.state.translator, payload.is_outgoing, text)
        latency_ms = int((time.perf_counter() - start) * 1000)
    except (RuntimeError, OSError) as exc:
        raise HTTPException(
            status_code=503, detail=f"Translation backend unavailable: {exc}"
        ) from exc
    except Exception as exc:
        logger.exception(
            "translate failed outgoing=%s input_lang=%s",
            payload.is_outgoing,
            app.state.input_language,
        )
        raise HTTPException(status_code=500, detail="Translation failed") from exc

    logger.info(
        "translate outgoing=%s input_lang=%s latency_ms=%d",
        payload.is_outgoing,
        app.state.input_language,
        latency_ms,
    )

    return TranslateResponse(translation=translated_text)
