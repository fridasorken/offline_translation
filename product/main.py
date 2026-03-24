from __future__ import annotations

import logging
import time

from config import PRELOAD_ON_STARTUP, RUN_MODE, SINGLE_TEXT, load_product_config
from inference import OpusTranslator

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("product")


def translate_once(translator: OpusTranslator, text: str) -> None:
    """Translate one string and print the result with latency.

    Parameters
    ----------
    translator : OpusTranslator
        Loaded product translator instance.
    text : str
        Input text to translate.
    """
    start = time.perf_counter()
    translated = translator.translate(text)
    latency_ms = int((time.perf_counter() - start) * 1000)
    print(f"source: {text}")
    print(f"translation: {translated}")
    print(f"latency_ms: {latency_ms}")


def interactive_loop(translator: OpusTranslator) -> None:
    """Run an interactive stdin translation loop until the user exits.

    Parameters
    ----------
    translator : OpusTranslator
        Loaded product translator instance reused across all entered sentences.
    """
    print("Interactive Opus translator ready. Press Ctrl+D or type 'exit' to quit.")
    while True:
        try:
            text = input("\ntext> ").strip()
        except EOFError:
            print()
            return
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            return
        translate_once(translator, text)


def main() -> None:
    """Start the product CLI, preload the selected model, and dispatch run mode."""
    if RUN_MODE == "api":
        import uvicorn

        from api import app
        from config import API_HOST, API_PORT

        uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")
        return

    config = load_product_config()
    logger.info(
        "Starting product translator with model=%s source=%s target=%s quantization=%s",
        config.model_id,
        config.source_lang,
        config.target_lang,
        config.quantization,
    )
    translator = OpusTranslator(config)

    if PRELOAD_ON_STARTUP:
        logger.info("Running startup warmup")
        translator.warmup()

    if RUN_MODE == "single":
        translate_once(translator, SINGLE_TEXT)
        return

    interactive_loop(translator)


if __name__ == "__main__":
    main()
