from __future__ import annotations

import logging

from app.config import OPUS_MODELS, load_product_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("product-preload")


def main() -> None:
    """Download and convert all configured product models during image build."""
    from app.inference import OpusTranslator

    seen_model_paths: set[str] = set()

    for source_lang, target_lang in sorted(OPUS_MODELS):
        config = load_product_config(source_lang=source_lang, target_lang=target_lang)
        if config.model_path in seen_model_paths:
            continue

        seen_model_paths.add(config.model_path)
        logger.info(
            "Preloading model=%s source=%s target=%s quantization=%s",
            config.model_id,
            config.source_lang,
            config.target_lang,
            config.quantization,
        )
        OpusTranslator(config)

    logger.info("Preloaded %s unique model artifacts", len(seen_model_paths))


if __name__ == "__main__":
    main()
