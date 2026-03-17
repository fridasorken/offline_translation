from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if _THIS_DIR.name == "product":
    BASE_DIR = _THIS_DIR.parent
else:
    BASE_DIR = _THIS_DIR
DEFAULT_MODEL_CACHE_DIR = BASE_DIR / "models" / "product_ct2"

SOURCE_LANG = "en"
TARGET_LANG = os.getenv("PRODUCT_TARGET_LANG", "nob")
MODEL_QUANTIZATION = os.getenv("PRODUCT_MODEL_QUANTIZATION", "int8")
DEVICE = "cpu"
NUM_BEAMS = 4
MAX_NEW_TOKENS = 256
NUM_THREADS: int | None = None
INTER_THREADS = 1
MODEL_CACHE_DIR = Path(
    os.getenv("PRODUCT_MODEL_CACHE_DIR", str(DEFAULT_MODEL_CACHE_DIR))
).expanduser()
LOCAL_FILES_ONLY = False
PRELOAD_ON_STARTUP = True
RUN_MODE = os.getenv("PRODUCT_RUN_MODE", "interactive")
SINGLE_TEXT = os.getenv("PRODUCT_SINGLE_TEXT", "We need backup now.")


@dataclass(frozen=True)
class ProductModelConfig:
    """Resolved runtime configuration for a single deployed product model.

    Attributes
    ----------
    source_lang : str
        Source language code supported by the running product instance.
    target_lang : str
        Target language code selected for this deployment.
    model_id : str
        Internal model identifier used for logging and reporting.
    model_path : str
        Hugging Face model id or local model path for the selected Opus checkpoint.
    quantization : str
        Quantization mode used when converting the model to CTranslate2 format.
    device : str
        Runtime device string passed to CTranslate2.
    num_beams : int
        Beam size used during decoding.
    max_new_tokens : int
        Maximum decoding length for generated output.
    num_threads : int | None
        Optional intra-op thread count for CTranslate2.
    inter_threads : int
        Inter-op thread count for CTranslate2.
    ct2_cache_dir : Path
        Directory root where converted CTranslate2 models are cached.
    local_files_only : bool
        If True, forbid remote Hugging Face downloads during startup.
    """

    source_lang: str
    target_lang: str
    model_id: str
    model_path: str
    quantization: str
    device: str
    num_beams: int
    max_new_tokens: int
    num_threads: int | None
    inter_threads: int
    ct2_cache_dir: Path
    local_files_only: bool


OPUS_MODELS: dict[str, dict[str, str]] = {
    "nob": {
        "model_id": "opus-mt-tc-big-en-nob-military",
        "model_path": "MariusBerg/opus-tc-big-en-nob-military",
    },
    "de": {
        "model_id": "opus-mt-tc-big-en-de-military-v1",
        "model_path": "MariusBerg/opus-tc-big-en-de-military-v1",
    },
    "pt": {
        "model_id": "opus-mt-tc-big-en-pt-military",
        "model_path": "MariusBerg/opus-tc-big-en-pt-military",
    },
}


def load_product_config() -> ProductModelConfig:
    """Load and validate product runtime configuration from environment.

    Returns
    -------
    ProductModelConfig
        Resolved configuration for the single Opus model this product instance should load.

    Raises
    ------
    ValueError
        If the configured source language is unsupported.
    ValueError
        If the configured target language is not one of the product's known Opus models.
    """
    target_lang = TARGET_LANG.strip().lower()
    source_lang = SOURCE_LANG.strip().lower()
    if source_lang != "en":
        raise ValueError("This product prototype currently supports only English source text.")
    if target_lang not in OPUS_MODELS:
        supported_targets = ", ".join(sorted(OPUS_MODELS))
        raise ValueError(
            f"Unsupported target language '{target_lang}'. Expected one of: {supported_targets}"
        )

    model_info = OPUS_MODELS[target_lang]
    return ProductModelConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        model_id=model_info["model_id"],
        model_path=model_info["model_path"],
        quantization=MODEL_QUANTIZATION,
        device=DEVICE,
        num_beams=NUM_BEAMS,
        max_new_tokens=MAX_NEW_TOKENS,
        num_threads=NUM_THREADS,
        inter_threads=INTER_THREADS,
        ct2_cache_dir=MODEL_CACHE_DIR,
        local_files_only=LOCAL_FILES_ONLY,
    )
