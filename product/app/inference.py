from __future__ import annotations

import logging
import re
from pathlib import Path

import ctranslate2
from transformers import AutoTokenizer

from app.config import ProductModelConfig

logger = logging.getLogger(__name__)


class OpusTranslator:
    """Minimal Opus-only translator backed by CTranslate2.

    This class is the core product runtime. It resolves a cache path, converts the
    configured Hugging Face checkpoint to CTranslate2 on first startup if needed,
    and keeps one warm translator in memory for repeated interactive translations.
    """

    def __init__(self, config: ProductModelConfig) -> None:
        """Initialize tokenizer, ensure CT2 conversion, and load the translator.

        Parameters
        ----------
        config : ProductModelConfig
            Resolved product configuration for the selected target language.
        """
        self.config = config
        self.ct2_model_dir = self._resolve_ct2_model_dir()

        logger.info("Loading tokenizer from %s", self.config.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            local_files_only=self.config.local_files_only,
        )

        self._ensure_converted_model()
        self.translator = self._load_translator()


    def _resolve_ct2_model_dir(self) -> Path:
        """Resolve the cache directory for the converted CT2 model.

        Returns
        -------
        Path
            Filesystem path where the converted CT2 model should live.
        """
        return self.config.ct2_cache_dir / self._slug(self.config.model_path)


    @staticmethod
    def _slug(value: str) -> str:
        """Convert a model reference into a filesystem-safe cache key.

        Parameters
        ----------
        value : str
            Raw model id or path.

        Returns
        -------
        str
            Normalized string safe to use as a cache directory name.
        """
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


    def _ensure_converted_model(self) -> None:
        """Ensure the configured model exists in CT2 format.

        Raises
        ------
        FileNotFoundError
            If remote downloads are disabled and the configured source model does not
            exist locally.
        """
        model_bin = self.ct2_model_dir / "model.bin"
        if model_bin.exists():
            logger.info("Using cached CT2 model: %s", self.ct2_model_dir)
            return

        if self.config.local_files_only and not Path(self.config.model_path).exists():
            raise FileNotFoundError(
                "Cannot convert remote model "
                f"'{self.config.model_path}' with PRODUCT_LOCAL_FILES_ONLY=true"
            )

        logger.info("Converting model to CT2: %s -> %s", self.config.model_path, self.ct2_model_dir)
        self.ct2_model_dir.mkdir(parents=True, exist_ok=True)
        converter = ctranslate2.converters.TransformersConverter(self.config.model_path)
        converter.convert(
            str(self.ct2_model_dir),
            quantization=self.config.quantization,
            force=True,
        )


    def _load_translator(self) -> ctranslate2.Translator:
        """Create the CTranslate2 translator for the converted model.

        Returns
        -------
        ctranslate2.Translator
            Ready-to-use translator instance for runtime inference.
        """
        kwargs: dict[str, object] = {
            "device": self.config.device,
            "compute_type": "default",
            "inter_threads": self.config.inter_threads,
        }
        if self.config.num_threads is not None:
            kwargs["intra_threads"] = self.config.num_threads

        logger.info(
            "Loading CT2 translator from %s (device=%s, quantization=%s)",
            self.ct2_model_dir,
            self.config.device,
            self.config.quantization,
        )
        return ctranslate2.Translator(str(self.ct2_model_dir), **kwargs)


    def warmup(self) -> None:
        """Run one small translation to warm tokenizer and translator state."""
        self.translate("System warmup.")


    def translate(self, text: str) -> str:
        """Translate one input string using the configured product language pair.

        Parameters
        ----------
        text : str
            Input text in the configured source language.

        Returns
        -------
        str
            Translated output text from the configured Opus model.
        """
        if self.config.use_target_tag:
            prepared_text = f">>{self.config.target_lang}<< {text}"
        else:
            prepared_text = text
        source_token_ids = self.tokenizer.encode(prepared_text)
        source_tokens = self.tokenizer.convert_ids_to_tokens(source_token_ids)
        results = self.translator.translate_batch(
            [source_tokens],
            beam_size=self.config.num_beams,
            max_decoding_length=self.config.max_new_tokens,
        )
        hypothesis_tokens = results[0].hypotheses[0]
        hypothesis_ids = self.tokenizer.convert_tokens_to_ids(hypothesis_tokens)
        return self.tokenizer.decode(hypothesis_ids, skip_special_tokens=True).strip()
