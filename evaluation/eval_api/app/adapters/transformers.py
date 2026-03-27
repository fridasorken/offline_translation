from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import ctranslate2
from transformers import AutoTokenizer

from .base import ModelAdapter

logger = logging.getLogger(__name__)
_UNSET = object()

EVAL_API_BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CT2_CACHE_DIR = EVAL_API_BASE_DIR / "models" / "ct2"


class _BaseCTranslate2Adapter(ModelAdapter):
    """Common CT2-backed seq2seq adapter with on-demand HF -> CT2 conversion."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        num_beams: int = 4,
        max_new_tokens: int = 256,
        forced_bos_token_id: Optional[int] = None,
        local_files_only: bool = True,
        num_threads: Optional[int] = None,
        tokenizer_path: Optional[str] = None,
        compute_type: str = "default",
        ct2_model_path: Optional[str] = None,
        ct2_cache_dir: Optional[str] = None,
        quantization: Optional[str] = "float32",
        inter_threads: int = 1,
        force_conversion: bool = False,
    ) -> None:
        """Initialize a CT2-backed adapter and load tokenizer + translator.

        Parameters
        ----------
        model_path : str
            Hugging Face model id or local path to the source model.
        device : str, optional
            Execution device for CTranslate2, by default `"cpu"`.
        num_beams : int, optional
            Beam size used during decoding, by default 4.
        max_new_tokens : int, optional
            Maximum decoding length in generated tokens, by default 256.
        forced_bos_token_id : Optional[int], optional
            Optional forced BOS token id for target language control, by default None.
        local_files_only : bool, optional
            If True, do not allow remote model/tokenizer resolution, by default True.
        num_threads : Optional[int], optional
            Optional CT2 intra-op thread count (`intra_threads`), by default None.
        tokenizer_path : Optional[str], optional
            Optional tokenizer source. Defaults to `model_path` when omitted.
        compute_type : str, optional
            CT2 compute type (for example `default`, `float32`, `int8`), by default `"default"`.
        ct2_model_path : Optional[str], optional
            Optional explicit path to an already converted CT2 model directory.
        ct2_cache_dir : Optional[str], optional
            Optional cache root used for on-demand CT2 conversion output.
        quantization : Optional[str], optional
            Quantization mode used during CT2 conversion, by default `"float32"`.
        inter_threads : int, optional
            CT2 inter-op thread count (`inter_threads`), by default 1.
        force_conversion : bool, optional
            If True, force reconversion even when a cached CT2 model exists, by default False.
        """
        self.model_path = model_path
        self.local_files_only = local_files_only
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.forced_bos_token_id = forced_bos_token_id
        self.num_threads = num_threads
        self.compute_type = compute_type
        self.quantization = quantization
        self.inter_threads = inter_threads
        self.force_conversion = force_conversion

        self.device = self._resolve_device(device)

        resolved_tokenizer_path = tokenizer_path or model_path
        logger.info("Loading tokenizer for CT2 adapter from %s", resolved_tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved_tokenizer_path,
            local_files_only=self.local_files_only,
        )

        self.ct2_model_dir = self._resolve_ct2_model_dir(
            model_path=model_path,
            ct2_model_path=ct2_model_path,
            ct2_cache_dir=ct2_cache_dir,
        )
        self._ensure_converted_model()
        self.translator = self._load_translator()

    def _resolve_device(self, requested_device: str) -> str:
        """Resolve runtime device with safe fallback behavior.

        Parameters
        ----------
        requested_device : str
            Requested device string from model config.

        Returns
        -------
        str
            A supported CT2 device (`"cpu"`, `"cuda"`, or `"auto"`). Falls back to `"cpu"` when invalid
            or when CUDA is requested but unavailable.
        """
        if requested_device == "cuda":
            try:
                if ctranslate2.get_cuda_device_count() > 0:
                    return "cuda"
                logger.warning("CUDA requested for CT2 but not available. Falling back to CPU.")
            except Exception:
                logger.warning("Could not query CT2 CUDA devices. Falling back to CPU.")
            return "cpu"
        if requested_device not in {"cpu", "cuda", "auto"}:
            logger.warning("Unsupported CT2 device '%s', using CPU.", requested_device)
            return "cpu"
        return requested_device

    @staticmethod
    def _slug(value: str) -> str:
        """Convert a string into a filesystem-safe cache key.

        Parameters
        ----------
        value : str
            Input value to normalize.

        Returns
        -------
        str
            Slugified value containing only alphanumerics, underscore, dot, and dash.
        """
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)

    def _resolve_ct2_model_dir(
        self,
        model_path: str,
        ct2_model_path: Optional[str],
        ct2_cache_dir: Optional[str],
    ) -> Path:
        """Resolve which CT2 model directory should be used.

        Parameters
        ----------
        model_path : str
            Source model reference (local path or HF id).
        ct2_model_path : Optional[str]
            Explicit CT2 model directory override.
        ct2_cache_dir : Optional[str]
            Optional CT2 cache root directory.

        Returns
        -------
        Path
            Resolved CT2 model directory path.
        """
        if ct2_model_path:
            candidate = Path(ct2_model_path)
            if not candidate.is_absolute():
                candidate = (EVAL_API_BASE_DIR / candidate).resolve()
            return candidate

        model_candidate = Path(model_path)
        if model_candidate.is_absolute() and model_candidate.exists() and (model_candidate / "model.bin").exists():
            return model_candidate

        cache_root = Path(ct2_cache_dir) if ct2_cache_dir else DEFAULT_CT2_CACHE_DIR
        if not cache_root.is_absolute():
            cache_root = (EVAL_API_BASE_DIR / cache_root).resolve()
        return cache_root / self._slug(model_path)

    def _ensure_converted_model(self) -> None:
        """Ensure a CT2 model exists, converting from Transformers format if needed.

        Raises
        ------
        FileNotFoundError
            Raised when conversion requires local files but `model_path` does not exist and
            `local_files_only=True`.
        """
        model_bin = self.ct2_model_dir / "model.bin"
        if model_bin.exists() and not self.force_conversion:
            logger.info("Using cached CT2 model: %s", self.ct2_model_dir)
            return

        source_path = Path(self.model_path)
        if self.local_files_only and not source_path.exists():
            raise FileNotFoundError(
                f"Cannot convert remote model '{self.model_path}' with local_files_only=true"
            )

        logger.info("Converting model to CT2: %s -> %s", self.model_path, self.ct2_model_dir)
        self.ct2_model_dir.mkdir(parents=True, exist_ok=True)
        force_convert = self.force_conversion or self.ct2_model_dir.exists()

        converter = ctranslate2.converters.TransformersConverter(self.model_path)
        converter.convert(
            str(self.ct2_model_dir),
            quantization=self.quantization,
            force=force_convert,
        )

    def _load_translator(self):
        """Instantiate and return a CTranslate2 translator.

        Returns
        -------
        ctranslate2.Translator
            Configured translator bound to the resolved CT2 model directory.
        """
        translator_kwargs = {
            "device": self.device,
            "compute_type": self.compute_type,
            "inter_threads": self.inter_threads,
        }
        if self.num_threads is not None:
            translator_kwargs["intra_threads"] = self.num_threads

        logger.info(
            "Loading CT2 translator from %s (device=%s, compute_type=%s)",
            self.ct2_model_dir,
            self.device,
            self.compute_type,
        )
        return ctranslate2.Translator(str(self.ct2_model_dir), **translator_kwargs)

    def _prepare_tokenizer_languages(self, src_lang: str, tgt_lang: str) -> tuple[object, object]:
        """Set tokenizer language attributes before tokenization.

        Parameters
        ----------
        src_lang : str
            Source language code.
        tgt_lang : str
            Target language code.

        Returns
        -------
        tuple[object, object]
            Previous `(src_lang, tgt_lang)` tokenizer values. `_UNSET` is used for attributes that
            are not present on the tokenizer.
        """
        prior_src_lang = _UNSET
        prior_tgt_lang = _UNSET

        if hasattr(self.tokenizer, "src_lang"):
            prior_src_lang = getattr(self.tokenizer, "src_lang")
            self.tokenizer.src_lang = src_lang

        if hasattr(self.tokenizer, "tgt_lang"):
            prior_tgt_lang = getattr(self.tokenizer, "tgt_lang")
            self.tokenizer.tgt_lang = tgt_lang

        return prior_src_lang, prior_tgt_lang

    def _restore_tokenizer_languages(self, prior_src_lang: object, prior_tgt_lang: object) -> None:
        """Restore tokenizer language attributes captured before translation.

        Parameters
        ----------
        prior_src_lang : object
            Previous source language attribute value or `_UNSET`.
        prior_tgt_lang : object
            Previous target language attribute value or `_UNSET`.
        """        
        if prior_src_lang is not _UNSET and hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = prior_src_lang
        if prior_tgt_lang is not _UNSET and hasattr(self.tokenizer, "tgt_lang"):
            self.tokenizer.tgt_lang = prior_tgt_lang

    def _resolve_target_lang_id(self, tgt_lang: str) -> Optional[int]:
        """Resolve target language token id for prefix-based constrained decoding.

        Parameters
        ----------
        tgt_lang : str
            Target language code or token key.

        Returns
        -------
        Optional[int]
            Resolved token id, or None if no reliable mapping is found.
        """        
        if self.forced_bos_token_id is not None:
            return int(self.forced_bos_token_id)

        lang_code_to_id = getattr(self.tokenizer, "lang_code_to_id", None)
        if isinstance(lang_code_to_id, dict) and tgt_lang in lang_code_to_id:
            return int(lang_code_to_id[tgt_lang])

        convert_tokens_to_ids = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if callable(convert_tokens_to_ids):
            token_id = convert_tokens_to_ids(tgt_lang)
            unk_id = getattr(self.tokenizer, "unk_token_id", None)
            if token_id is not None and token_id != unk_id:
                try:
                    resolved = self.tokenizer.convert_ids_to_tokens(int(token_id))
                    if resolved == tgt_lang:
                        return int(token_id)
                except Exception:
                    logger.debug("Could not validate target token id for %s", tgt_lang)

        get_lang_id = getattr(self.tokenizer, "get_lang_id", None)
        if callable(get_lang_id):
            try:
                return int(get_lang_id(tgt_lang))
            except Exception:
                logger.debug("Could not resolve language id for %s", tgt_lang)

        return None

    def _resolve_target_prefix(self, tgt_lang: str) -> Optional[list[str]]:
        """Build CT2 `target_prefix` tokens for the selected target language.

        Parameters
        ----------
        tgt_lang : str
            Target language code.

        Returns
        -------
        Optional[list[str]]
            A single-token prefix list for CT2 decoding, or None when target language id
            cannot be resolved.
        """
        target_lang_id = self._resolve_target_lang_id(tgt_lang)
        if target_lang_id is None:
            return None

        token = self.tokenizer.convert_ids_to_tokens(int(target_lang_id))
        if isinstance(token, list):
            token = token[0] if token else None
        if not token:
            return None
        return [token]

    def _translate_tokens(self, source_tokens: list[str], target_prefix: Optional[list[str]]) -> str:
        """Run CT2 translation on tokenized input and decode to text.

        Parameters
        ----------
        source_tokens : list[str]
            Tokenized source sequence.
        target_prefix : Optional[list[str]]
            Optional target prefix token sequence used to control decoding language.

        Returns
        -------
        str
            Decoded translated text.
        """
        translate_kwargs = {
            "beam_size": self.num_beams,
            "max_decoding_length": self.max_new_tokens,
        }
        if target_prefix is not None:
            translate_kwargs["target_prefix"] = [target_prefix]

        results = self.translator.translate_batch([source_tokens], **translate_kwargs)
        hypothesis_tokens = results[0].hypotheses[0]
        hypothesis_ids = self.tokenizer.convert_tokens_to_ids(hypothesis_tokens)
        return self.tokenizer.decode(hypothesis_ids, skip_special_tokens=True).strip()

    def translate(self, src_lang: str, tgt_lang: str, text: str) -> str:
        """Translate a single text string from source to target language.

        Parameters
        ----------
        src_lang : str
            Source language code.
        tgt_lang : str
            Target language code.
        text : str
            Input text to translate.

        Returns
        -------
        str
            Translated output text.
        """
        prior_src_lang, prior_tgt_lang = self._prepare_tokenizer_languages(src_lang, tgt_lang)
        try:
            source_token_ids = self.tokenizer.encode(text)
            source_tokens = self.tokenizer.convert_ids_to_tokens(source_token_ids)
            target_prefix = self._resolve_target_prefix(tgt_lang)
            return self._translate_tokens(source_tokens, target_prefix)
        finally:
            self._restore_tokenizer_languages(prior_src_lang, prior_tgt_lang)

    def count_tokens(self, text: str) -> int:
        """Count tokenizer tokens for throughput reporting.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        int
            Number of tokenizer tokens without special tokens.
        """
        return len(self.tokenizer.encode(text, add_special_tokens=False))


class TransformersAdapter(_BaseCTranslate2Adapter):
    """CT2-backed adapter for generic seq2seq models such as M2M100."""


class NLLBAdapter(_BaseCTranslate2Adapter):
    """CT2-backed adapter for NLLB models."""

    def _prepare_tokenizer_languages(self, src_lang: str, tgt_lang: str) -> tuple[object, object]:
        """Set only source language on NLLB tokenizer.

        Parameters
        ----------
        src_lang : str
            Source language code.
        tgt_lang : str
            Target language code.

        Returns
        -------
        tuple[object, object]
            Previous source language value and `_UNSET` for target language.
        """
        prior_src_lang = _UNSET
        if hasattr(self.tokenizer, "src_lang"):
            prior_src_lang = getattr(self.tokenizer, "src_lang")
            self.tokenizer.src_lang = src_lang
        return prior_src_lang, _UNSET


class OpusMTAdapter(_BaseCTranslate2Adapter):
    """CT2-backed adapter for OPUS models using inline target-language tags."""

    def translate(self, src_lang: str, tgt_lang: str, text: str) -> str:
        """Translate text with OPUS-style inline target-language tags.

        Parameters
        ----------
        src_lang : str
            Source language code.
        tgt_lang : str
            Target language code inserted into the OPUS control tag.
        text : str
            Input text to translate.

        Returns
        -------
        str
            Translated output text.
        """
        tagged_text = f">>{tgt_lang}<< {text}"
        source_token_ids = self.tokenizer.encode(tagged_text)
        source_tokens = self.tokenizer.convert_ids_to_tokens(source_token_ids)
        return self._translate_tokens(source_tokens, None)
