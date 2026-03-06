import logging
from typing import Optional

import ctranslate2
from transformers import AutoTokenizer

from .base import ModelAdapter

logger = logging.getLogger(__name__)
_UNSET = object()


class TransformersAdapter(ModelAdapter):
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
    ) -> None:
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.forced_bos_token_id = forced_bos_token_id
        self.local_files_only = local_files_only
        self.num_threads = num_threads
        self.tokenizer_path = tokenizer_path or model_path
        self.compute_type = compute_type

        logger.info("Loading transformers model from %s on %s", model_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            local_files_only=self.local_files_only,
        )
        self.translator = ctranslate2.Translator(
            model_path,
            device=str(self.device),
            inter_threads=1,
            intra_threads=self.num_threads or 0,
            compute_type=self.compute_type,
        )

    def translate(self, src_lang: str, tgt_lang: str, text: str) -> str:
        forced_bos_token_id, prior_src_lang, prior_tgt_lang = self._prepare_language(
            src_lang,
            tgt_lang,
        )
        try:
            input_tokens = self._encode_source(text)
            target_prefix = self._build_target_prefix(forced_bos_token_id)
        finally:
            self._restore_language(prior_src_lang, prior_tgt_lang)

        results = self.translator.translate_batch(
            [input_tokens],
            target_prefix=[target_prefix] if target_prefix else None,
            beam_size=self.num_beams,
            max_decoding_length=self.max_new_tokens,
        )

        output_tokens = self._strip_prefix(results[0].hypotheses[0], target_prefix)
        return self.tokenizer.convert_tokens_to_string(output_tokens).strip()
    
    def _encode_source(self, text: str) -> list[str]:
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        return self.tokenizer.convert_ids_to_tokens(input_ids)

    def _build_target_prefix(self, forced_bos_token_id: Optional[int]) -> list[str]:
        if forced_bos_token_id is None:
            return []
        return self.tokenizer.convert_ids_to_tokens([forced_bos_token_id])

    @staticmethod
    def _strip_prefix(output_tokens: list[str], prefix_tokens: list[str]) -> list[str]:
        if prefix_tokens and output_tokens[:len(prefix_tokens)] == prefix_tokens:
            return output_tokens[len(prefix_tokens):]
        return output_tokens

    def _prepare_language(
        self,
        src_lang: str,
        tgt_lang: str,
    ) -> tuple[Optional[int], object, object]:
        prior_src_lang = _UNSET
        prior_tgt_lang = _UNSET

        if hasattr(self.tokenizer, "src_lang"):
            prior_src_lang = getattr(self.tokenizer, "src_lang")
            self.tokenizer.src_lang = src_lang

        if hasattr(self.tokenizer, "tgt_lang"):
            prior_tgt_lang = getattr(self.tokenizer, "tgt_lang")
            self.tokenizer.tgt_lang = tgt_lang

        forced_bos_token_id = self.forced_bos_token_id
        if forced_bos_token_id is None:
            lang_code_to_id = getattr(self.tokenizer, "lang_code_to_id", None)
            if isinstance(lang_code_to_id, dict) and tgt_lang in lang_code_to_id:
                forced_bos_token_id = lang_code_to_id[tgt_lang]
            else:
                get_lang_id = getattr(self.tokenizer, "get_lang_id", None)
                if callable(get_lang_id):
                    try:
                        forced_bos_token_id = get_lang_id(tgt_lang)
                    except Exception:
                        logger.debug("Could not resolve language id for %s", tgt_lang)

        return forced_bos_token_id, prior_src_lang, prior_tgt_lang

    def _restore_language(
        self,
        prior_src_lang: object,
        prior_tgt_lang: object,
    ) -> None:
        if prior_src_lang is not _UNSET and hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = prior_src_lang
        if prior_tgt_lang is not _UNSET and hasattr(self.tokenizer, "tgt_lang"):
            self.tokenizer.tgt_lang = prior_tgt_lang

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in text"""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def _resolve_device(device: str) -> str:
        return device
    
class NLLBAdapter(TransformersAdapter):    
    def _prepare_language(
        self,
        src_lang: str,
        tgt_lang: str,
    ) -> tuple[Optional[int], object, object]:
        """Override to handle NLLB set_tgt_lang_special_tokens correctly."""
        prior_src_lang = _UNSET
        prior_tgt_lang = _UNSET

        if hasattr(self.tokenizer, "src_lang"):
            prior_src_lang = getattr(self.tokenizer, "src_lang")
            self.tokenizer.src_lang = src_lang

        if hasattr(self.tokenizer, "tgt_lang"):
            prior_tgt_lang = getattr(self.tokenizer, "tgt_lang")

        if hasattr(self.tokenizer, "set_tgt_lang_special_tokens"):
            self.tokenizer.set_tgt_lang_special_tokens(tgt_lang)

        forced_bos_token_id = self.forced_bos_token_id
        if forced_bos_token_id is None:
            cur_lang_code = getattr(self.tokenizer, "cur_lang_code", None)
            if isinstance(cur_lang_code, int):
                forced_bos_token_id = cur_lang_code
                logger.debug(f"Found forced_bos_token_id={forced_bos_token_id} from cur_lang_code")

        return forced_bos_token_id, prior_src_lang, prior_tgt_lang
    
class OpusMTAdapter(TransformersAdapter):
    def translate(self, src_lang: str, tgt_lang: str, text: str) -> str:
        tagged_text = f">>{tgt_lang}<< {text}"
        return super().translate(src_lang, tgt_lang, tagged_text)