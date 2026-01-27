import logging
from typing import Optional

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

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
    ) -> None:
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.forced_bos_token_id = forced_bos_token_id
        self.local_files_only = local_files_only

        logger.info("Loading transformers model from %s on %s", model_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=self.local_files_only,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            local_files_only=self.local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()

    def translate(self, src_lang: str, tgt_lang: str, text: str) -> str:
        forced_bos_token_id, prior_src_lang, prior_tgt_lang = self._prepare_language(
            src_lang,
            tgt_lang,
        )
        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        finally:
            self._restore_language(prior_src_lang, prior_tgt_lang)

        generate_kwargs = {
            "num_beams": self.num_beams,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
        }
        if forced_bos_token_id is not None:
            generate_kwargs["forced_bos_token_id"] = forced_bos_token_id

        with torch.no_grad():
            output_tokens = self.model.generate(**inputs, **generate_kwargs)

        translated = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return translated

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

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device)
