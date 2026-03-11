import logging
from typing import Optional

from .base import ModelAdapter
from .opusmt_ctranslate2 import OpusMTCTranslate2Adapter
from .transformers_ctranslate2 import M2MCTranslate2Adapter, NLLBCTranslate2Adapter

logger = logging.getLogger(__name__)


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
        ct2_model_path: Optional[str] = None,
        ct2_cache_dir: Optional[str] = None,
        quantization: Optional[str] = "float32",
        inter_threads: int = 1,
        force_conversion: bool = False,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.forced_bos_token_id = forced_bos_token_id
        self.local_files_only = local_files_only
        self.num_threads = num_threads
        self.tokenizer_path = tokenizer_path
        self.compute_type = compute_type
        self.ct2_model_path = ct2_model_path
        self.ct2_cache_dir = ct2_cache_dir
        self.quantization = quantization
        self.inter_threads = inter_threads
        self.force_conversion = force_conversion

        logger.info("Loading CT2-backed adapter for %s", model_path)
        self._delegate = self._build_delegate()

    def _build_delegate(self) -> ModelAdapter:
        return M2MCTranslate2Adapter(
            model_path=self.model_path,
            device=self.device,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            forced_bos_token_id=self.forced_bos_token_id,
            local_files_only=self.local_files_only,
            num_threads=self.num_threads,
            ct2_model_path=self.ct2_model_path,
            ct2_cache_dir=self.ct2_cache_dir,
            tokenizer_path=self.tokenizer_path,
            quantization=self.quantization,
            compute_type=self.compute_type,
            inter_threads=self.inter_threads,
            force_conversion=self.force_conversion,
        )

    def translate(self, src_lang: str, tgt_lang: str, text: str) -> str:
        return self._delegate.translate(src_lang, tgt_lang, text)

    def count_tokens(self, text: str) -> int:
        return self._delegate.count_tokens(text)


class NLLBAdapter(TransformersAdapter):
    def _build_delegate(self) -> ModelAdapter:
        return NLLBCTranslate2Adapter(
            model_path=self.model_path,
            device=self.device,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            forced_bos_token_id=self.forced_bos_token_id,
            local_files_only=self.local_files_only,
            num_threads=self.num_threads,
            ct2_model_path=self.ct2_model_path,
            ct2_cache_dir=self.ct2_cache_dir,
            tokenizer_path=self.tokenizer_path,
            quantization=self.quantization,
            compute_type=self.compute_type,
            inter_threads=self.inter_threads,
            force_conversion=self.force_conversion,
        )


class OpusMTAdapter(TransformersAdapter):
    def _build_delegate(self) -> ModelAdapter:
        return OpusMTCTranslate2Adapter(
            model_path=self.model_path,
            device=self.device,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            local_files_only=self.local_files_only,
            num_threads=self.num_threads,
            ct2_model_path=self.ct2_model_path,
            ct2_cache_dir=self.ct2_cache_dir,
            tokenizer_path=self.tokenizer_path,
            quantization=self.quantization,
            compute_type=self.compute_type,
            inter_threads=self.inter_threads,
            force_conversion=self.force_conversion,
        )
