import gc
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

from .adapters.base import ModelAdapter
from .adapters.transformers import (
    NLLBAdapter,
    OpusMTAdapter,
    TransformersAdapter,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_CONFIG_ENV = "MODELS_CONFIG_PATH"
DEFAULT_MODELS_CONFIG = BASE_DIR / "models.json"


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    adapter: str
    model_path: str
    supported_pairs: Set[Tuple[str, str]]
    adapter_params: Dict[str, object]


class ModelRegistry:
    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config_path = config_path or self._resolve_config_path()
        self._configs: Dict[str, ModelConfig] = {}
        self._adapters: Dict[str, ModelAdapter] = {}
        self._cache_lock = threading.RLock()

    def load(self) -> None:
        logger.info("Loading model registry from %s", self.config_path)
        with self.config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, dict):
            raise ValueError("models.json must contain an object at the top level")

        configs: Dict[str, ModelConfig] = {}

        for model_id, raw_config in data.items():
            if not isinstance(raw_config, dict):
                raise ValueError(f"Config for {model_id} must be an object")

            adapter_name = raw_config.get("adapter")
            model_path_raw = raw_config.get("model_path")
            if not adapter_name or not model_path_raw:
                raise ValueError(f"Config for {model_id} must include adapter and model_path")

            adapter_params = {
                key: value
                for key, value in raw_config.items()
                if key not in {"adapter", "model_path", "supported_pairs"}
            }
            supported_pairs = self._parse_supported_pairs(raw_config.get("supported_pairs"))
            model_ref = self._resolve_model_ref(adapter_name, model_path_raw, adapter_params)

            configs[model_id] = ModelConfig(
                model_id=model_id,
                adapter=adapter_name,
                model_path=model_ref,
                supported_pairs=supported_pairs,
                adapter_params=adapter_params,
            )

        self._configs = configs
        self.clear_adapter_cache()

    def get_adapter(self, model_id: str) -> ModelAdapter:
        with self._cache_lock:
            if model_id in self._adapters:
                return self._adapters[model_id]

            # Keep at most one loaded model in the main process.
            if self._adapters:
                self.clear_adapter_cache()

            config = self.get_config(model_id)
            adapter = self._build_adapter(config)
            self._adapters[model_id] = adapter
            return adapter

    def clear_adapter_cache(self) -> None:
        with self._cache_lock:
            if not self._adapters:
                return
            self._adapters.clear()
        gc.collect()


    def get_config(self, model_id: str) -> ModelConfig:
        try:
            return self._configs[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown model_id: {model_id}") from exc

    def is_supported_pair(self, model_id: str, src_lang: str, tgt_lang: str) -> bool:
        config = self.get_config(model_id)
        if not config.supported_pairs:
            return True
        return (src_lang, tgt_lang) in config.supported_pairs

    def list_models(self) -> Iterable[str]:
        return self._configs.keys()

    def _build_adapter(self, config: ModelConfig) -> ModelAdapter:
        if config.adapter == "transformers":
            return TransformersAdapter(config.model_path, **config.adapter_params)
        elif config.adapter == "nllb":
            return NLLBAdapter(config.model_path, **config.adapter_params)
        elif config.adapter == "opusmt":
            return OpusMTAdapter(config.model_path, **config.adapter_params)
        raise ValueError(f"Unsupported adapter: {config.adapter}")

    @staticmethod
    def _parse_supported_pairs(raw_pairs: Optional[Iterable[Iterable[str]]]) -> Set[Tuple[str, str]]:
        pairs: Set[Tuple[str, str]] = set()
        if not raw_pairs:
            return pairs
        for pair in raw_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("supported_pairs must be a list of [src_lang, tgt_lang]")
            src, tgt = pair
            pairs.add((str(src), str(tgt)))
        return pairs

    @staticmethod
    def _resolve_config_path() -> Path:
        override = os.getenv(MODELS_CONFIG_ENV)
        if not override:
            return DEFAULT_MODELS_CONFIG
        override_path = Path(override)
        if not override_path.is_absolute():
            override_path = (BASE_DIR / override_path).resolve()
        return override_path

    @staticmethod
    def _resolve_model_ref(
        adapter_name: str,
        model_path_raw: object,
        adapter_params: Dict[str, object],
    ) -> str:
        if not isinstance(model_path_raw, str):
            raise ValueError("model_path must be a string")

        local_files_only = bool(adapter_params.get("local_files_only", True))
        if adapter_name in (
            "transformers",
            "nllb",
            "opusmt",
        ):
            candidate = Path(model_path_raw)
            if not candidate.is_absolute():
                candidate = (BASE_DIR / candidate).resolve()

            if candidate.exists():
                return str(candidate)

            if local_files_only:
                raise FileNotFoundError(
                    f"Model path not found for {model_path_raw} and local_files_only is true"
                )

            return model_path_raw

        candidate = Path(model_path_raw)
        if not candidate.is_absolute():
            candidate = (BASE_DIR / candidate).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Model path not found: {candidate}")
        return str(candidate)
