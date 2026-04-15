from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

import preload_models


def test_preload_models_only_builds_unique_model_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_model_ids: list[str] = []

    class FakeTranslator:
        def __init__(self, config: SimpleNamespace) -> None:
            created_model_ids.append(config.model_id)

    fake_inference = types.ModuleType("app.inference")
    fake_inference.OpusTranslator = FakeTranslator
    monkeypatch.setitem(sys.modules, "app.inference", fake_inference)

    monkeypatch.setattr(
        preload_models,
        "OPUS_MODELS",
        {
            ("en", "nob"): {},
            ("de", "en"): {},
            ("en", "de"): {},
        },
    )

    configs = {
        ("de", "en"): SimpleNamespace(
            model_id="de-en",
            model_path="/models/shared",
            source_lang="de",
            target_lang="en",
            quantization="int8",
        ),
        ("en", "de"): SimpleNamespace(
            model_id="en-de",
            model_path="/models/en-de",
            source_lang="en",
            target_lang="de",
            quantization="int8",
        ),
        ("en", "nob"): SimpleNamespace(
            model_id="en-nob",
            model_path="/models/shared",
            source_lang="en",
            target_lang="nob",
            quantization="int8",
        ),
    }
    monkeypatch.setattr(
        preload_models,
        "load_product_config",
        lambda source_lang, target_lang: configs[(source_lang, target_lang)],
    )

    preload_models.main()

    assert created_model_ids == ["de-en", "en-de"]
