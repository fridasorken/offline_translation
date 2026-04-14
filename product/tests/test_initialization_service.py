from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

import app.services.initialization_service as initialization_service


def test_translation_runtime_builds_both_models_and_warms_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_translators: list[object] = []

    class FakeTranslator:
        def __init__(self, config: object) -> None:
            self.config = config
            self.warmup_calls = 0
            created_translators.append(self)

        def warmup(self) -> None:
            self.warmup_calls += 1

    fake_inference = types.ModuleType("app.inference")
    fake_inference.OpusTranslator = FakeTranslator
    monkeypatch.setitem(sys.modules, "app.inference", fake_inference)

    configs = {
        ("de", "en"): SimpleNamespace(name="sender"),
        ("en", "de"): SimpleNamespace(name="receiver"),
    }
    monkeypatch.setattr(
        initialization_service,
        "load_product_config",
        lambda source_lang, target_lang: configs[(source_lang, target_lang)],
    )

    runtime = initialization_service.TranslationRuntime("de")

    assert [translator.config for translator in created_translators] == [
        configs[("de", "en")],
        configs[("en", "de")],
    ]
    assert runtime.for_direction(True) is runtime.sender
    assert runtime.for_direction(False) is runtime.receiver
    assert [translator.warmup_calls for translator in created_translators] == [1, 1]


def test_load_translator_wraps_runtime_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed_languages: list[str] = []

    class FakeRuntime:
        def __init__(self, language: str) -> None:
            self.language = language
            constructed_languages.append(language)

    monkeypatch.setattr(initialization_service, "TranslationRuntime", FakeRuntime)

    runtime = initialization_service.load_translator("pt")

    assert constructed_languages == ["pt"]
    assert isinstance(runtime, FakeRuntime)
    assert runtime.language == "pt"
