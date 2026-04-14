from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_inference_module(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {
        "converter_init": [],
        "convert": [],
        "translator_init": [],
    }

    class FakeBackendTranslator:
        def __init__(self, model_dir: str, **kwargs: object) -> None:
            calls["translator_init"].append((model_dir, kwargs))
            self.model_dir = model_dir
            self.kwargs = kwargs

        def translate_batch(
            self,
            tokens: list[list[str]],
            beam_size: int,
            max_decoding_length: int,
        ) -> list[SimpleNamespace]:
            calls["translate_batch"] = {
                "tokens": tokens,
                "beam_size": beam_size,
                "max_decoding_length": max_decoding_length,
            }
            return [SimpleNamespace(hypotheses=[["decoded", "tokens"]])]

    class FakeConverter:
        def __init__(self, model_path: str) -> None:
            calls["converter_init"].append(model_path)

        def convert(self, output_dir: str, quantization: str, force: bool) -> None:
            calls["convert"].append((output_dir, quantization, force))

    class FakeTokenizer:
        def __init__(self) -> None:
            self.encoded_inputs: list[str] = []
            self.ids_to_tokens_inputs: list[list[int]] = []
            self.tokens_to_ids_inputs: list[list[str]] = []
            self.decode_inputs: list[tuple[list[int], bool]] = []

        def encode(self, text: str) -> list[int]:
            self.encoded_inputs.append(text)
            return [101, 102]

        def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
            self.ids_to_tokens_inputs.append(list(token_ids))
            return ["src-1", "src-2"]

        def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
            self.tokens_to_ids_inputs.append(list(tokens))
            return [201, 202]

        def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
            self.decode_inputs.append((list(token_ids), skip_special_tokens))
            return " decoded text "

    tokenizer = FakeTokenizer()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_path: str, local_files_only: bool = False) -> FakeTokenizer:
            calls["tokenizer_load"] = (model_path, local_files_only)
            return tokenizer

    fake_ctranslate2 = types.ModuleType("ctranslate2")
    fake_ctranslate2.Translator = FakeBackendTranslator
    fake_ctranslate2.converters = SimpleNamespace(TransformersConverter=FakeConverter)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer

    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ctranslate2)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    sys.modules.pop("app.inference", None)

    module = importlib.import_module("app.inference")
    return module, calls, tokenizer


def test_slug_normalizes_model_references(monkeypatch: pytest.MonkeyPatch) -> None:
    inference, _, _ = load_inference_module(monkeypatch)

    assert inference.OpusTranslator._slug("MariusBerg/model name@v1") == "MariusBerg_model_name_v1"


def test_ensure_converted_model_uses_existing_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inference, calls, _ = load_inference_module(monkeypatch)
    translator = inference.OpusTranslator.__new__(inference.OpusTranslator)
    translator.ct2_model_dir = tmp_path / "cached-model"
    translator.ct2_model_dir.mkdir()
    (translator.ct2_model_dir / "model.bin").touch()
    translator.config = SimpleNamespace(
        local_files_only=False,
        model_path="remote/model",
        quantization="int8",
    )

    translator._ensure_converted_model()

    assert calls["converter_init"] == []
    assert calls["convert"] == []


def test_ensure_converted_model_requires_local_source_when_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inference, _, _ = load_inference_module(monkeypatch)
    translator = inference.OpusTranslator.__new__(inference.OpusTranslator)
    translator.ct2_model_dir = tmp_path / "ct2-cache"
    translator.config = SimpleNamespace(
        local_files_only=True,
        model_path=str(tmp_path / "missing-model"),
        quantization="int8",
    )

    with pytest.raises(FileNotFoundError, match="PRODUCT_LOCAL_FILES_ONLY=true"):
        translator._ensure_converted_model()


def test_ensure_converted_model_converts_uncached_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inference, calls, _ = load_inference_module(monkeypatch)
    translator = inference.OpusTranslator.__new__(inference.OpusTranslator)
    translator.ct2_model_dir = tmp_path / "ct2-cache"
    local_model = tmp_path / "source-model"
    local_model.mkdir()
    translator.config = SimpleNamespace(
        local_files_only=True,
        model_path=str(local_model),
        quantization="int8",
    )

    translator._ensure_converted_model()

    assert calls["converter_init"] == [str(local_model)]
    assert calls["convert"] == [(str(translator.ct2_model_dir), "int8", True)]


def test_load_translator_passes_thread_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inference, calls, _ = load_inference_module(monkeypatch)
    translator = inference.OpusTranslator.__new__(inference.OpusTranslator)
    translator.ct2_model_dir = tmp_path / "ct2-cache"
    translator.config = SimpleNamespace(
        device="cpu",
        quantization="int8",
        inter_threads=2,
        num_threads=6,
    )

    backend = translator._load_translator()

    assert calls["translator_init"] == [
        (
            str(translator.ct2_model_dir),
            {
                "device": "cpu",
                "compute_type": "default",
                "inter_threads": 2,
                "intra_threads": 6,
            },
        )
    ]
    assert backend.model_dir == str(translator.ct2_model_dir)


def test_translate_adds_target_tag_and_decodes_first_hypothesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference, _, tokenizer = load_inference_module(monkeypatch)
    batch_calls: list[tuple[list[list[str]], int, int]] = []

    class FakeLoadedTranslator:
        def translate_batch(
            self,
            tokens: list[list[str]],
            beam_size: int,
            max_decoding_length: int,
        ) -> list[SimpleNamespace]:
            batch_calls.append((tokens, beam_size, max_decoding_length))
            return [SimpleNamespace(hypotheses=[["hyp-1", "hyp-2"]])]

    translator = inference.OpusTranslator.__new__(inference.OpusTranslator)
    translator.config = SimpleNamespace(
        use_target_tag=True,
        target_lang="de",
        num_beams=4,
        max_new_tokens=32,
    )
    translator.tokenizer = tokenizer
    translator.translator = FakeLoadedTranslator()

    translated = translator.translate("Hold position")

    assert tokenizer.encoded_inputs == [">>de<< Hold position"]
    assert batch_calls == [([["src-1", "src-2"]], 4, 32)]
    assert tokenizer.tokens_to_ids_inputs == [["hyp-1", "hyp-2"]]
    assert tokenizer.decode_inputs == [([201, 202], True)]
    assert translated == "decoded text"


def test_warmup_uses_static_probe_text(monkeypatch: pytest.MonkeyPatch) -> None:
    inference, _, _ = load_inference_module(monkeypatch)
    translator = inference.OpusTranslator.__new__(inference.OpusTranslator)
    seen_inputs: list[str] = []
    translator.translate = lambda text: seen_inputs.append(text) or "ok"

    translator.warmup()

    assert seen_inputs == ["System warmup."]
