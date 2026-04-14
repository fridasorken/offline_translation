from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.main as product_main


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(product_main.app) as test_client:
        yield test_client


def test_initialize_loads_runtime_and_acronyms(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = object()
    requested_languages: list[str] = []

    def fake_load_translator(language: str) -> object:
        requested_languages.append(language)
        return runtime

    monkeypatch.setattr(product_main, "load_translator", fake_load_translator)
    monkeypatch.setattr(
        product_main.pd,
        "read_excel",
        lambda *args, **kwargs: pd.DataFrame(
            {"acronym": ["NATO"], "expansion": ["North Atlantic Treaty Organization"]}
        ),
    )

    response = client.post("/initialize", json={"language": "NOB"})

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "language": "nob"}
    assert requested_languages == ["nob"]
    assert product_main.app.state.translator is runtime
    assert product_main.app.state.input_language == "nob"
    assert product_main.app.state.acronyms == {
        "nato": "North Atlantic Treaty Organization",
    }


def test_initialize_english_skips_model_loading(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        product_main,
        "load_translator",
        lambda language: (_ for _ in ()).throw(AssertionError("translator should not load")),
    )
    monkeypatch.setattr(
        product_main.pd,
        "read_excel",
        lambda *args, **kwargs: pd.DataFrame({"acronym": [], "expansion": []}),
    )

    response = client.post("/initialize", json={"language": "en"})

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "language": "en"}
    assert not hasattr(product_main.app.state, "translator")


def test_initialize_uses_empty_acronym_map_when_sheet_is_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = object()

    monkeypatch.setattr(product_main, "load_translator", lambda language: runtime)
    monkeypatch.setattr(
        product_main.pd,
        "read_excel",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing sheet")),
    )

    response = client.post("/initialize", json={"language": "de"})

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "language": "de"}
    assert product_main.app.state.translator is runtime
    assert product_main.app.state.acronyms == {}


def test_initialize_returns_400_for_configuration_errors(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        product_main,
        "load_translator",
        lambda language: (_ for _ in ()).throw(ValueError("unsupported language pair")),
    )

    response = client.post("/initialize", json={"language": "fr"})

    assert response.status_code == 400
    assert response.json() == {"detail": "unsupported language pair"}


def test_initialize_returns_503_when_acronym_file_is_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(product_main, "load_translator", lambda language: object())
    monkeypatch.setattr(
        product_main.pd,
        "read_excel",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("acronyms missing")),
    )

    response = client.post("/initialize", json={"language": "pt"})

    assert response.status_code == 503
    assert response.json() == {"detail": "acronyms missing"}


def test_initialize_returns_500_for_unexpected_failures(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        product_main,
        "load_translator",
        lambda language: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    response = client.post("/initialize", json={"language": "de"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Initialization failed"}


def test_translate_requires_initialize(client: TestClient) -> None:
    response = client.post("/translate", json={"is_outgoing": False, "text": "Hello"})

    assert response.status_code == 409
    assert response.json() == {"detail": "Call /initialize first"}


def test_translate_expands_outgoing_acronyms_before_translation(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = object()
    calls: list[tuple[object, bool, str]] = []

    def fake_translate_message(translator: object, is_outgoing: bool, text: str) -> str:
        calls.append((translator, is_outgoing, text))
        return "Translated text"

    product_main.app.state.input_language = "nob"
    product_main.app.state.translator = runtime
    product_main.app.state.acronyms = {"nato": "North Atlantic Treaty Organization"}
    monkeypatch.setattr(product_main, "translate_message", fake_translate_message)

    response = client.post("/translate", json={"is_outgoing": True, "text": "NATO ready."})

    assert response.status_code == 200
    assert response.json() == {"translation": "Translated text"}
    assert calls == [(runtime, True, "North Atlantic Treaty Organization ready.")]


def test_translate_does_not_expand_acronyms_for_incoming_messages(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_translate_message(translator: object, is_outgoing: bool, text: str) -> str:
        calls.append(text)
        return "Hei"

    monkeypatch.setattr(
        product_main,
        "parse_acronyms",
        lambda text, acronym_map: (_ for _ in ()).throw(AssertionError("should not parse")),
    )
    monkeypatch.setattr(product_main, "translate_message", fake_translate_message)
    product_main.app.state.input_language = "de"
    product_main.app.state.translator = object()
    product_main.app.state.acronyms = {"nato": "North Atlantic Treaty Organization"}

    response = client.post("/translate", json={"is_outgoing": False, "text": "Hello team"})

    assert response.status_code == 200
    assert response.json() == {"translation": "Hei"}
    assert calls == ["Hello team"]


def test_translate_returns_original_text_for_english_runtime(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        product_main,
        "translate_message",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("backend should not run")),
    )
    product_main.app.state.input_language = "en"
    product_main.app.state.acronyms = {"nato": "North Atlantic Treaty Organization"}

    response = client.post("/translate", json={"is_outgoing": True, "text": " NATO now "})

    assert response.status_code == 200
    assert response.json() == {"translation": "North Atlantic Treaty Organization now"}


def test_translate_returns_503_for_backend_runtime_failures(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        product_main,
        "translate_message",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("backend offline")),
    )
    product_main.app.state.input_language = "pt"
    product_main.app.state.translator = object()
    product_main.app.state.acronyms = {}

    response = client.post("/translate", json={"is_outgoing": False, "text": "Hello"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Translation backend unavailable: backend offline"}


def test_translate_returns_500_for_unexpected_errors(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        product_main,
        "translate_message",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad request")),
    )
    product_main.app.state.input_language = "pt"
    product_main.app.state.translator = object()
    product_main.app.state.acronyms = {}

    response = client.post("/translate", json={"is_outgoing": False, "text": "Hello"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Translation failed"}


def test_translate_rejects_blank_text(client: TestClient) -> None:
    product_main.app.state.input_language = "en"
    product_main.app.state.acronyms = {}

    response = client.post("/translate", json={"is_outgoing": False, "text": "   "})

    assert response.status_code == 422
