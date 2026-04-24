from __future__ import annotations

import threading
from collections.abc import Iterator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.main as product_main
from app.config import MAX_TRANSLATION_CHARS
from app.schemas import InitializeRequest, TranslateRequest


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


def test_initialize_allows_language_switching_by_default(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtimes: list[str] = []

    def fake_load_translator(language: str) -> object:
        runtimes.append(language)
        return object()

    monkeypatch.setattr(product_main, "load_translator", fake_load_translator)
    monkeypatch.setattr(
        product_main.pd,
        "read_excel",
        lambda *args, **kwargs: pd.DataFrame({"acronym": [], "expansion": []}),
    )

    first_response = client.post("/initialize", json={"language": "de"})
    second_response = client.post("/initialize", json={"language": "pt"})

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert second_response.json() == {"status": "ok", "language": "pt"}
    assert product_main.app.state.input_language == "pt"
    assert runtimes == ["de", "pt"]


def test_initialize_english_clears_previous_translator(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    product_main.app.state.input_language = "pt"
    product_main.app.state.translator = object()
    product_main.app.state.acronyms = {"nato": "North Atlantic Treaty Organization"}
    monkeypatch.setattr(
        product_main.pd,
        "read_excel",
        lambda *args, **kwargs: pd.DataFrame({"acronym": [], "expansion": []}),
    )

    response = client.post("/initialize", json={"language": "en"})

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "language": "en"}
    assert not hasattr(product_main.app.state, "translator")
    assert product_main.app.state.acronyms == {}


def test_initialize_can_reject_reinitialization_when_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(product_main, "ALLOW_REINITIALIZE", False)
    monkeypatch.setattr(
        product_main.pd,
        "read_excel",
        lambda *args, **kwargs: pd.DataFrame({"acronym": [], "expansion": []}),
    )

    first_response = client.post("/initialize", json={"language": "en"})
    second_response = client.post("/initialize", json={"language": "en"})

    assert first_response.status_code == 200
    assert second_response.status_code == 409
    assert second_response.json() == {"detail": "Runtime is already initialized"}


def test_initialize_waits_for_active_translation_to_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translation_started = threading.Event()
    release_translation = threading.Event()
    initialize_started = threading.Event()
    results: dict[str, object] = {}
    errors: dict[str, Exception] = {}
    runtime = object()

    def fake_translate_message(translator: object, is_outgoing: bool, text: str) -> str:
        translation_started.set()
        assert release_translation.wait(timeout=1)
        return "Hei"

    def fake_load_translator(language: str) -> object:
        initialize_started.set()
        return runtime

    monkeypatch.setattr(product_main, "translate_message", fake_translate_message)
    monkeypatch.setattr(product_main, "load_translator", fake_load_translator)
    monkeypatch.setattr(
        product_main.pd,
        "read_excel",
        lambda *args, **kwargs: pd.DataFrame({"acronym": [], "expansion": []}),
    )

    product_main.app.state.input_language = "pt"
    product_main.app.state.translator = object()
    product_main.app.state.acronyms = {}

    def run_translate() -> None:
        try:
            results["translate"] = product_main.translate(
                TranslateRequest(is_outgoing=False, text="Hello")
            )
        except Exception as exc:
            errors["translate"] = exc

    def run_initialize() -> None:
        try:
            results["initialize"] = product_main.initialize(InitializeRequest(language="de"))
        except Exception as exc:
            errors["initialize"] = exc

    translate_thread = threading.Thread(target=run_translate)
    initialize_thread = threading.Thread(target=run_initialize)

    translate_thread.start()
    assert translation_started.wait(timeout=1)

    initialize_thread.start()
    assert not initialize_started.wait(timeout=0.1)

    release_translation.set()
    translate_thread.join(timeout=1)
    initialize_thread.join(timeout=1)

    assert not translate_thread.is_alive()
    assert not initialize_thread.is_alive()
    assert not errors
    assert initialize_started.is_set()
    assert results["translate"].translation == "Hei"
    assert results["initialize"] == {"status": "ok", "language": "de"}
    assert product_main.app.state.input_language == "de"
    assert product_main.app.state.translator is runtime


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


def test_translate_rejects_too_large_text(client: TestClient) -> None:
    product_main.app.state.input_language = "en"
    product_main.app.state.acronyms = {}

    response = client.post(
        "/translate",
        json={"is_outgoing": False, "text": "x" * (MAX_TRANSLATION_CHARS + 1)},
    )

    assert response.status_code == 422


def test_translate_waits_for_active_translation_to_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_translation_started = threading.Event()
    release_first_translation = threading.Event()
    second_translation_started = threading.Event()
    call_order: list[str] = []
    results: dict[str, object] = {}
    errors: dict[str, Exception] = {}

    def fake_translate_message(translator: object, is_outgoing: bool, text: str) -> str:
        call_order.append(text)
        if text == "Hello one":
            first_translation_started.set()
            assert release_first_translation.wait(timeout=1)
        else:
            second_translation_started.set()

        return f"{text} translated"

    monkeypatch.setattr(product_main, "translate_message", fake_translate_message)
    product_main.app.state.input_language = "pt"
    product_main.app.state.translator = object()
    product_main.app.state.acronyms = {}

    def run_translate(name: str, text: str) -> None:
        try:
            results[name] = product_main.translate(TranslateRequest(is_outgoing=False, text=text))
        except Exception as exc:
            errors[name] = exc

    first_thread = threading.Thread(target=run_translate, args=("first", "Hello one"))
    second_thread = threading.Thread(target=run_translate, args=("second", "Hello two"))

    first_thread.start()
    assert first_translation_started.wait(timeout=1)

    second_thread.start()
    assert not second_translation_started.wait(timeout=0.1)

    release_first_translation.set()
    first_thread.join(timeout=1)
    second_thread.join(timeout=1)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert not errors
    assert second_translation_started.is_set()
    assert results["first"].translation == "Hello one translated"
    assert results["second"].translation == "Hello two translated"
    assert call_order == ["Hello one", "Hello two"]
