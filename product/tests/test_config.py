import pytest

from app.config import (
    ALLOW_REINITIALIZE,
    API_HOST,
    MODEL_CACHE_DIR,
    _normalize_lang,
    _read_bool_env,
    _read_int_env,
    load_product_config,
)


@pytest.mark.parametrize(
    ("raw_code", "expected"),
    [
        (" no ", "nob"),
        ("GER", "de"),
        ("pt", "pt"),
    ],
)
def test_normalize_lang_maps_supported_aliases(raw_code: str, expected: str) -> None:
    assert _normalize_lang(raw_code) == expected


def test_load_product_config_resolves_aliases_for_supported_pairs() -> None:
    config = load_product_config(source_lang="nor", target_lang="en")

    assert config.source_lang == "nob"
    assert config.target_lang == "en"
    assert config.model_id == "opus-tc-big-nob-en-military"
    assert config.model_path == "MariusBerg/opus-tc-big-nob-en-military"
    assert config.ct2_cache_dir == MODEL_CACHE_DIR
    assert config.use_target_tag is False


def test_load_product_config_rejects_unsupported_language_pairs() -> None:
    with pytest.raises(ValueError, match="Unsupported language pair 'fr->en'"):
        load_product_config(source_lang="fr", target_lang="en")


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
    ],
)
def test_read_bool_env_parses_truthy_and_falsey_values(
    monkeypatch: pytest.MonkeyPatch, raw_value: str, expected: bool
) -> None:
    monkeypatch.setenv("TEST_BOOL_ENV", raw_value)

    assert _read_bool_env("TEST_BOOL_ENV", default=not expected) is expected


def test_read_bool_env_returns_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_BOOL_ENV", raising=False)

    assert _read_bool_env("TEST_BOOL_ENV", default=True) is True
    assert _read_bool_env("TEST_BOOL_ENV", default=False) is False


def test_api_host_defaults_to_localhost() -> None:
    assert API_HOST == "127.0.0.1"


def test_reinitialize_defaults_to_enabled() -> None:
    assert ALLOW_REINITIALIZE is True


def test_read_int_env_parses_integer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_INT_ENV", "42")

    assert _read_int_env("TEST_INT_ENV", default=1, minimum=1) == 42


def test_read_int_env_validates_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_INT_ENV", "0")

    with pytest.raises(ValueError, match="TEST_INT_ENV must be at least 1"):
        _read_int_env("TEST_INT_ENV", default=1, minimum=1)
