import pytest

from app.config import MODEL_CACHE_DIR, _normalize_lang, load_product_config


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
