from app.services.acronym_service import build_acronym_map, parse_acronyms


def test_build_acronym_map_lowercases_lookup_keys() -> None:
    acronym_map = build_acronym_map({"NATO": "North Atlantic Treaty Organization"})

    assert acronym_map == {"nato": "North Atlantic Treaty Organization"}


def test_parse_acronyms_replaces_tokens_case_insensitively_and_keeps_punctuation() -> None:
    acronym_map = {"nato": "North Atlantic Treaty Organization", "cbn": "chemical battalion"}

    parsed = parse_acronyms("NATO, move with CBN.", acronym_map)

    assert parsed == "North Atlantic Treaty Organization, move with chemical battalion."


def test_parse_acronyms_returns_original_text_when_lookup_is_empty() -> None:
    text = "Keep NATO in the report."

    assert parse_acronyms(text, {}) == text
