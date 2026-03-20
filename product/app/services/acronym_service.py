from __future__ import annotations


def parse_acronyms(text: str, acronyms: dict[str, str]) -> str:
    """Replace acronym tokens in text using the loaded acronym->expansion map."""
    if not acronyms:
        return text

    lower_map = {k.lower(): v for k, v in acronyms.items()}
    output_tokens: list[str] = []

    for token in text.split():
        output_tokens.append(lower_map.get(token.lower(), token))

    return " ".join(output_tokens)
