from __future__ import annotations

import re

# Splits a token into (word, trailing_punctuation). E.g. "NATO." -> ("NATO", ".")
_TRAILING_PUNCT = re.compile(r"^(.*?)([.,!?;:]+)$")


def build_acronym_map(raw: dict[str, str]) -> dict[str, str]:
    """Pre-compute a lowercased acronym lookup map.

    Call this once at initialization time so we don't rebuild it per request.
    """
    return {k.lower(): v for k, v in raw.items()}


def parse_acronyms(text: str, acronym_map: dict[str, str]) -> str:
    """Replace acronym tokens in text using a pre-built lowercase acronym map.

    Handles trailing punctuation so e.g. "NATO." still matches "nato".
    """
    if not acronym_map:
        return text

    output_tokens: list[str] = []

    for token in text.split():
        m = _TRAILING_PUNCT.match(token)
        if m:
            word, punct = m.group(1), m.group(2)
        else:
            word, punct = token, ""

        replaced = acronym_map.get(word.lower(), word)
        output_tokens.append(replaced + punct)

    return " ".join(output_tokens)
