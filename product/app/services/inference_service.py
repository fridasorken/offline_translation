from __future__ import annotations

from app.services.initialization_service import TranslationRuntime


def translate_message(translator: TranslationRuntime, is_outgoing: bool, text: str) -> str:
    return translator.for_direction(is_outgoing).translate(text)
