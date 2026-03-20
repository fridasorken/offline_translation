from __future__ import annotations

from app.services.initialization_service import TranslationRuntime


def translate_message(translator: TranslationRuntime, sender: bool, text: str) -> str:
    return translator.resolve_direction(sender).translate(text)