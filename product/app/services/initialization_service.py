from app.config import load_product_config
from app.inference import OpusTranslator


class TranslationRuntime:
    def __init__(self, language: str) -> None:
        self.sender = OpusTranslator(load_product_config(source_lang=language, target_lang="en"))
        self.receiver = OpusTranslator(load_product_config(source_lang="en", target_lang=language))

        self.sender.warmup()
        self.receiver.warmup()

    def resolve_direction(self, is_sender: bool) -> OpusTranslator:
        if is_sender:
            return self.sender
        return self.receiver


def load_translator(language: str) -> TranslationRuntime:
    return TranslationRuntime(language=language)
