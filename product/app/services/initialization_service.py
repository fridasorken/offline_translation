from app.config import load_product_config
from app.inference import OpusTranslator


class TranslationRuntime:
    """
    Runtime container for bidirectional translation between user language and English.

    This class initializes two translation models: 
    one for translating outgoing messages to English, and one for translating 
    incoming messages from English back to user language. 
    
    Both models are warmed up on initialization to keep latency consistent from first translation.

    Parameters
    ----------
    language : str
        The target language code. This language is used as the source language
        for outgoing translations (to English) and as the target language for
        incoming translations (from English).

    Attributes
    ----------
    sender : OpusTranslator
        Translator used for outgoing messages. Translates from the target language to English.
    receiver : OpusTranslator
        Translator used for incoming messages. Translates from English to the target language.
    """
    
    def __init__(self, language: str) -> None:
        self.sender = OpusTranslator(load_product_config(source_lang=language, target_lang="en"))
        self.receiver = OpusTranslator(load_product_config(source_lang="en", target_lang=language))

        self.sender.warmup()
        self.receiver.warmup()

    def for_direction(self, is_outgoing: bool) -> OpusTranslator:
        """
        Returns the appropriate translator based on message direction.

        Parameters
        ----------
        is_outgoing : bool
            If True, returns the translator for outgoing messages (user language -> English).
            If False, returns the translator for incoming messages (English -> user language).

        Returns
        -------
        OpusTranslator
            The translator for the specified direction.
        """
              
        if is_outgoing:
            return self.sender
        return self.receiver


def load_translator(language: str) -> TranslationRuntime:
    """
    Creates a runtime container for bidirectional translation between the given user language and English.

    Parameters
    ----------
    language : str
        The target language code. This language is used as the source language
        for outgoing translations (to English) and as the target language for
        incoming translations (from English). 

    Returns
    -------
    TranslationRuntime
        The set of machine translation models configured to the given user language.
    """
    
    return TranslationRuntime(language=language)
