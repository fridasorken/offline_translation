from __future__ import annotations

from app.services.initialization_service import TranslationRuntime


def translate_message(translator: TranslationRuntime, is_outgoing: bool, text: str) -> str:
    """
    Translates given text. 
    
    Based on whether the message is outgoing or incoming, it is translated from 
    user language to english or from english to user language, respectively.

    Parameters
    ----------
    translator : TranslationRuntime
        The set of machine translation models configured for outgoing and incoming messages.
    is_outgoing : bool
        Whether the message is outgoing or not.
    text : str
        The text to be translated.

    Returns
    -------
    str
        The text translated to english if the message is outgoing, or to user language if incoming.
    """
    
    return translator.for_direction(is_outgoing).translate(text)
