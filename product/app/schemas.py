from pydantic import BaseModel, constr

from app.config import MAX_LANGUAGE_CODE_CHARS, MAX_TRANSLATION_CHARS

LanguageCode = constr(strip_whitespace=True, min_length=1, max_length=MAX_LANGUAGE_CODE_CHARS)
NonEmptyStr = constr(strip_whitespace=True, min_length=1)
TranslationText = constr(strip_whitespace=True, min_length=1, max_length=MAX_TRANSLATION_CHARS)


class InitializeRequest(BaseModel):
    """
    Request payload for initializing a translation runtime.

    Parameters
    ----------
    language : LanguageCode
        Target language code used for translations to and from English.
    """

    language: LanguageCode


class TranslateRequest(BaseModel):
    """
    Request payload for performing a translation.

    Parameters
    ----------
    is_outgoing : bool
        Direction of the message.
        If True, returns the translator for outgoing messages (user language -> English).
        If False, returns the translator for incoming messages (English -> user language).

    text : TranslationText
        The text to be translated.
    """

    is_outgoing: bool
    text: TranslationText


class TranslateResponse(BaseModel):
    """
    Response payload containing the translated text.

    Parameters
    ----------
    translation : NonEmptyStr
        The resulting translated text.
    """

    translation: NonEmptyStr
