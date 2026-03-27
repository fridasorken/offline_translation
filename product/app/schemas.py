from pydantic import BaseModel, constr

NonEmptyStr = constr(strip_whitespace=True, min_length=1)


class InitializeRequest(BaseModel):
    """
    Request payload for initializing a translation runtime.

    Parameters
    ----------
    language : NonEmptyStr
        Target language code used for translations to and from English.
    """
    
    language: NonEmptyStr


class TranslateRequest(BaseModel):
    """
    Request payload for performing a translation.

    Parameters
    ----------
    is_outgoing : bool
        Direction of the message. 
        If True, returns the translator for outgoing messages (user language -> English).
        If False, returns the translator for incoming messages (English -> user language).
    
    text : NonEmptyStr
        The text to be translated.
    """
    
    is_outgoing: bool
    text: NonEmptyStr


class TranslateResponse(BaseModel):
    """
    Response payload containing the translated text.

    Parameters
    ----------
    translation : NonEmptyStr
        The resulting translated text.
    """
    
    translation: NonEmptyStr
