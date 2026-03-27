from pydantic import BaseModel, constr

NonEmptyStr = constr(strip_whitespace=True, min_length=1)


class InitializeRequest(BaseModel):
    language: NonEmptyStr


class TranslateRequest(BaseModel):
    is_outgoing: bool
    text: NonEmptyStr


class TranslateResponse(BaseModel):
    translation: NonEmptyStr
