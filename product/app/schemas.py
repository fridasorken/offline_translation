from pydantic import BaseModel, constr

NonEmptyStr = constr(strip_whitespace=True, min_length=1)


class InitializeRequest(BaseModel):
    language: NonEmptyStr


class TranslateRequest(BaseModel):
    sender: bool
    text: NonEmptyStr


class TranslateResponse(BaseModel):
    translation: NonEmptyStr
