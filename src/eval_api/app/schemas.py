from pydantic import BaseModel, Field, constr

NonEmptyStr = constr(strip_whitespace=True, min_length=1)


class TranslateRequest(BaseModel):
    src_lang: NonEmptyStr
    tgt_lang: NonEmptyStr
    source: NonEmptyStr
    model_id: NonEmptyStr


class TranslateResponse(BaseModel):
    model_id: NonEmptyStr
    translated_value: NonEmptyStr
    latency_ms: int = Field(ge=0)
