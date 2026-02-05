from pydantic import BaseModel, Field, constr
from typing import Optional

NonEmptyStr = constr(strip_whitespace=True, min_length=1)


class TranslateRequest(BaseModel):
    src_lang: NonEmptyStr
    tgt_lang: NonEmptyStr
    source: NonEmptyStr
    model_id: NonEmptyStr
    reference: Optional[NonEmptyStr] = None
    references: Optional[list[NonEmptyStr]] = None
    metrics: Optional[list[NonEmptyStr]] = None


class TranslateResponse(BaseModel):
    model_id: NonEmptyStr
    translated_value: NonEmptyStr
    latency_ms: int = Field(ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    """Information about an available translation model."""
    model_id: NonEmptyStr
    adapter: NonEmptyStr
    supported_pairs: list[tuple[str, str]]


class ModelsListResponse(BaseModel):
    """Response containing list of available models."""
    models: list[ModelInfo]
