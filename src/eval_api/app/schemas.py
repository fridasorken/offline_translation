from pydantic import BaseModel, Field, constr
from typing import Optional

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


class EvaluateItem(BaseModel):
    source: NonEmptyStr
    reference: Optional[NonEmptyStr] = None
    item_id: Optional[NonEmptyStr] = None


class EvaluateRequest(BaseModel):
    model_id: NonEmptyStr
    src_lang: NonEmptyStr
    tgt_lang: NonEmptyStr
    items: list[EvaluateItem] = Field(min_length=1)
    metrics: Optional[list[NonEmptyStr]] = None


class EvaluateItemResult(BaseModel):
    source: NonEmptyStr
    reference: Optional[NonEmptyStr] = None
    translated_value: NonEmptyStr
    latency_ms: int = Field(ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)
    item_id: Optional[NonEmptyStr] = None


class EvaluateResponse(BaseModel):
    model_id: NonEmptyStr
    src_lang: NonEmptyStr
    tgt_lang: NonEmptyStr
    results: list[EvaluateItemResult]
    aggregates: dict[str, float] = Field(default_factory=dict)
    average_latency_ms: float = Field(ge=0)


class ModelInfo(BaseModel):
    """Information about an available translation model."""
    model_id: NonEmptyStr
    adapter: NonEmptyStr
    supported_pairs: list[tuple[str, str]]


class ModelsListResponse(BaseModel):
    """Response containing list of available models."""
    models: list[ModelInfo]
