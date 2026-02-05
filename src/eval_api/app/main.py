import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .metrics import MetricsEngine, REFERENCE_METRICS, REFERENCE_FREE_METRICS
from .registry import ModelRegistry
from .schemas import (
    EvaluateRequest,
    EvaluateResponse,
    TranslateRequest,
    TranslateResponse,
    ModelsListResponse,
    ModelInfo,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = ModelRegistry()
    try:
        registry.load()
    except Exception:
        logger.exception("Failed to load model registry")
        raise
    app.state.registry = registry
    yield


app = FastAPI(title="Offline Translation Eval API", version="0.1.0", lifespan=lifespan)
metrics_engine = MetricsEngine()

# CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/translate", response_model=TranslateResponse)
def translate(request: TranslateRequest) -> TranslateResponse:
    registry: ModelRegistry = app.state.registry

    try:
        registry.get_config(request.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="model_id not found")

    if not registry.is_supported_pair(request.model_id, request.src_lang, request.tgt_lang):
        raise HTTPException(status_code=400, detail="unsupported language pair")

    adapter = registry.get_adapter(request.model_id)

    start = time.perf_counter()
    try:
        translated = adapter.translate(request.src_lang, request.tgt_lang, request.source)
    except Exception:
        logger.exception("Translation failed for model_id=%s", request.model_id)
        raise HTTPException(status_code=500, detail="translation failed")
    latency_ms = int((time.perf_counter() - start) * 1000)

    logger.info(
        "translate model_id=%s src=%s tgt=%s latency_ms=%d",
        request.model_id,
        request.src_lang,
        request.tgt_lang,
        latency_ms,
    )

    return TranslateResponse(
        model_id=request.model_id,
        translated_value=translated,
        latency_ms=latency_ms,
    )


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    references: list[str] = []
    if request.reference:
        references.append(request.reference)
    if request.references:
        references.extend(request.references)

    if request.metrics:
        selected_metrics = [metric.lower() for metric in request.metrics]
    else:
        if references:
            selected_metrics = list(REFERENCE_METRICS)
        else:
            selected_metrics = list(REFERENCE_FREE_METRICS)

    try:
        metrics = metrics_engine.compute(
            request.source,
            request.hypothesis,
            references,
            selected_metrics,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return EvaluateResponse(metrics=metrics)


@app.get("/models", response_model=ModelsListResponse)
def list_models() -> ModelsListResponse:
    """List all available translation models and their supported language pairs."""
    registry: ModelRegistry = app.state.registry

    models_info = []
    for model_id in registry.list_models():
        config = registry.get_config(model_id)
        models_info.append(
            ModelInfo(
                model_id=config.model_id,
                adapter=config.adapter,
                supported_pairs=list(config.supported_pairs),
            )
        )

    return ModelsListResponse(models=models_info)
