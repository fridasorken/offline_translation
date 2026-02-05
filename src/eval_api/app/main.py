import logging
import os
import time
from statistics import mean, median, pstdev
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from memory_profiler import memory_usage
import psutil
from fastapi.middleware.cors import CORSMiddleware

from .metrics import MetricsEngine, REFERENCE_METRICS, REFERENCE_FREE_METRICS
from .registry import ModelRegistry
from .schemas import (
    EvaluateItemResult,
    EvaluateRequest,
    EvaluateResponse,
    TranslateRequest,
    TranslateResponse,
    ModelsListResponse,
    ModelInfo,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_PROFILE_RESOURCES = os.getenv("EVAL_PROFILE_RESOURCES", "true").lower() == "true"
EVAL_MEM_INTERVAL = float(os.getenv("EVAL_MEM_INTERVAL", "0.1"))
EVAL_MEM_BACKEND = os.getenv("EVAL_MEM_BACKEND", "psutil")


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


def _cpu_percent_per_core(start_cpu: object, end_cpu: object, wall: float) -> float:
    """Estimate CPU usage over the full translation window (per logical core)."""
    if wall <= 0:
        return 0.0
    cpu_delta = (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)
    cores = psutil.cpu_count(logical=True) or 1
    return max(0.0, (cpu_delta / wall) * 100.0 / cores)


def _translate_with_resources(adapter, src_lang: str, tgt_lang: str, text: str) -> tuple[str, int, float | None, float | None, float | None]:
    """
    Translate and optionally profile resource usage.

    RAM is sampled at a fixed interval during translation to estimate mean and peak.
    CPU usage is derived from CPU time deltas over the full translation duration.
    """
    process = psutil.Process(os.getpid())

    if not EVAL_PROFILE_RESOURCES:
        start = time.perf_counter()
        translated = adapter.translate(src_lang, tgt_lang, text)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return translated, latency_ms, None, None, None

    start_cpu = process.cpu_times()
    start_time = time.perf_counter()
    mem_samples, translated = memory_usage(
        (adapter.translate, (src_lang, tgt_lang, text), {}),
        interval=EVAL_MEM_INTERVAL,
        backend=EVAL_MEM_BACKEND,
        retval=True,
    )
    wall = time.perf_counter() - start_time
    end_cpu = process.cpu_times()
    latency_ms = int(wall * 1000)

    ram_mean = float(mean(mem_samples)) if mem_samples else None
    ram_peak = float(max(mem_samples)) if mem_samples else None
    cpu_percent = _cpu_percent_per_core(start_cpu, end_cpu, wall)

    return translated, latency_ms, cpu_percent, ram_mean, ram_peak

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
    registry: ModelRegistry = app.state.registry

    try:
        registry.get_config(request.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="model_id not found")

    if not registry.is_supported_pair(request.model_id, request.src_lang, request.tgt_lang):
        raise HTTPException(status_code=400, detail="unsupported language pair")

    adapter = registry.get_adapter(request.model_id)

    if request.metrics:
        selected_metrics = [metric.lower() for metric in request.metrics]
    else:
        selected_metrics = list(REFERENCE_METRICS)

    if not request.items:
        raise HTTPException(status_code=400, detail="items must not be empty")

    results: list[EvaluateItemResult] = []
    metric_buckets: dict[str, list[float]] = {}
    latency_values: list[int] = []

    for item in request.items:
        if not item.reference and any(metric in REFERENCE_METRICS for metric in selected_metrics):
            raise HTTPException(
                status_code=400,
                detail="reference required for requested metrics",
            )

        try:
            translated, latency_ms, cpu_percent, ram_mean, ram_peak = _translate_with_resources(
                adapter,
                request.src_lang,
                request.tgt_lang,
                item.source,
            )
        except Exception:
            logger.exception("Evaluation translation failed for model_id=%s", request.model_id)
            raise HTTPException(status_code=500, detail="translation failed")
        latency_values.append(latency_ms)

        references = [item.reference] if item.reference else []
        try:
            metrics = metrics_engine.compute(
                item.source,
                translated,
                references,
                selected_metrics if references else list(REFERENCE_FREE_METRICS),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        for metric_name, metric_value in metrics.items():
            metric_buckets.setdefault(metric_name, []).append(metric_value)

        results.append(
            EvaluateItemResult(
                source=item.source,
                reference=item.reference,
                translated_value=translated,
                latency_ms=latency_ms,
                metrics=metrics,
                item_id=item.item_id,
                cpu_percent_per_core=cpu_percent,
                ram_mean_mb=ram_mean,
                ram_peak_mb=ram_peak,
            )
        )

    aggregates: dict[str, float] = {}
    for metric_name, values in metric_buckets.items():
        aggregates[f"{metric_name}_mean"] = mean(values)
        aggregates[f"{metric_name}_median"] = median(values)
        aggregates[f"{metric_name}_stdev"] = pstdev(values)

    cpu_values = [item.cpu_percent_per_core for item in results if item.cpu_percent_per_core is not None]
    if cpu_values:
        aggregates["cpu_percent_per_core_mean"] = mean(cpu_values)
        aggregates["cpu_percent_per_core_median"] = median(cpu_values)
        aggregates["cpu_percent_per_core_stdev"] = pstdev(cpu_values)

    ram_mean_values = [item.ram_mean_mb for item in results if item.ram_mean_mb is not None]
    if ram_mean_values:
        aggregates["ram_mean_mb_mean"] = mean(ram_mean_values)
        aggregates["ram_mean_mb_median"] = median(ram_mean_values)
        aggregates["ram_mean_mb_stdev"] = pstdev(ram_mean_values)

    ram_peak_values = [item.ram_peak_mb for item in results if item.ram_peak_mb is not None]
    if ram_peak_values:
        aggregates["ram_peak_mb_mean"] = mean(ram_peak_values)
        aggregates["ram_peak_mb_median"] = median(ram_peak_values)
        aggregates["ram_peak_mb_stdev"] = pstdev(ram_peak_values)
    average_latency_ms = mean(latency_values) if latency_values else 0.0

    return EvaluateResponse(
        model_id=request.model_id,
        src_lang=request.src_lang,
        tgt_lang=request.tgt_lang,
        results=results,
        aggregates=aggregates,
        average_latency_ms=average_latency_ms,
    )


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
