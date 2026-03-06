from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Iterable, Sequence

from sacrebleu.metrics import BLEU, CHRF, TER

REFERENCE_METRICS = ("bleu", "chrf", "ter", "comet")
REFERENCE_FREE_METRICS = ("cometkiwi",)
ALL_METRICS = (*REFERENCE_METRICS, *REFERENCE_FREE_METRICS)

DEFAULT_COMET_MODEL = os.getenv("COMET_MODEL_NAME", "Unbabel/wmt22-comet-da")
DEFAULT_COMET_KIWI_MODEL = os.getenv("COMET_KIWI_MODEL_NAME", "Unbabel/wmt22-cometkiwi-da")
DEFAULT_COMET_BATCH_SIZE = int(os.getenv("COMET_BATCH_SIZE", "8"))
DEFAULT_COMET_GPUS = int(os.getenv("COMET_GPUS", "0"))
DEFAULT_COMET_NUM_WORKERS = int(os.getenv("COMET_NUM_WORKERS", "1"))


class CometScorer:
    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int,
        gpus: int,
        num_workers: int,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.gpus = gpus
        self.num_workers = num_workers
        self._model = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        # Double-checked locking to prevent concurrent model loading
        if self._model is not None:
            return

        with self._lock:
            # Check again inside lock in case another thread loaded while waiting
            if self._model is not None:
                return

            from comet import download_model, load_from_checkpoint

            candidate = Path(self.model_name_or_path)
            if candidate.exists():
                checkpoint_path = str(candidate)
            else:
                checkpoint_path = download_model(self.model_name_or_path)
            self._model = load_from_checkpoint(checkpoint_path)

    def score_reference(self, source: str, hypothesis: str, reference: str) -> float:
        self._load()
        data = [{"src": source, "mt": hypothesis, "ref": reference}]
        output = self._model.predict(
            data,
            batch_size=self.batch_size,
            gpus=self.gpus,
            num_workers=self.num_workers,
        )
        return float(output.scores[0])

    def score_reference_free(self, source: str, hypothesis: str) -> float:
        self._load()
        data = [{"src": source, "mt": hypothesis}]
        output = self._model.predict(
            data,
            batch_size=self.batch_size,
            gpus=self.gpus,
            num_workers=self.num_workers,
        )
        return float(output.scores[0])


class MetricsEngine:
    def __init__(
        self,
        comet_model: str = DEFAULT_COMET_MODEL,
        cometkiwi_model: str = DEFAULT_COMET_KIWI_MODEL,
        comet_batch_size: int = DEFAULT_COMET_BATCH_SIZE,
        comet_gpus: int = DEFAULT_COMET_GPUS,
        comet_num_workers: int = DEFAULT_COMET_NUM_WORKERS,
    ) -> None:
        self._bleu = BLEU(effective_order=True)
        self._chrf = CHRF()
        self._ter = TER()
        self._comet = CometScorer(
            comet_model,
            comet_batch_size,
            comet_gpus,
            comet_num_workers,
        )
        self._cometkiwi = CometScorer(
            cometkiwi_model,
            comet_batch_size,
            comet_gpus,
            comet_num_workers,
        )

    def compute(
        self,
        source: str,
        hypothesis: str,
        references: Sequence[str],
        metrics: Iterable[str],
    ) -> dict[str, float]:
        selected = [metric.lower() for metric in metrics]
        unknown = [metric for metric in selected if metric not in ALL_METRICS]
        if unknown:
            raise ValueError(f"Unknown metrics requested: {', '.join(sorted(set(unknown)))}")

        refs = [ref for ref in references if ref]

        results: dict[str, float] = {}
        if any(metric in selected for metric in REFERENCE_METRICS):
            if not refs:
                raise ValueError("reference or references required for requested metrics")
            if "bleu" in selected:
                results["bleu"] = self._bleu.sentence_score(hypothesis, refs).score
            if "chrf" in selected:
                results["chrf"] = self._chrf.sentence_score(hypothesis, refs).score
            if "ter" in selected:
                results["ter"] = self._ter.sentence_score(hypothesis, refs).score
            if "comet" in selected:
                results["comet"] = self._comet.score_reference(source, hypothesis, refs[0])

        if "cometkiwi" in selected:
            results["cometkiwi"] = self._cometkiwi.score_reference_free(source, hypothesis)

        return results
