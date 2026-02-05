from __future__ import annotations

from typing import Iterable, Sequence

from sacrebleu.metrics import BLEU, CHRF, TER

DEFAULT_METRICS = ("bleu", "chrf", "ter")


class MetricsEngine:
    def __init__(self) -> None:
        self._bleu = BLEU()
        self._chrf = CHRF()
        self._ter = TER()

    def compute(
        self,
        hypothesis: str,
        references: Sequence[str],
        metrics: Iterable[str] | None = None,
    ) -> dict[str, float]:
        refs = [ref for ref in references if ref]
        if not refs:
            return {}

        selected = [metric.lower() for metric in (metrics or DEFAULT_METRICS)]
        unknown = [metric for metric in selected if metric not in DEFAULT_METRICS]
        if unknown:
            raise ValueError(f"Unknown metrics requested: {', '.join(sorted(set(unknown)))}")

        results: dict[str, float] = {}
        if "bleu" in selected:
            results["bleu"] = self._bleu.sentence_score(hypothesis, refs).score
        if "chrf" in selected:
            results["chrf"] = self._chrf.sentence_score(hypothesis, refs).score
        if "ter" in selected:
            results["ter"] = self._ter.sentence_score(hypothesis, refs).score

        return results
