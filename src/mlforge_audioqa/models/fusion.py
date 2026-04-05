from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..schemas import WindowPrediction
from .ast_tagger import ASTTagger
from .clap_ranker import CLAPTagger


class FusedTagger:
    """Simple weighted fusion of AST and CLAP window predictions."""

    def __init__(self, labels: list[str], top_k: int = 5, ast_weight: float = 0.30, clap_weight: float = 0.70) -> None:
        self.ast = ASTTagger(labels=labels, top_k=top_k)
        self.clap = CLAPTagger(labels=labels, top_k=top_k)
        self.top_k = top_k
        self.ast_weight = ast_weight
        self.clap_weight = clap_weight
        self.class_calibration = {
            "drilling": 1.25,
            "hammering": 1.25,
            "engine idling": 0.82,
            "car horn": 0.85,
            "car passing": 0.88,
        }

    def predict_window(self, audio: np.ndarray, sample_rate: int, start: float, end: float) -> list[WindowPrediction]:
        ast_preds = self.ast.predict_window(audio, sample_rate, start, end)
        clap_preds = self.clap.predict_window(audio, sample_rate, start, end)

        score_map: dict[str, float] = defaultdict(float)
        for p in ast_preds:
            score_map[p.label] += self.ast_weight * p.score
        for p in clap_preds:
            score_map[p.label] += self.clap_weight * p.score

        for label in list(score_map.keys()):
            score_map[label] *= self.class_calibration.get(label, 1.0)

        rows = [
            WindowPrediction(label=label, score=score, start=start, end=end)
            for label, score in score_map.items()
        ]
        rows = sorted(rows, key=lambda p: p.score, reverse=True)
        return rows[: self.top_k]
