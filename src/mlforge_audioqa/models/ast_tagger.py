from __future__ import annotations

from typing import Iterable

import numpy as np
from transformers import pipeline

from ..schemas import WindowPrediction
from .base import AudioTagger


class ASTTagger(AudioTagger):
    """Window-level tagging using a pretrained AST model."""

    def __init__(self, labels: Iterable[str], top_k: int = 5) -> None:
        self.labels = list(labels)
        self.top_k = top_k
        self.label_aliases = self._build_aliases(self.labels)
        self.classifier = pipeline(
            task="audio-classification",
            model="MIT/ast-finetuned-audioset-10-10-0.4593",
            top_k=max(50, top_k),
        )

    @staticmethod
    def _build_aliases(labels: list[str]) -> dict[str, list[str]]:
        aliases: dict[str, list[str]] = {label: [label] for label in labels}

        def add(label: str, extra: list[str]) -> None:
            if label in aliases:
                aliases[label].extend(extra)

        add("drilling", ["drill", "power drill", "dentist drill", "construction drilling"])
        add("hammering", ["jackhammer", "hammer", "pounding", "construction hammering"])
        add("engine idling", ["engine", "motor", "idling"])
        add("car passing", ["car", "vehicle passing", "traffic"])
        add("dog bark", ["dog", "bark", "barking"])

        return aliases

    def predict_window(self, audio: np.ndarray, sample_rate: int, start: float, end: float) -> list[WindowPrediction]:
        raw = self.classifier({"array": audio, "sampling_rate": sample_rate})

        # Map model labels to dataset labels via simple lexical overlap.
        rows: list[WindowPrediction] = []
        for item in raw:
            pred_label = item["label"].lower()
            score = float(item["score"])
            for target in self.labels:
                aliases = self.label_aliases.get(target, [target])
                if any(a.lower() in pred_label or pred_label in a.lower() for a in aliases):
                    rows.append(WindowPrediction(label=target, score=score, start=start, end=end))

        rows = sorted(rows, key=lambda p: p.score, reverse=True)
        return rows[: self.top_k]
