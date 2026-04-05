from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..schemas import WindowPrediction


class AudioTagger(ABC):
    @abstractmethod
    def predict_window(self, audio: np.ndarray, sample_rate: int, start: float, end: float) -> list[WindowPrediction]:
        raise NotImplementedError
