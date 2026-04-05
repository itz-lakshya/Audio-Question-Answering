from dataclasses import dataclass


@dataclass
class WindowPrediction:
    label: str
    score: float
    start: float
    end: float


@dataclass
class AudioEvent:
    label: str
    start: float
    end: float
    confidence: float


@dataclass
class QAResult:
    answer: str
    evidence: list[str]
    description: str = ""
