from __future__ import annotations

from collections import defaultdict

from .audio.events import predictions_to_events
from .audio.segmenter import iter_windows, load_audio
from .config import PipelineConfig
from .models.fusion import FusedTagger
from .qa.description import generate_scene_description, generate_transparent_trace
from .qa.parser import parse_question
from .qa.reasoner import answer_from_events
from .schemas import QAResult, WindowPrediction


class AudioQAPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.tagger = FusedTagger(labels=config.labels, top_k=config.top_k_per_window)

    def extract_events(self, audio_path: str):
        all_predictions = self.extract_window_predictions(audio_path)

        return predictions_to_events(
            predictions=all_predictions,
            default_threshold=self.config.default_threshold,
            per_label_threshold=self.config.per_label_threshold,
            min_event_duration=self.config.min_event_duration,
            merge_gap=self.config.event_merge_gap,
        )

    def extract_window_predictions(self, audio_path: str) -> list[WindowPrediction]:
        audio = load_audio(audio_path, self.config.sample_rate)
        all_predictions: list[WindowPrediction] = []

        for start, end, chunk in iter_windows(
            audio,
            sample_rate=self.config.sample_rate,
            window_seconds=self.config.window_seconds,
            hop_seconds=self.config.hop_seconds,
        ):
            all_predictions.extend(
                self.tagger.predict_window(
                    audio=chunk,
                    sample_rate=self.config.sample_rate,
                    start=start,
                    end=end,
                )
            )

        return all_predictions

    def clip_label_scores(self, audio_path: str) -> dict[str, float]:
        predictions = self.extract_window_predictions(audio_path)
        score_map: dict[str, float] = defaultdict(float)
        for row in predictions:
            score_map[row.label] = max(score_map[row.label], row.score)

        return dict(score_map)

    def predict_clip_label(self, audio_path: str) -> str | None:
        score_map = self.clip_label_scores(audio_path)
        if not score_map:
            return None
        return max(score_map.items(), key=lambda x: x[1])[0]

    def answer(self, audio_path: str, question: str) -> QAResult:
        events = self.extract_events(audio_path)
        scene_description = generate_scene_description(events)
        parsed = parse_question(question, known_labels=self.config.labels)
        result = answer_from_events(
            parsed,
            events,
            question=question,
            scene_description=scene_description,
        )
        result.description = generate_transparent_trace(
            question=question,
            scene_description=scene_description,
            answer=result.answer,
            events=events,
        )
        return result
