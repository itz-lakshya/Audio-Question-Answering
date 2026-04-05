from __future__ import annotations

from collections import defaultdict

from ..schemas import AudioEvent, WindowPrediction


def _threshold_for(label: str, default_threshold: float, per_label_threshold: dict[str, float]) -> float:
    return per_label_threshold.get(label, default_threshold)


def predictions_to_events(
    predictions: list[WindowPrediction],
    default_threshold: float,
    per_label_threshold: dict[str, float],
    min_event_duration: float,
    merge_gap: float,
) -> list[AudioEvent]:
    grouped: dict[str, list[WindowPrediction]] = defaultdict(list)
    for pred in predictions:
        threshold = _threshold_for(pred.label, default_threshold, per_label_threshold)
        if pred.score >= threshold:
            grouped[pred.label].append(pred)

    events: list[AudioEvent] = []
    for label, rows in grouped.items():
        rows = sorted(rows, key=lambda p: p.start)
        if not rows:
            continue

        current_start = rows[0].start
        current_end = rows[0].end
        scores = [rows[0].score]

        for row in rows[1:]:
            if row.start - current_end <= merge_gap:
                current_end = max(current_end, row.end)
                scores.append(row.score)
                continue

            if current_end - current_start >= min_event_duration:
                events.append(
                    AudioEvent(
                        label=label,
                        start=current_start,
                        end=current_end,
                        confidence=sum(scores) / len(scores),
                    )
                )

            current_start = row.start
            current_end = row.end
            scores = [row.score]

        if current_end - current_start >= min_event_duration:
            events.append(
                AudioEvent(
                    label=label,
                    start=current_start,
                    end=current_end,
                    confidence=sum(scores) / len(scores),
                )
            )

    return sorted(events, key=lambda e: e.start)
