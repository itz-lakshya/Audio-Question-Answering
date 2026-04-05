from __future__ import annotations

from collections import Counter

from ..schemas import AudioEvent


def generate_grounded_description(question: str, answer: str, events: list[AudioEvent], max_events: int = 12) -> str:
    if not events:
        return (
            "The system is not working for this query because no reliable events were extracted from the audio. "
            "No grounded explanation can be produced."
        )

    strongest = sorted(events, key=lambda e: e.confidence, reverse=True)[:max_events]
    strongest_bits = [
        f"{e.label} ({e.start:.2f}s-{e.end:.2f}s, confidence {e.confidence:.2f})"
        for e in strongest
    ]

    timeline = sorted(events, key=lambda e: e.start)[:max_events]
    timeline_bits = [f"{e.start:.2f}s-{e.end:.2f}s: {e.label}" for e in timeline]

    return (
        f"For the question '{question}', the grounded answer is: {answer} "
        "Most confident detected events are: "
        + "; ".join(strongest_bits)
        + ". Full timeline with timestamps: "
        + " | ".join(timeline_bits)
        + "."
    )


def generate_scene_description(events: list[AudioEvent], max_events: int = 12) -> str:
    if not events:
        return "No reliable sound events were detected from the audio."

    ordered = sorted(events, key=lambda e: e.start)
    label_counts = Counter(e.label for e in ordered)
    labels = ", ".join(f"{label} x{count}" for label, count in sorted(label_counts.items()))
    timeline = " | ".join(f"{e.start:.2f}s-{e.end:.2f}s: {e.label}" for e in ordered[:max_events])
    return f"Detected sounds summary: {labels}. Timeline: {timeline}."


def generate_transparent_trace(question: str, scene_description: str, answer: str, events: list[AudioEvent]) -> str:
    return (
        f"Stage 1 - Audio to Events: extracted {len(events)} event(s).\n"
        f"Stage 2 - Events to Description: {scene_description}\n"
        f"Stage 3 - Description + Question to Answer: question='{question}' | answer='{answer}'"
    )
