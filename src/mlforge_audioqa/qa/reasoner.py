from __future__ import annotations

from collections import defaultdict

from ..schemas import AudioEvent, QAResult
from .description import generate_grounded_description
from .extractive import ExtractiveQAModel
from .parser import ParsedQuestion


_QA_MODEL = ExtractiveQAModel(model_name="deepset/roberta-base-squad2")


def _format_event(e: AudioEvent) -> str:
    return f"{e.label}: {e.start:.2f}s-{e.end:.2f}s (conf={e.confidence:.2f})"


def _top_events(events: list[AudioEvent], n: int = 8) -> list[AudioEvent]:
    return sorted(events, key=lambda e: e.confidence, reverse=True)[:n]


def _build_extractive_context(
    events: list[AudioEvent],
    scene_description: str = "",
    max_events: int = 24,
) -> str:
    by_label: dict[str, list[AudioEvent]] = defaultdict(list)
    for event in events:
        by_label[event.label.lower()].append(event)

    lines: list[str] = []

    for event in sorted(events, key=lambda e: e.start)[:max_events]:
        lines.append(
            f"event: {event.label}; start: {event.start:.2f}; end: {event.end:.2f}; confidence: {event.confidence:.2f}"
        )

    labels = sorted({e.label for e in events})
    lines.append("detected_labels: " + ", ".join(labels))
    if scene_description:
        lines.append("scene_description: " + scene_description)

    for label, rows in sorted(by_label.items()):
        first_start = min(r.start for r in rows)
        first_end = min(rows, key=lambda r: r.start).end
        lines.append(
            f"summary: {label} occurred {len(rows)} time(s); first_occurrence_start: {first_start:.2f}; first_occurrence_end: {first_end:.2f}"
        )

    return "\n".join(lines)


def _try_extractive_answer(
    question: str,
    events: list[AudioEvent],
    scene_description: str = "",
    min_score: float = 0.20,
) -> str | None:
    if not question.strip() or not events:
        return None

    context = _build_extractive_context(events, scene_description=scene_description)
    try:
        pred = _QA_MODEL.answer(question=question, context=context)
    except Exception:
        return None
    if pred.score < min_score or not pred.answer:
        return None

    # Basic guard against null/no-answer placeholders.
    if pred.answer.lower() in {"", "[cls]", "unknown", "none"}:
        return None

    return pred.answer


def answer_from_events(
    parsed: ParsedQuestion,
    events: list[AudioEvent],
    question: str = "",
    scene_description: str = "",
) -> QAResult:
    if not events:
        answer = "It is not working: no confident sound events were detected."
        return QAResult(
            answer=answer,
            evidence=[],
            description=generate_grounded_description("unknown", answer, events),
        )

    extractive_answer = _try_extractive_answer(question, events, scene_description=scene_description)

    by_label: dict[str, list[AudioEvent]] = defaultdict(list)
    for event in events:
        by_label[event.label.lower()].append(event)

    if parsed.qtype == "what":
        labels = sorted({e.label for e in events})
        evidence = [_format_event(e) for e in _top_events(events, 8)]
        answer = extractive_answer if extractive_answer else "Detected sounds: " + ", ".join(labels)
        return QAResult(
            answer=answer,
            evidence=evidence,
            description=generate_grounded_description("what is happening", answer, events),
        )

    if parsed.qtype == "count":
        if extractive_answer:
            evidence = [_format_event(e) for e in _top_events(events, 8)]
            return QAResult(
                answer=extractive_answer,
                evidence=evidence,
                description=generate_grounded_description(question or "count", extractive_answer, events),
            )
        if not parsed.target_a:
            answer = "It is not working: the target sound for counting is missing."
            evidence = [_format_event(e) for e in _top_events(events, 5)]
            return QAResult(
                answer=answer,
                evidence=evidence,
                description=generate_grounded_description("count", answer, events),
            )
        target = parsed.target_a.lower()
        count = len(by_label.get(target, []))
        evidence = [_format_event(e) for e in _top_events(by_label.get(target, []), 8)]
        answer = f"{parsed.target_a} occurred {count} time(s)."
        return QAResult(
            answer=answer,
            evidence=evidence,
            description=generate_grounded_description(f"count {parsed.target_a}", answer, events),
        )

    if parsed.qtype == "absence":
        if extractive_answer:
            evidence = [_format_event(e) for e in _top_events(events, 8)]
            return QAResult(
                answer=extractive_answer,
                evidence=evidence,
                description=generate_grounded_description(question or "absence", extractive_answer, events),
            )
        if not parsed.target_a:
            answer = "It is not working: the target sound for presence check is missing."
            evidence = [_format_event(e) for e in _top_events(events, 5)]
            return QAResult(
                answer=answer,
                evidence=evidence,
                description=generate_grounded_description("absence", answer, events),
            )
        target = parsed.target_a.lower()
        present = len(by_label.get(target, [])) > 0
        answer = (
            f"Yes, {parsed.target_a} is present in the audio."
            if present
            else f"No, {parsed.target_a} was not detected in the audio."
        )
        evidence = [_format_event(e) for e in _top_events(by_label.get(target, []), 8)] if present else []
        return QAResult(
            answer=answer,
            evidence=evidence,
            description=generate_grounded_description(f"absence {parsed.target_a}", answer, events),
        )

    if parsed.qtype == "sequence":
        if extractive_answer:
            evidence = [_format_event(e) for e in _top_events(events, 8)]
            return QAResult(
                answer=extractive_answer,
                evidence=evidence,
                description=generate_grounded_description(question or "sequence", extractive_answer, events),
            )
        if not parsed.target_a or not parsed.target_b:
            answer = "It is not working: two sounds are required for sequence comparison."
            evidence = [_format_event(e) for e in _top_events(events, 8)]
            return QAResult(
                answer=answer,
                evidence=evidence,
                description=generate_grounded_description("sequence", answer, events),
            )

        a_events = by_label.get(parsed.target_a.lower(), [])
        b_events = by_label.get(parsed.target_b.lower(), [])

        if not a_events or not b_events:
            answer = "It is not working: both requested sounds were not detected reliably."
            evidence = [_format_event(e) for e in _top_events(events, 8)]
            return QAResult(
                answer=answer,
                evidence=evidence,
                description=generate_grounded_description("sequence", answer, events),
            )

        a_first = min(a_events, key=lambda e: e.start)
        b_first = min(b_events, key=lambda e: e.start)

        if a_first.start < b_first.start:
            answer = f"{parsed.target_a} happened before {parsed.target_b}."
        else:
            answer = f"{parsed.target_b} happened before {parsed.target_a}."

        evidence = [_format_event(a_first), _format_event(b_first)]
        return QAResult(
            answer=answer,
            evidence=evidence,
            description=generate_grounded_description(
                f"sequence {parsed.target_a} and {parsed.target_b}",
                answer,
                events,
            ),
        )

    answer = "It is not working: could not parse the question type."
    return QAResult(
        answer=answer,
        evidence=[],
        description=generate_grounded_description("unknown", answer, events),
    )
