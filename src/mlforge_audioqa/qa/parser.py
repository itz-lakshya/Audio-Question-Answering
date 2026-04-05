from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParsedQuestion:
    qtype: str
    target_a: str | None = None
    target_b: str | None = None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def parse_question(question: str, known_labels: list[str]) -> ParsedQuestion:
    q = _normalize(question)

    matched = [label for label in known_labels if label.lower() in q]
    target_a = matched[0] if matched else None
    target_b = matched[1] if len(matched) > 1 else None

    if any(k in q for k in ["how many", "count", "number of"]):
        return ParsedQuestion(qtype="count", target_a=target_a)

    if any(k in q for k in ["before", "after", "first", "then"]):
        return ParsedQuestion(qtype="sequence", target_a=target_a, target_b=target_b)

    if (
        any(k in q for k in ["is there", "any", "present", "exist", "did "])
        or re.search(r"did\s+.+\s+happen", q) is not None
    ):
        return ParsedQuestion(qtype="absence", target_a=target_a)

    return ParsedQuestion(qtype="what", target_a=target_a)
