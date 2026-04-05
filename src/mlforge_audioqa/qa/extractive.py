from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import torch


@dataclass
class ExtractiveAnswer:
    answer: str
    score: float


class ExtractiveQAModel:
    def __init__(self, model_name: str = "deepset/roberta-base-squad2") -> None:
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._load_lock = Lock()

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        with self._load_lock:
            if self._model is None or self._tokenizer is None:
                from transformers import AutoModelForQuestionAnswering, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
                self._model.eval()

        return self._model, self._tokenizer

    def answer(self, question: str, context: str) -> ExtractiveAnswer:
        model, tokenizer = self._load()

        inputs = tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=386,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        start_probs = torch.softmax(start_logits, dim=-1)
        end_probs = torch.softmax(end_logits, dim=-1)

        start_idx = int(torch.argmax(start_logits).item())
        end_idx = int(torch.argmax(end_logits).item())

        # RoBERTa SQuAD2 uses CLS/no-answer behavior; also guard invalid spans.
        if end_idx < start_idx or (start_idx == 0 and end_idx == 0):
            return ExtractiveAnswer(answer="", score=0.0)

        input_ids = inputs["input_ids"][0]
        answer_ids = input_ids[start_idx : end_idx + 1]
        text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        score = float((start_probs[start_idx] * end_probs[end_idx]).item())

        if not text:
            return ExtractiveAnswer(answer="", score=0.0)

        return ExtractiveAnswer(answer=text, score=score)
