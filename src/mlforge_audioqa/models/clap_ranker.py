from __future__ import annotations

from typing import Iterable

import laion_clap
import librosa
import numpy as np
import torch

from ..schemas import WindowPrediction
from .base import AudioTagger


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return a @ b.T


class CLAPTagger(AudioTagger):
    """Semantic audio-text matching using official LAION CLAP."""

    def __init__(self, labels: Iterable[str], top_k: int = 5, enable_fusion: bool = False) -> None:
        self.labels = list(labels)
        self.top_k = top_k
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, device=device)
        # Uses official default checkpoint from the laion-clap package.
        self.model.load_ckpt()

        synonym_map: dict[str, list[str]] = {
            "drilling": ["drill sound", "power drill", "electric drilling", "construction drilling"],
            "hammering": ["jackhammer", "hammer hits", "construction hammering", "pounding metal"],
            "car passing": ["car drive by", "vehicle passing", "passing automobile"],
            "street music": ["music on street", "outdoor music", "busker music"],
            "dog bark": ["dog barking", "barking dog"],
        }

        prompts: list[str] = []
        self.prompt_to_label: list[str] = []
        for label in self.labels:
            variants = [label] + synonym_map.get(label, [])
            for variant in variants:
                prompts.append(f"This is a sound of {variant}.")
                self.prompt_to_label.append(label)

        text_emb = self.model.get_text_embedding(prompts, use_tensor=True)
        if not isinstance(text_emb, torch.Tensor):
            text_emb = torch.tensor(text_emb)
        self.text_emb = text_emb.float()

    @staticmethod
    def _to_48k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate == 48000:
            return audio.astype(np.float32)
        return librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=48000)

    def predict_window(self, audio: np.ndarray, sample_rate: int, start: float, end: float) -> list[WindowPrediction]:
        audio_48k = self._to_48k(audio, sample_rate)
        audio_batch = audio_48k.reshape(1, -1)
        audio_tensor = torch.from_numpy(audio_batch).float()

        with torch.no_grad():
            audio_emb = self.model.get_audio_embedding_from_data(x=audio_tensor, use_tensor=True)
            if not isinstance(audio_emb, torch.Tensor):
                audio_emb = torch.tensor(audio_emb)
            audio_emb = audio_emb.float()
            prompt_sims = _cosine_sim(audio_emb, self.text_emb)[0]

        # Aggregate prompt similarities per canonical label using max pooling.
        label_best: dict[str, float] = {label: float("-inf") for label in self.labels}
        for sim, label in zip(prompt_sims.tolist(), self.prompt_to_label):
            if sim > label_best[label]:
                label_best[label] = sim

        logits = torch.tensor([label_best[label] for label in self.labels], dtype=torch.float32)
        probs = torch.softmax(logits, dim=0)
        top_scores, top_idx = torch.topk(probs, k=min(self.top_k, len(self.labels)))
        return [
            WindowPrediction(
                label=self.labels[int(i)],
                score=float(s),
                start=start,
                end=end,
            )
            for s, i in zip(top_scores.tolist(), top_idx.tolist())
        ]
