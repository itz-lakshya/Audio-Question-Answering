from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random


@dataclass(frozen=True)
class AudioExample:
    path: str
    class_id: str
    label: str


def parse_label_id_from_filename(file_name: str) -> str | None:
    # Supports filenames like 50223-3-0-1.wav, using the second token as class id.
    parts = Path(file_name).stem.split("-")
    if len(parts) < 2:
        return None
    return parts[1]


def collect_examples(dataset_dir: str) -> list[AudioExample]:
    root = Path(dataset_dir)
    examples: list[AudioExample] = []

    for wav_path in sorted(root.glob("*.wav")):
        class_id = parse_label_id_from_filename(wav_path.name)
        if class_id is None:
            continue

        # Keep labels generic and dataset-agnostic.
        label = f"class_{class_id}"
        examples.append(AudioExample(path=str(wav_path), class_id=class_id, label=label))

    return examples


def stratified_split(examples: list[AudioExample], test_ratio: float = 0.25, seed: int = 42):
    by_class: dict[str, list[AudioExample]] = {}
    for row in examples:
        by_class.setdefault(row.class_id, []).append(row)

    rng = random.Random(seed)
    train: list[AudioExample] = []
    test: list[AudioExample] = []

    for _, rows in sorted(by_class.items(), key=lambda t: t[0]):
        rows_copy = rows[:]
        rng.shuffle(rows_copy)
        n_test = max(1, int(len(rows_copy) * test_ratio)) if len(rows_copy) > 1 else 1
        test.extend(rows_copy[:n_test])
        train.extend(rows_copy[n_test:])

    return train, test
