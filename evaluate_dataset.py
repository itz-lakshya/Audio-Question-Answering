from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from mlforge_audioqa.config import PipelineConfig
from mlforge_audioqa.data.local_dataset import collect_examples, stratified_split
from mlforge_audioqa.pipeline import AudioQAPipeline


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def write_csv_report(path: str, rows: list[dict[str, str]]) -> None:
    if not rows:
        return

    fieldnames = [
        "file",
        "truth",
        "pred",
        "correct",
        "top1_score",
        "top2_label",
        "top2_score",
        "top3_label",
        "top3_score",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AudioQA pipeline on uploaded dataset")
    parser.add_argument("--dataset-dir", default="Dataset", help="Directory containing wav files")
    parser.add_argument("--test-ratio", type=float, default=0.25, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-test", type=int, default=0, help="Limit number of test files (0 = all)")
    parser.add_argument(
        "--report-csv",
        default="evaluation_report.csv",
        help="Path to CSV report with per-file correctness",
    )
    parser.add_argument(
        "--show-errors",
        type=int,
        default=10,
        help="How many incorrect samples to print",
    )
    args = parser.parse_args()

    examples = collect_examples(args.dataset_dir)
    if not examples:
        raise SystemExit(f"No valid wav files found in: {args.dataset_dir}")

    labels = sorted({e.label for e in examples})
    config = PipelineConfig(labels=labels)
    pipeline = AudioQAPipeline(config)

    train_set, test_set = stratified_split(examples, test_ratio=args.test_ratio, seed=args.seed)

    if args.max_test > 0:
        rng = random.Random(args.seed)
        rng.shuffle(test_set)
        test_set = test_set[: args.max_test]

    total = len(test_set)
    cls_hits = 0
    qa_presence_hits = 0
    qa_absence_hits = 0
    qa_count_hits = 0

    per_class_total: Counter[str] = Counter()
    per_class_hit: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    wrong_rows: list[dict[str, str]] = []
    report_rows: list[dict[str, str]] = []

    for row in test_set:
        per_class_total[row.label] += 1

        pred = pipeline.predict_clip_label(row.path)
        if pred == row.label:
            cls_hits += 1
            per_class_hit[row.label] += 1
        else:
            confusion[(row.label, pred or "<none>")] += 1

        score_map = pipeline.clip_label_scores(row.path)
        top = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        top1_score = f"{top[0][1]:.4f}" if len(top) > 0 else ""
        top2_label = top[1][0] if len(top) > 1 else ""
        top2_score = f"{top[1][1]:.4f}" if len(top) > 1 else ""
        top3_label = top[2][0] if len(top) > 2 else ""
        top3_score = f"{top[2][1]:.4f}" if len(top) > 2 else ""

        report_row = {
            "file": Path(row.path).name,
            "truth": row.label,
            "pred": pred or "<none>",
            "correct": "1" if pred == row.label else "0",
            "top1_score": top1_score,
            "top2_label": top2_label,
            "top2_score": top2_score,
            "top3_label": top3_label,
            "top3_score": top3_score,
        }
        report_rows.append(report_row)
        if pred != row.label:
            wrong_rows.append(report_row)

        # Presence question.
        q_presence = f"Is there any {row.label}?"
        a_presence = pipeline.answer(row.path, q_presence).answer.lower()
        if a_presence.startswith("yes"):
            qa_presence_hits += 1

        # Absence question with a random negative class.
        negatives = [x for x in labels if x != row.label]
        neg = random.choice(negatives) if negatives else row.label
        q_absence = f"Is there any {neg}?"
        a_absence = pipeline.answer(row.path, q_absence).answer.lower()
        if a_absence.startswith("no"):
            qa_absence_hits += 1

        # Count should be positive for the ground-truth class.
        q_count = f"How many times did {row.label} occur?"
        a_count = pipeline.answer(row.path, q_count).answer.lower()
        if "occurred" in a_count and not a_count.endswith("0 time(s)."):
            qa_count_hits += 1

    print("=== Dataset Evaluation ===")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Total files: {len(examples)}")
    print(f"Train size: {len(train_set)}")
    print(f"Test size: {len(test_set)}")
    print(f"Active labels: {', '.join(labels)}")
    print()
    print("--- Metrics ---")
    print(f"Clip classification accuracy: {safe_div(cls_hits, total):.3f} ({cls_hits}/{total})")
    print(f"QA presence accuracy:        {safe_div(qa_presence_hits, total):.3f} ({qa_presence_hits}/{total})")
    print(f"QA absence accuracy:         {safe_div(qa_absence_hits, total):.3f} ({qa_absence_hits}/{total})")
    print(f"QA count-positive accuracy:  {safe_div(qa_count_hits, total):.3f} ({qa_count_hits}/{total})")
    print()

    print("--- Per-class Accuracy ---")
    for label in sorted(per_class_total.keys()):
        hits = per_class_hit[label]
        count = per_class_total[label]
        print(f"{label:20s}: {safe_div(hits, count):.3f} ({hits}/{count})")

    print()
    print("--- Top Confusions ---")
    if confusion:
        for (truth, pred), c in confusion.most_common(10):
            print(f"{truth} -> {pred}: {c}")
    else:
        print("No confusions found.")

    print()
    print(f"--- Incorrect Samples (up to {args.show_errors}) ---")
    if wrong_rows:
        for row in wrong_rows[: args.show_errors]:
            print(
                f"{row['file']}: truth={row['truth']} pred={row['pred']} "
                f"top1_score={row['top1_score']}"
            )
    else:
        print("No incorrect samples in this run.")

    write_csv_report(args.report_csv, report_rows)
    print()
    print(f"Saved per-file report: {args.report_csv}")


if __name__ == "__main__":
    main()
