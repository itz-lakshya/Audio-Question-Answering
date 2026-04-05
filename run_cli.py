from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from mlforge_audioqa.config import DEFAULT_CONFIG
from mlforge_audioqa.pipeline import AudioQAPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Audio QA CLI")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument(
        "--question",
        required=False,
        default="Describe what is happening in this audio.",
        help="Question about the audio (defaults to description mode)",
    )
    args = parser.parse_args()

    try:
        pipeline = AudioQAPipeline(DEFAULT_CONFIG)
        result = pipeline.answer(args.audio, args.question)
    except Exception:
        print("Answer:")
        print("It is not working.")
        print("\nEvidence:")
        print("- None")
        print("\nGrounded Description:")
        print("The system is not working for this run.")
        return

    print("Answer:")
    print(result.answer)
    print("\nEvidence:")
    if result.evidence:
        for row in result.evidence:
            print("-", row)
    else:
        print("- None")

    print("\nGrounded Description:")
    print(result.description)


if __name__ == "__main__":
    main()
