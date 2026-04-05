from __future__ import annotations

import os
import sys

import gradio as gr

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from mlforge_audioqa.config import DEFAULT_CONFIG
from mlforge_audioqa.pipeline import AudioQAPipeline


pipeline = AudioQAPipeline(DEFAULT_CONFIG)


def run_qa(audio_file, question: str):
    if audio_file is None:
        return "It is not working: no audio uploaded.", "", ""

    if not question or not question.strip():
        question = "Describe what is happening in this audio."

    try:
        result = pipeline.answer(audio_file, question)
    except Exception:
        return "It is not working.", "", ""

    evidence = "\n".join(result.evidence) if result.evidence else "No evidence events."
    return result.answer, evidence, result.description


with gr.Blocks(title="ML Forge Audio QA") as demo:
    gr.Markdown("# Audio Question Answering (Audio-Grounded)")
    gr.Markdown("Upload audio and ask a question about events, counts, order, or absence.")

    audio = gr.Audio(type="filepath", label="Audio File")
    question = gr.Textbox(
        label="Question",
        placeholder="Describe what is happening in this audio.",
    )

    btn = gr.Button("Answer")
    answer = gr.Textbox(label="Answer")
    evidence = gr.Textbox(label="Evidence", lines=8)
    description = gr.Textbox(label="Grounded Description", lines=8)

    btn.click(fn=run_qa, inputs=[audio, question], outputs=[answer, evidence, description])


if __name__ == "__main__":
    demo.launch()
