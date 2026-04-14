"""Gradio chat UI for the ER-Assistant RAG system."""

import sys
from pathlib import Path

import gradio as gr

# Make the scripts/ folder importable so we can use the shared pipeline
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from pipeline import generate_answer, load_index, retrieve

# Load index and chunks once at startup
print("Loading FAISS index and chunks...")
index, chunks = load_index()
print(f"Ready. {index.ntotal} vectors loaded.")


def format_sources(retrieved):
    """Format retrieved chunks as a readable sources block."""
    lines = ["**Sources used:**"]
    for i, c in enumerate(retrieved, 1):
        preview = c["text"][:150].replace("\n", " ")
        lines.append(
            f"{i}. `{c['source']}` p.{c['page']} (score: {c['score']:.3f})  \n"
            f"   > {preview}..."
        )
    return "\n".join(lines)


def chat(message, history):
    """Handle one chat turn: retrieve, generate, and append sources."""
    retrieved = retrieve(message, index, chunks, k=5)
    answer = generate_answer(message, retrieved)

    # Don't show sources on refusals
    if "don't have information" in answer.lower():
        return answer

    return f"{answer}\n\n---\n\n{format_sources(retrieved)}"


demo = gr.ChatInterface(
    fn=chat,
    title="🚑 ER-Assistant — First Aid & CPR Q&A",
    description=(
        "Ask questions about the Canadian Red Cross Comprehensive Guide "
        "for First Aid & CPR. Answers include source citations."
    ),
    examples=[
        "What bleach-to-water ratio is recommended for disinfecting a blood spill?",
        "How deep should adult CPR chest compressions be?",
        "What does the FAST acronym mean for stroke assessment?",
        "What's the weather in Paris today?",
    ],
)

if __name__ == "__main__":
    demo.launch()
