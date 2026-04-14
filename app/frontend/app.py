import gradio as gr

from rag import generate_answer, load_index, retrieve

# Load once at startup
print("Loading FAISS index and chunks...")
index, chunks = load_index()
print(f"Ready. {index.ntotal} vectors loaded.")


def format_sources(retrieved):
    lines = ["**Sources used:**"]
    for i, c in enumerate(retrieved, 1):
        preview = c["text"][:150].replace("\n", " ")
        lines.append(
            f"{i}. `{c['source']}` p.{c['page']} (score: {c['score']:.3f})  \n"
            f"   > {preview}..."
        )
    return "\n".join(lines)


def chat(message, history):
    retrieved = retrieve(message, index, chunks, k=5)
    answer = generate_answer(message, retrieved)

    if "don't have information" in answer.lower():
        return answer

    return f"{answer}\n\n---\n\n{format_sources(retrieved)}"


demo = gr.ChatInterface(
    fn=chat,
    title="📄 RAG Q&A — First Aid & CPR Guide",
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
