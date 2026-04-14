"""Gradio chat UI for the ER-Assistant RAG system, with conversation history."""

import sys
from pathlib import Path

import gradio as gr

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from rag_pipeline import retrieve, generate_answer, load_index, client

print("Loading FAISS index and chunks...")
index, chunks = load_index()
print(f"Ready. {index.ntotal} vectors loaded.")


def rewrite_query(message, history):
    """Turn a follow-up question into a standalone query using prior turns."""
    if not history:
        return message

    # Build a short transcript of the last few turns
    transcript = ""
    for turn in history[-3:]:  # last 3 turns is plenty of context
        if isinstance(turn, dict):
            # Gradio "messages" format
            transcript += f"{turn['role']}: {turn['content']}\n"
        else:
            # Gradio tuple format: (user, assistant)
            user_msg, assistant_msg = turn
            transcript += f"user: {user_msg}\nassistant: {assistant_msg}\n"

    prompt = (
        "Given the conversation below and a follow-up question, rewrite the "
        "follow-up as a standalone question that can be understood without "
        "the conversation history. If the follow-up is already standalone, "
        "return it unchanged. Output ONLY the rewritten question, nothing else.\n\n"
        f"Conversation:\n{transcript}\n"
        f"Follow-up: {message}\n\n"
        "Standalone question:"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
    )
    rewritten = response.choices[0].message.content.strip()
    print(f"[rewrite] '{message}' -> '{rewritten}'")
    return rewritten


def format_sources(retrieved):
    """Show just source filenames and pages, deduplicated."""
    seen = set()
    lines = ["**Sources:**"]
    for c in retrieved:
        key = (c["source"], c["page"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- `{c['source']}` p.{c['page']}")
    return "\n".join(lines)


def chat(message, history):
    standalone_query = rewrite_query(message, history)
    retrieved = retrieve(standalone_query, index, chunks)
    answer = generate_answer(standalone_query, retrieved)

    # Hide sources on any kind of refusal
    refusal_phrases = ["don't have", "do not have", "cannot answer", "no information"]
    if any(phrase in answer.lower() for phrase in refusal_phrases):
        return answer

    return f"{answer}\n\n---\n\n{format_sources(retrieved)}"


demo = gr.ChatInterface(
    fn=chat,
    title="🚑 ER-Assistant — First Aid & CPR Q&A",
    description=(
        "Ask questions about the Canadian Red Cross Comprehensive Guide "
        "for First Aid & CPR. Follow-up questions are supported."
    ),
    examples=[
        "How deep should adult CPR chest compressions be?",
        "How should I treat a choking pregnant person?",
        "What does the FAST acronym mean for stroke assessment?",
    ],
)

if __name__ == "__main__":
    demo.launch()
