"""Gradio chat UI for ER-Assistant with conversation history and feedback logging."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from rag_pipeline import client, generate_answer, load_index, retrieve

FEEDBACK_LOG = BASE_DIR / "data" / "feedback.jsonl"

print("Loading FAISS index and chunks...")
index, chunks = load_index()
print(f"Ready. {index.ntotal} vectors loaded.")


# ---------- query rewriting for follow-ups ----------
def rewrite_query(message, history):
    if not history:
        return message

    transcript = ""
    for turn in history[-3:]:
        if isinstance(turn, dict):
            transcript += f"{turn['role']}: {turn['content']}\n"
        else:
            user_msg, assistant_msg = turn
            transcript += f"user: {user_msg}\nassistant: {assistant_msg}\n"

    prompt = (
        "Given the conversation below and a follow-up question, rewrite the "
        "follow-up as a standalone question. If it's already standalone, "
        "return it unchanged. Output ONLY the rewritten question.\n\n"
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
    seen = set()
    lines = ["**Sources:**"]
    for c in retrieved:
        key = (c["source"], c["page"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- `{c['source']}` p.{c['page']}")
    return "\n".join(lines)


# ---------- feedback logging ----------
def log_feedback(rating, last_exchange):
    """Append a feedback record and disable both buttons."""
    if not last_exchange:
        return "No answer to rate yet.", gr.update(), gr.update()

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rating": rating,
        "question": last_exchange["question"],
        "rewritten_query": last_exchange["rewritten_query"],
        "answer": last_exchange["answer"],
        "sources": [
            {"source": c["source"], "page": c["page"], "score": c["score"]}
            for c in last_exchange["retrieved"]
        ],
    }
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    status = f"Feedback recorded ({rating}). Thank you!"
    # Disable both buttons after voting
    return status, gr.update(interactive=False), gr.update(interactive=False)


# ---------- chat handler ----------
def chat(message, history, last_exchange):
    standalone_query = rewrite_query(message, history)
    retrieved = retrieve(standalone_query, index, chunks)
    answer = generate_answer(standalone_query, retrieved)

    refusal_phrases = ["don't have", "do not have", "cannot answer", "no information"]
    is_refusal = any(p in answer.lower() for p in refusal_phrases)

    display_answer = (
        answer if is_refusal else f"{answer}\n\n---\n\n{format_sources(retrieved)}"
    )

    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": display_answer},
    ]
    new_last = {
        "question": message,
        "rewritten_query": standalone_query,
        "answer": answer,
        "retrieved": retrieved,
    }

    # Re-enable feedback buttons for the new answer (unless it's a refusal)
    btn_update = gr.update(interactive=not is_refusal)

    return new_history, new_last, "", "", btn_update, btn_update


with gr.Blocks(title="🚑 ER-Assistant") as demo:
    gr.Markdown("# 🚑 ER-Assistant — First Aid & CPR Q&A")
    gr.Markdown(
        "Ask questions about the Canadian Red Cross First Aid & CPR Guide. "
        "Follow-ups are supported. Use 👍 / 👎 to rate answers."
    )

    # State: holds the most recent exchange so feedback can reference it
    last_exchange_state = gr.State(None)

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="Ask a question...", label="Your question")

    with gr.Row():
        thumbs_up = gr.Button("👍 Helpful", variant="secondary", interactive=False)
        thumbs_down = gr.Button(
            "👎 Not helpful", variant="secondary", interactive=False
        )
        clear_btn = gr.Button("Clear chat")

    feedback_status = gr.Markdown("")

    msg.submit(
        chat,
        inputs=[msg, chatbot, last_exchange_state],
        outputs=[
            chatbot,
            last_exchange_state,
            msg,
            feedback_status,
            thumbs_up,
            thumbs_down,
        ],
    )

    thumbs_up.click(
        lambda exch: log_feedback("helpful", exch),
        inputs=[last_exchange_state],
        outputs=[feedback_status, thumbs_up, thumbs_down],
    )
    thumbs_down.click(
        lambda exch: log_feedback("not_helpful", exch),
        inputs=[last_exchange_state],
        outputs=[feedback_status, thumbs_up, thumbs_down],
    )
    clear_btn.click(
        lambda: (
            [],
            None,
            "",
            "",
            gr.update(interactive=False),
            gr.update(interactive=False),
        ),
        outputs=[
            chatbot,
            last_exchange_state,
            msg,
            feedback_status,
            thumbs_up,
            thumbs_down,
        ],
    )


if __name__ == "__main__":
    demo.launch()
