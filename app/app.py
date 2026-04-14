"""Gradio chat UI for ER-Assistant: history, feedback, export."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from rag_pipeline import client, generate_answer, load_index, retrieve

FEEDBACK_LOG = BASE_DIR / "data" / "feedback.jsonl"
EXPORT_DIR = BASE_DIR / "data" / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading FAISS index and chunks...")
index, chunks = load_index()
print(f"Ready. {index.ntotal} vectors loaded.")


def to_text(content):
    """Normalize Gradio message content to a plain string.

    Gradio 6 may pass content as a list of blocks like
    [{"text": "...", "type": "text"}] or as a plain string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content)


# ---------- query rewriting for follow-ups ----------
def rewrite_query(message, history):
    if not history:
        return message

    transcript = ""
    for turn in history[-3:]:
        if isinstance(turn, dict):
            content = to_text(turn["content"])
            transcript += f"{turn['role']}: {content}\n"
        else:
            user_msg, assistant_msg = turn
            transcript += (
                f"user: {to_text(user_msg)}\nassistant: {to_text(assistant_msg)}\n"
            )

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

    return (
        f"Feedback recorded ({rating}). Thank you!",
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


# ---------- export ----------
def export_conversation(history):
    if not history:
        return None, "Nothing to export."

    lines = [
        "# ER-Assistant Conversation",
        f"*Exported: {datetime.now(timezone.utc).isoformat()}*",
        "",
    ]
    for turn in history:
        role = turn["role"].capitalize()
        content = to_text(turn["content"])
        lines.append(f"## {role}")
        lines.append(content)
        lines.append("")

    md = "\n".join(lines)
    filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    filepath = EXPORT_DIR / filename
    filepath.write_text(md, encoding="utf-8")
    return str(filepath), f"Exported {len(history)} messages."


def export_last_answer(last_exchange):
    if not last_exchange:
        return None, "No answer to export."

    md = (
        f"# ER-Assistant Answer\n"
        f"*Exported: {datetime.now(timezone.utc).isoformat()}*\n\n"
        f"## Question\n{last_exchange['question']}\n\n"
        f"## Answer\n{last_exchange['answer']}\n\n"
        f"## Sources\n"
        + "\n".join(
            f"- `{c['source']}` p.{c['page']} (score: {c['score']:.3f})"
            for c in last_exchange["retrieved"]
        )
    )
    filename = f"answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    filepath = EXPORT_DIR / filename
    filepath.write_text(md, encoding="utf-8")
    return str(filepath), "Exported last answer."


# ---------- chat handler ----------
def chat(message, history, last_exchange):
    message = to_text(message)
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

    # Re-enable feedback buttons (unless refusal) and enable both export buttons
    fb_update = gr.update(interactive=not is_refusal)
    return (
        new_history,
        new_last,
        "",
        "",
        fb_update,
        fb_update,
        gr.update(interactive=True),  # export conversation
        gr.update(interactive=True),  # export last answer
        gr.update(interactive=False),  # send_btn ← new
    )


# ---------- UI ----------
with gr.Blocks(title="🚑 ER-Assistant") as demo:
    gr.Markdown("# 🚑 ER-Assistant — First Aid & CPR Q&A")
    gr.Markdown(
        "Ask questions about the Canadian Red Cross First Aid & CPR Guide. "
        "Follow-ups are supported. Use 👍 / 👎 to rate answers."
    )

    last_exchange_state = gr.State(None)

    chatbot = gr.Chatbot(height=500)
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask a question...",
            show_label=False,
            scale=9,
            container=False,
        )
        send_btn = gr.Button(
            "Send", variant="primary", scale=1, min_width=80, interactive=False
        )

    with gr.Row():
        thumbs_up = gr.Button("👍 Helpful", variant="secondary", interactive=False)
        thumbs_down = gr.Button(
            "👎 Not helpful", variant="secondary", interactive=False
        )
        clear_btn = gr.Button("Clear chat")

    feedback_status = gr.Markdown("")

    with gr.Row():
        export_chat_btn = gr.Button(
            "📥 Export chat", size="sm", scale=0, interactive=False
        )
        export_answer_btn = gr.Button(
            "📥 Export answer", size="sm", scale=0, interactive=False
        )
    export_file = gr.File(label="Download")
    export_status = gr.Markdown("")

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
            export_chat_btn,
            export_answer_btn,
            send_btn,
        ],
    )

    send_btn.click(
        chat,
        inputs=[msg, chatbot, last_exchange_state],
        outputs=[
            chatbot,
            last_exchange_state,
            msg,
            feedback_status,
            thumbs_up,
            thumbs_down,
            export_chat_btn,
            export_answer_btn,
            send_btn,
        ],
    )

    msg.change(
        lambda text: gr.update(interactive=bool(text.strip())),
        inputs=[msg],
        outputs=[send_btn],
    )

    msg.change(
        lambda text: gr.update(interactive=bool(text.strip())),
        inputs=[msg],
        outputs=[send_btn],
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
            export_chat_btn,
            export_answer_btn,
        ],
    )

    export_chat_btn.click(
        export_conversation,
        inputs=[chatbot],
        outputs=[export_file, export_status],
    )
    export_answer_btn.click(
        export_last_answer,
        inputs=[last_exchange_state],
        outputs=[export_file, export_status],
    )


if __name__ == "__main__":
    demo.launch()
