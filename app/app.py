"""Gradio chat UI for ER-Assistant: history, feedback, export."""

import json
import sys
import io
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import fitz  # PyMuPDF
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from rag_pipeline import client, generate_answer, load_index, retrieve

FEEDBACK_LOG = BASE_DIR / "data" / "feedback.jsonl"
EXPORT_DIR = BASE_DIR / "data" / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR = BASE_DIR / "data" / "pdfs"

print("Loading FAISS index and chunks...")
index, chunks = load_index()
print(f"Ready. {index.ntotal} vectors loaded.")


# ---------- PDF Highlighting Logic ----------

def get_verbatim_quotes(page_text, llm_answer):
    """Uses LLM to find exact quotes from the page used to generate the answer."""
    prompt = (
        "You are an exact text matcher. Given the 'Page Text' and the 'LLM Answer', "
        "extract the exact, verbatim phrases or sentences from the 'Page Text' that support the 'LLM Answer'. "
        "Return ONLY the exact quotes from the page text, separated by a newline. "
        "Do not alter the text, do not fix typos, and do not add conversational filler.\n\n"
        f"Page Text:\n{page_text}\n\n"
        f"LLM Answer:\n{llm_answer}\n\n"
        "Verbatim Quotes:"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    quotes = response.choices[0].message.content.strip().split('\n')
    return [q.strip() for q in quotes if q.strip()]


def process_and_highlight_pages(answer, retrieved):
    """Opens PDF dynamically, checks if chunk is cited in answer, highlights quotes, returns images for gallery."""
    highlighted_images = []
    seen_pages = set()
                                                                                
    for c in retrieved:
        # Simple check: only process this page if the LLM's answer actually mentions its source file AND page number
        if c["source"] not in answer or str(c["page"]) not in answer:
            continue
        
        page_num = c.get("page")
    
        if page_num is None or page_num in seen_pages:
            continue
        seen_pages.add(page_num)

        # Build path using PDF_DIR and the source filename from the chunk
        pdf_path = PDF_DIR / c["source"]
        if not pdf_path.exists():
            print(f"[Warning] PDF not found at {pdf_path}")
            continue
                                                   
        doc = fitz.open(pdf_path)
        
        # Assuming 'page' in chunk is 1-based index; PyMuPDF is 0-based
        try:
            actual_page_index = int(page_num) - 1
        except ValueError:
            doc.close()
            continue
             
        if actual_page_index < 0 or actual_page_index >= len(doc):
            doc.close()
            continue

        # 1. Get the page and its text
        page = doc[actual_page_index]
        page_text = page.get_text()
        
        # 2. Ask LLM to extract verbatim quotes used for the answer
        quotes = get_verbatim_quotes(page_text, answer)
        
        # 3. Highlight the quotes on the PDF page
        for quote in quotes:
            if not quote: 
                continue
            text_instances = page.search_for(quote)
            for inst in text_instances:
                page.add_highlight_annot(inst)

        # 4. Render to an image
        pix = page.get_pixmap(dpi=150)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        # Format for gr.Gallery: tuple of (Image, Caption)
        caption = f"{c['source']} - Page {page_num}"
        highlighted_images.append((img, caption))
        
        doc.close()

    return highlighted_images


# ---------- General Helpers ----------

def to_text(content):
    """Normalize Gradio message content to a plain string."""
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

    display_answer = answer

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

    # Generate highlighted PDF screenshots ONLY for cited pages
    highlighted_pages = process_and_highlight_pages(answer, retrieved) if not is_refusal else []
    
    fb_update = gr.update(interactive=not is_refusal)
    
    return (
        new_history,
        new_last,
        "",  # clear msg box
        "",  # clear feedback status
        fb_update,
        fb_update,
        gr.update(interactive=True),  # export conversation
        gr.update(interactive=True),  # export last answer
        gr.update(interactive=False), # send_btn
        highlighted_pages             # pdf_gallery gets the list of (image, caption)
    )


# ---------- UI ----------
with gr.Blocks(title="🚑 ER-Assistant") as demo:
    gr.Markdown("# 🚑 ER-Assistant — First Aid & CPR Q&A")
    gr.Markdown(
        "Ask questions about the Canadian Red Cross First Aid & CPR Guide. "
        "Follow-ups are supported. Use 👍 / 👎 to rate answers."
    )

    last_exchange_state = gr.State(None)

    with gr.Row():
        # LEFT COLUMN: Chat UI
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(height=600)
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
        
        # RIGHT COLUMN: PDF Viewer (gr.Gallery has built-in next/prev functionality!)
        with gr.Column(scale=1):
            pdf_gallery = gr.Gallery(
                label="Cited PDF Pages",
                show_label=True,
                height=700,
                object_fit="contain",
                preview=True
            )

    # 10 Outputs total to map directly to the `chat` return values
    chat_outputs = [
        chatbot,
        last_exchange_state,
        msg,
        feedback_status,
        thumbs_up,
        thumbs_down,
        export_chat_btn,
        export_answer_btn,
        send_btn,
        pdf_gallery
    ]

    msg.submit(
        chat,
        inputs=[msg, chatbot, last_exchange_state],
        outputs=chat_outputs,
    )

    send_btn.click(
        chat,
        inputs=[msg, chatbot, last_exchange_state],
        outputs=chat_outputs,        
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
            [],                           # clear chatbot
            None,                         # clear last_exchange_state
            "",                           # clear msg
            "",                           # clear feedback status
            gr.update(interactive=False), # thumbs_up
            gr.update(interactive=False), # thumbs_down
            gr.update(interactive=False), # export_chat_btn
            gr.update(interactive=False), # export_answer_btn
            gr.update(interactive=False), # send_btn
            None                          # clear pdf_gallery
        ),
        outputs=chat_outputs,
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