"""Gradio chat UI for ER-Assistant: history, feedback, export."""

import json
import sys
import io
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import fitz  # PyMuPDF
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from agents import (
    run_agents,
    is_refusal,
    rerun_with_more_research,
    rerun_with_rewrite,
)
from rag_pipeline import client, generate_answer, load_index, retrieve

USAGE_LOG = BASE_DIR / "data" / "usage_log.jsonl"
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
    quotes = response.choices[0].message.content.strip().split("\n")
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


def format_trace(pending: dict) -> str:
    a1 = pending.get("a1", {})
    a2 = pending.get("a2", {})
    a3 = pending.get("a3", {})

    lines = []
    lines.append("### 🔎 Agent 1 — Internal Researcher")
    lines.append(f"- Domain check: **{a1.get('domain_check', '?')}**")
    lines.append(f"- Status: **{a1.get('status', '?')}**")
    n_sources = len(a1.get("sources", []))
    lines.append(f"- Internal sources retrieved: {n_sources}")
    if a1.get("answer"):
        preview = a1["answer"][:300]
        lines.append(f"- Draft: > {preview}{'…' if len(a1['answer']) > 300 else ''}")
    lines.append("")

    lines.append("### 🌐 Agent 2 — External Fact-Checker")
    tc = a2.get("tool_calls", [])
    lines.append(f"- Tools called: {', '.join(tc) if tc else 'none'}")
    lines.append(f"- External sources found: {len(a2.get('sources', []))}")
    if a2.get("web_findings"):
        lines.append(f"- Findings: > {a2['web_findings']}")
    lines.append("")

    lines.append("### ✍️ Agent 3 — Synthesizer + Router")
    lines.append(f"- Router action: **{a3.get('action', '?')}**")
    lines.append(f"- Reason: {a3.get('reason', '')}")
    if a3.get("retry_hint"):
        lines.append(f"- Retry hint: _{a3['retry_hint']}_")
    lines.append("")
    lines.append(
        "_Draft shown in review panel above — approve, request more research, or request rewrite._"
    )

    return "\n".join(lines)


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


# ---------- analytics data ----------
def get_analytics_data():
    """Reads the feedback log and returns stats for the dashboard."""
    if not FEEDBACK_LOG.exists():
        return "No data yet.", None, None

    data = []
    with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)

    # 1. Basic Stats
    total_q = len(df)
    avg_score = 0
    if "sources" in df.columns:
        # Flatten scores from the sources list
        all_scores = [s["score"] for sublist in df["sources"] for s in sublist]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

    # 2. Ratings Distribution
    rating_counts = df["rating"].value_counts().reset_index()
    rating_counts.columns = ["Rating", "Count"]

    # 3. Simple Topic Mockup (In a real app, you'd use LLM to categorize)
    summary_text = (
        f"**Total Questions:** {total_q} | **Avg Retrieval Score:** {avg_score:.3f}"
    )

    return summary_text, rating_counts, df[["timestamp", "rating"]]


def load_analytics_data():
    if not USAGE_LOG.exists():
        return "No usage data yet.", pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(
        [json.loads(line) for line in open(USAGE_LOG, "r", encoding="utf-8")]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["short_time"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    df_display = df.copy()
    df_display["timestamp"] = df_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Summary Stats
    total_q = len(df)
    avg_quality = df["avg_score"].mean()
    total_cost = df["cost"].sum()
    summary = f"### 📊 Summary\n**Total Questions:** {total_q} | **Avg Score:** {avg_quality:.3f} | **Total Est. Cost:** ${total_cost:.4f}"

    # Topic Data (for BarPlot)
    topics = ["CPR", "Choking", "Bleeding", "Burn", "Stroke", "Heart"]
    df_topics = pd.DataFrame(
        [
            {"Topic": t, "Count": int(df["question"].str.contains(t, case=False).sum())}
            for t in topics
        ]
    )
    df_table = df_display[["timestamp", "question", "avg_score", "cost"]].sort_values(
        "timestamp", ascending=False
    )

    return summary, df_topics, df, df_table


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
    """Produce an answer and send it directly to chat. Keep state for HITL follow-ups."""
    message = to_text(message)
    standalone_query = rewrite_query(message, history)

    result = run_agents(standalone_query, index, chunks)
    draft = result["draft"]
    retrieved = result["retrieved"]

    # Commit the answer to chat immediately
    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": draft},
    ]

    new_last = {
        "question": message,
        "rewritten_query": standalone_query,
        "answer": draft,
        "retrieved": retrieved,
    }

    # Stash everything HITL buttons will need if the user wants to revise
    pending = {
        "question": message,
        "rewritten_query": standalone_query,
        "draft": draft,
        "a1": result["a1"],
        "a2": result["a2"],
        "a3": result["a3"],
        "retrieved": retrieved,
    }

    is_refusal_answer = is_refusal(draft)

    # Usage log
    avg_score = (
        sum(c.get("score", 0.0) for c in retrieved) / len(retrieved) if retrieved else 0
    )
    est_cost = (len(message) + len(draft)) * 0.0000002
    usage_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": message,
        "answer": draft[:100] + "...",
        "avg_score": avg_score,
        "cost": est_cost,
        "iterations": result.get("iterations", 1),
    }
    with open(USAGE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(usage_record) + "\n")

    highlighted = (
        process_and_highlight_pages(draft, retrieved)
        if not is_refusal_answer and retrieved
        else []
    )

    trace_md = format_trace(pending)
    fb = gr.update(interactive=not is_refusal_answer)
    hitl_visible = gr.update(visible=not is_refusal_answer)

    return (
        new_history,  # chatbot (message added)
        new_last,  # last_exchange_state
        "",  # clear msg
        "",  # clear feedback_status
        fb,  # thumbs_up
        fb,  # thumbs_down
        gr.update(interactive=True),  # export_chat_btn
        gr.update(interactive=True),  # export_answer_btn
        gr.update(interactive=False),  # send_btn (re-disabled)
        highlighted,  # pdf_gallery
        trace_md,  # trace_display
        pending,  # pending_state
        hitl_visible,  # hitl_panel (show unless refusal)
        "",  # review_hint (clear)
    )


def on_needs_more_research(pending, hint, history, last_exchange):
    """Rerun Agent 2 + Agent 3 and append revised answer as a new assistant turn."""
    if not pending:
        return history, last_exchange, gr.update(), None, [], "", ""

    print(f"\n[HITL] 'Needs more research' clicked. Hint: {hint!r}")

    # Add a small system-style note so the user sees work is happening
    intro = f"🔁 Revising based on your feedback"
    if hint:
        intro += f" (hint: _{hint}_)"
    intro += "…"

    result = rerun_with_more_research(
        pending["rewritten_query"], pending["a1"], hint=hint
    )
    revised = result["draft"]

    # Update pending with the revised state so subsequent HITL clicks use the new base
    pending["draft"] = revised
    pending["a2"] = result["a2"]
    pending["a3"] = result["a3"]
    pending["retrieved"] = result["retrieved"]

    # Append two new turns: a pseudo-user note + the revised assistant answer
    new_history = history + [
        {"role": "user", "content": intro},
        {"role": "assistant", "content": revised},
    ]

    new_last = {
        "question": pending["question"],
        "rewritten_query": pending["rewritten_query"],
        "answer": revised,
        "retrieved": pending["retrieved"],
    }

    trace_md = format_trace(pending)
    highlighted = (
        process_and_highlight_pages(revised, pending["retrieved"])
        if pending["retrieved"]
        else []
    )

    return (
        new_history,  # chatbot (two new turns appended)
        new_last,  # last_exchange
        gr.update(visible=True),  # hitl_panel stays visible for another round
        pending,  # updated pending
        highlighted,  # pdf_gallery refresh
        trace_md,  # trace_display
        "",  # clear review_hint
    )


def on_needs_rewrite(pending, hint, history, last_exchange):
    """Rerun Agent 3 with rewrite hint and append revised answer as a new assistant turn."""
    if not pending:
        return history, last_exchange, gr.update(), None, [], "", ""

    print(f"\n[HITL] 'Needs rewrite' clicked. Hint: {hint!r}")

    intro = f"✏️ Rewriting based on your feedback"
    if hint:
        intro += f" (hint: _{hint}_)"
    intro += "…"

    result = rerun_with_rewrite(
        pending["rewritten_query"], pending["a1"], pending["a2"], hint=hint
    )
    revised = result["draft"]

    pending["draft"] = revised
    pending["a3"] = result["a3"]

    new_history = history + [
        {"role": "user", "content": intro},
        {"role": "assistant", "content": revised},
    ]

    new_last = {
        "question": pending["question"],
        "rewritten_query": pending["rewritten_query"],
        "answer": revised,
        "retrieved": pending["retrieved"],
    }

    trace_md = format_trace(pending)

    return (
        new_history,
        new_last,
        gr.update(visible=True),
        pending,
        [],  # pdf_gallery not refreshed (sources didn't change)
        trace_md,
        "",
    )


# ---------- UI ----------
with gr.Blocks(title="🚑 ER-Assistant") as demo:
    gr.Markdown("# 🚑 ER-Assistant — First Aid & CPR Q&A")

    with gr.Tabs() as tabs:

        # TAB 1: Chatbot (Your Existing UI)
        with gr.Tab("Chatbot", id=0):
            gr.Markdown(
                "Ask questions about the Canadian Red Cross First Aid & CPR Guide."
            )

            last_exchange_state = gr.State(None)

            with gr.Row():
                # LEFT COLUMN: Chat UI
                with gr.Column(scale=1):
                    chatbot = gr.Chatbot(height=600)

                    # === Human-in-the-loop review panel ===
                    pending_state = gr.State(None)

                    with gr.Group(visible=False) as hitl_panel:
                        review_hint = gr.Textbox(
                            label="Optional guidance for revision",
                            placeholder="e.g. 'focus on 2025 updates' or 'make it shorter'",
                            lines=1,
                        )
                        with gr.Row():
                            more_research_btn = gr.Button(
                                "🔁 Needs more research", size="sm"
                            )
                            rewrite_btn = gr.Button("✏️ Needs rewrite", size="sm")

                    with gr.Accordion("🔬 Agent trace", open=False):
                        trace_display = gr.Markdown("")
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask a question...",
                            show_label=False,
                            scale=9,
                            container=False,
                        )
                        send_btn = gr.Button(
                            "Send",
                            variant="primary",
                            scale=1,
                            min_width=80,
                            interactive=False,
                        )

                    with gr.Row():
                        thumbs_up = gr.Button(
                            "👍 Helpful", variant="secondary", interactive=False
                        )
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

                # RIGHT COLUMN: PDF Viewer
                with gr.Column(scale=1):
                    pdf_gallery = gr.Gallery(
                        label="Cited PDF Pages",
                        show_label=True,
                        height=700,
                        object_fit="contain",
                        preview=True,
                    )

        # TAB 2: Analytics (The New Dashboard)
        with gr.Tab("Analytics", id=1) as analytics_tab:
            gr.Markdown("## 📊 Usage Analytics")
            stats_output = gr.Markdown("Loading...")
            refresh_btn = gr.Button("🔄 Refresh")

            with gr.Row():
                topic_plot = gr.BarPlot(x="Topic", y="Count", title="Common Topics")
                score_plot = gr.LinePlot(
                    x="short_time",
                    y="avg_score",
                    title="Quality Trend",
                    x_title="timestamp",
                )

            gr.Markdown("### 📜 Detailed Question Log")
            question_table = gr.Dataframe(
                headers=["Time", "Question", "Score", "Cost ($)"],
                datatype=["str", "str", "number", "number"],
                wrap=True,
            )
    # ---------- Event Handlers ----------

    # Function to trigger analytics update
    def update_dashboard():
        summary, df_topics, df_scores = load_analytics_data()
        return summary, df_topics, df_scores

    # Auto-update analytics when user clicks the tab
    analytics_tab.select(
        fn=load_analytics_data,
        outputs=[stats_output, topic_plot, score_plot, question_table],
    )

    # Manual refresh button
    refresh_btn.click(
        fn=load_analytics_data, outputs=[stats_output, topic_plot, score_plot]
    )

    # [Existing Chatbot Handlers]
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
        pdf_gallery,
        trace_display,
        pending_state,
        hitl_panel,
        review_hint,
    ]

    msg.submit(chat, inputs=[msg, chatbot, last_exchange_state], outputs=chat_outputs)
    send_btn.click(
        chat, inputs=[msg, chatbot, last_exchange_state], outputs=chat_outputs
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

    more_research_btn.click(
        on_needs_more_research,
        inputs=[pending_state, review_hint, chatbot, last_exchange_state],
        outputs=[
            chatbot,
            last_exchange_state,
            hitl_panel,
            pending_state,
            pdf_gallery,
            trace_display,
            review_hint,
        ],
    )

    rewrite_btn.click(
        on_needs_rewrite,
        inputs=[pending_state, review_hint, chatbot, last_exchange_state],
        outputs=[
            chatbot,
            last_exchange_state,
            hitl_panel,
            pending_state,
            pdf_gallery,
            trace_display,
            review_hint,
        ],
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
            gr.update(interactive=False),
            None,
            "",
            None,
            gr.update(visible=False),
            "",
        ),
        outputs=chat_outputs,
    )

    export_chat_btn.click(
        export_conversation, inputs=[chatbot], outputs=[export_file, export_status]
    )
    export_answer_btn.click(
        export_last_answer,
        inputs=[last_exchange_state],
        outputs=[export_file, export_status],
    )


if __name__ == "__main__":
    demo.launch()
