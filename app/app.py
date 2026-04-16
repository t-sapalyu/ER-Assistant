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
    run_simple_rag,
    is_refusal,
    rerun_with_more_research,
    rerun_with_rewrite,
)
from rag_pipeline import add_to_database, check_content_scope, client, generate_answer, load_index, retrieve

USAGE_LOG = BASE_DIR / "data" / "usage_log.jsonl"
FEEDBACK_LOG = BASE_DIR / "data" / "feedback.jsonl"
EXPORT_DIR = BASE_DIR / "data" / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR = BASE_DIR / "data" / "pdfs"

print("Loading FAISS index and chunks...")
index, chunks = load_index()
print(f"Ready. {index.ntotal} vectors loaded.")


# Place this near the top of app.py, after imports
CUSTOM_CSS = """
.mode-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 8px;
}
.mode-pill.simple { background: #1f2937; color: #a7f3d0; }
.mode-pill.agentic { background: #064e3b; color: #6ee7b7; }

.hitl-panel {
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 12px;
    margin-top: 8px;
}

.agent-trace {
    background: #1f2937 !important;
    border-left: 3px solid #10b981 !important;
    padding: 8px 12px !important;
    font-size: 0.9rem !important;
    border-radius: 12px !important;
}

.small-btn button {
    font-size: 0.85rem !important;
    padding: 4px 12px !important;
}

footer { display: none !important; }
"""


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
        if c["source"] not in answer or str(c["page"]) not in answer:
            continue

        page_num = c.get("page")

        if page_num is None or page_num in seen_pages:
            continue
        seen_pages.add(page_num)

        pdf_path = PDF_DIR / c["source"]
        if not pdf_path.exists():
            print(f"[Warning] PDF not found at {pdf_path}")
            continue

        doc = fitz.open(pdf_path)

        try:
            actual_page_index = int(page_num) - 1
        except ValueError:
            doc.close()
            continue

        if actual_page_index < 0 or actual_page_index >= len(doc):
            doc.close()
            continue

        page = doc[actual_page_index]
        page_text = page.get_text()

        quotes = get_verbatim_quotes(page_text, answer)

        for quote in quotes:
            if not quote:
                continue
            text_instances = page.search_for(quote)
            for inst in text_instances:
                page.add_highlight_annot(inst)

        pix = page.get_pixmap(dpi=150)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))

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

    total_q = len(df)
    avg_quality = df["avg_score"].mean()
    total_cost = df["cost"].sum()
    summary = f"### 📊 Summary\n**Total Questions:** {total_q} | **Avg Score:** {avg_quality:.3f} | **Total Est. Cost:** ${total_cost:.4f}"

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
def chat(message, history, last_exchange, agentic_mode, add_knowledge_mode):
    """Produce an answer and send it directly to chat. Keep state for HITL follow-ups."""
    message = to_text(message)

    # In add-knowledge mode: treat the typed text as content to index
    if add_knowledge_mode:
        in_scope, reason = check_content_scope(message)
        preview = message[:80] + ("…" if len(message) > 80 else "")
        if not in_scope:
            refusal = (
                f"Sorry, I can't add that to the knowledge base.\n\n"
                f"**Reason:** {reason}\n\n"
                "This assistant only stores content related to first aid, CPR, and emergency medicine. "
                "Please try again with relevant content, or type anything else to cancel."
            )
            new_history = history + [
                {"role": "user", "content": f"[Adding to knowledge base]\n_{preview}_"},
                {"role": "assistant", "content": refusal},
            ]
            return (
                new_history, None,
                gr.update(value="", placeholder="Paste text to add, then press Send…"), "",
                gr.update(interactive=False), gr.update(interactive=False),
                gr.update(interactive=bool(history)), gr.update(interactive=False),
                gr.update(interactive=False), [],
                "", None,
                gr.update(visible=False), "",
                gr.update(visible=True),    # keep panel open
                True,                       # stay in add-knowledge mode
            )
        result_msg = handle_add_to_database(message, "chat_input")
        new_history = history + [
            {"role": "user", "content": f"[Adding to knowledge base]\n_{preview}_"},
            {"role": "assistant", "content": f"✅ {result_msg}"},
        ]
        return (
            new_history, None,
            gr.update(value="", placeholder="Ask a question..."), "",
            gr.update(interactive=False), gr.update(interactive=False),
            gr.update(interactive=True), gr.update(interactive=False),
            gr.update(interactive=False), [],
            "", None,
            gr.update(visible=False), "",
            gr.update(visible=False),   # hide add-knowledge panel
            False,                      # exit add-knowledge mode
        )

    # Detect "add knowledge" intent — open inline panel
    if detect_add_knowledge_intent(message):
        bot_reply = (
            "Sure! The **Add Knowledge** panel is now open below.\n\n"
            "- 📁 **Upload a file** — use the file row to attach a PDF or .txt\n"
            "- ✏️ **Type or paste text** — just type your content in this chat box and press **Send**\n\n"
            "Once added, the content will be immediately available for future questions."
        )
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": bot_reply},
        ]
        return (
            new_history, None,
            gr.update(value="", placeholder="Paste text to add, then press Send…"), "",
            gr.update(interactive=False), gr.update(interactive=False),
            gr.update(interactive=bool(history)), gr.update(interactive=False),
            gr.update(interactive=False), [],
            "_No agent trace for knowledge addition._", None,
            gr.update(visible=False), "",
            gr.update(visible=True),    # show add-knowledge panel
            True,                       # enter add-knowledge mode
        )

    standalone_query = rewrite_query(message, history)

    if agentic_mode:
        result = run_agents(standalone_query, index, chunks)
    else:
        result = run_simple_rag(standalone_query, index, chunks)

    draft = result["draft"]
    retrieved = result["retrieved"]
    is_refusal_answer = is_refusal(draft)

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

    pending = {
        "question": message,
        "rewritten_query": standalone_query,
        "draft": draft,
        "a1": result["a1"],
        "a2": result["a2"],
        "a3": result["a3"],
        "retrieved": retrieved,
    }

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
        "agentic_mode": agentic_mode,
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
    hitl_visible = gr.update(visible=agentic_mode and not is_refusal_answer)

    return (
        new_history,
        new_last,
        "",
        "",
        fb,
        fb,
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=False),
        highlighted,
        trace_md,
        pending,
        hitl_visible,
        "",
        gr.update(visible=False),   # hide add-knowledge panel on normal answers
        False,                      # ensure add-knowledge mode is off
    )


def on_needs_more_research(pending, hint, history, last_exchange):
    if not pending:
        return history, last_exchange, gr.update(), None, [], "", ""

    print(f"\n[HITL] 'Needs more research' clicked. Hint: {hint!r}")

    intro = f"🔁 Revising based on your feedback"
    if hint:
        intro += f" (hint: _{hint}_)"
    intro += "…"

    result = rerun_with_more_research(
        pending["rewritten_query"], pending["a1"], hint=hint
    )
    revised = result["draft"]

    pending["draft"] = revised
    pending["a2"] = result["a2"]
    pending["a3"] = result["a3"]
    pending["retrieved"] = result["retrieved"]

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
        new_history,
        new_last,
        gr.update(visible=True),
        pending,
        highlighted,
        trace_md,
        "",
    )


def on_needs_rewrite(pending, hint, history, last_exchange):
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
        [],
        trace_md,
        "",
    )


# ---------- Add Knowledge handler ----------
def handle_add_to_database(text: str, source_name: str):
    """Chunk + embed new text and add it to the live FAISS index."""
    text = text.strip()
    source_name = source_name.strip() or "user_input"
    if not text:
        return "Please enter some text to add."
    result = add_to_database(text, source_name, index, chunks)
    return result["message"]


def handle_upload_file(file, source_name: str):
    """Extract text from an uploaded PDF or .txt file and add it to the database."""
    if file is None:
        return "No file uploaded."

    path = Path(file.name)
    source = source_name.strip() or path.name

    if path.suffix.lower() == ".pdf":
        doc = fitz.open(str(path))
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
    elif path.suffix.lower() == ".txt":
        text = path.read_text(encoding="utf-8")
    else:
        return "Unsupported file type. Please upload a PDF or .txt file."

    if not text.strip():
        return "Could not extract any text from the file."

    result = add_to_database(text, source, index, chunks)
    return result["message"]


# ---------- Inline Add Knowledge helpers ----------
ADD_KNOWLEDGE_PHRASES = [
    "add knowledge", "add new", "upload pdf", "upload file",
    "add information", "new information", "add text", "update the database",
    "add to the database", "i want to add", "can i add", "how to add",
    "add data", "insert knowledge", "teach you", "teach the bot",
]


def detect_add_knowledge_intent(message: str) -> bool:
    msg_lower = message.lower()
    return any(p in msg_lower for p in ADD_KNOWLEDGE_PHRASES)


def handle_upload_file_inline(file, source_name: str, history: list):
    """Upload file, scope-check, add to database, confirm in chat, and exit add-knowledge mode."""
    if file is None:
        return history, gr.update(visible=True), None, "No file uploaded.", True

    path = Path(file.name)
    source = source_name.strip() or path.name

    # Extract text first so we can scope-check before indexing
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(str(path))
        extracted = "\n\n".join(page.get_text() for page in doc)
        doc.close()
    elif path.suffix.lower() == ".txt":
        extracted = path.read_text(encoding="utf-8")
    else:
        return history, gr.update(visible=True), None, "Unsupported file type. Please upload a PDF or .txt file.", True

    if not extracted.strip():
        return history, gr.update(visible=True), None, "Could not extract any text from the file.", True

    in_scope, reason = check_content_scope(extracted)
    if not in_scope:
        refusal = (
            f"Sorry, **{source}** was not added.\n\n"
            f"**Reason:** {reason}\n\n"
            "This assistant only stores content related to first aid, CPR, and emergency medicine."
        )
        new_history = history + [{"role": "assistant", "content": refusal}]
        return new_history, gr.update(visible=True), None, "", True

    result = add_to_database(extracted, source, index, chunks)
    msg = result["message"]
    success = result["added"] > 0
    new_history = history + [{"role": "assistant", "content": f"{'✅' if success else '❌'} {msg}"}]
    return new_history, gr.update(visible=not success), None, "", not success


# ---------- UI ----------
with gr.Blocks(
    title="🚑 ER-Assistant",
    theme=gr.themes.Soft(
        primary_hue="red",
        secondary_hue="blue",
        neutral_hue="slate",
    ),
    css=CUSTOM_CSS,
) as demo:
    gr.Markdown("# ER-Assistant - First Aid & CPR Q&A")

    with gr.Tabs() as tabs:

        # ========== TAB 1: Chatbot ==========
        with gr.Tab("💬 Chat", id=0):

            last_exchange_state = gr.State(None)
            pending_state = gr.State(None)
            add_knowledge_mode_state = gr.State(False)

            with gr.Row():

                # LEFT: chat column
                with gr.Column(scale=1):
                    chatbot = gr.Chatbot(
                        height=550,
                        avatar_images=(None, "🚑"),
                    )

                    # ── Input row: message + Send ──────────────────────────────
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask a question...",
                            show_label=False,
                            scale=9,
                            container=False,
                            autofocus=True,
                        )
                        send_btn = gr.Button(
                            "Send",
                            variant="primary",
                            scale=1,
                            min_width=80,
                            interactive=False,
                        )

                    # ── Inline Add Knowledge panel (shown on intent) ───────────
                    with gr.Group(visible=False) as add_knowledge_panel:
                        gr.Markdown(
                            "📚 **Add Knowledge** — type or paste text in the box below and press **Send**, "
                            "or upload a file:"
                        )
                        with gr.Row():
                            inline_upload_file = gr.File(
                                label=None,
                                file_types=[".pdf", ".txt"],
                                scale=1,
                                min_width=120,
                            )
                            inline_upload_source = gr.Textbox(
                                placeholder="Source name (optional)",
                                show_label=False,
                                container=False,
                                scale=4,
                                lines=1,
                            )
                            inline_upload_btn = gr.Button(
                                "📎 Add file", variant="secondary", size="sm", scale=1, min_width=80
                            )
                        inline_add_status = gr.Markdown("")

                    # Mode toggle + indicator row
                    with gr.Row():
                        with gr.Column(scale=3):
                            mode_indicator = gr.Markdown(
                                '<span class="mode-pill simple">RAG mode</span>'
                            )
                    with gr.Column(scale=1, min_width=200):
                        agentic_mode = gr.Checkbox(
                            label="Agentic mode (web search + synthesizer)",
                            value=False,
                            info="Uses 3-agent pipeline with live web search.",
                        )

                    # ── Feedback row: sits right under the answer ──────────────
                    with gr.Row(elem_classes="small-btn"):
                        thumbs_up = gr.Button(
                            "👍 Helpful",
                            variant="secondary",
                            size="sm",
                            interactive=False,
                        )
                        thumbs_down = gr.Button(
                            "👎 Not helpful",
                            variant="secondary",
                            size="sm",
                            interactive=False,
                        )
                    feedback_status = gr.Markdown("")

                    # ── HITL panel (agentic mode only) ─────────────────────────
                    with gr.Group(
                        visible=False, elem_classes="hitl-panel"
                    ) as hitl_panel:
                        gr.Markdown("**Not satisfied with the answer?**")
                        review_hint = gr.Textbox(
                            label="Optional guidance",
                            placeholder="e.g. 'focus on 2025 updates' or 'make it shorter'",
                            lines=1,
                        )
                        with gr.Row(elem_classes="small-btn"):
                            more_research_btn = gr.Button(
                                "🔁 Needs more research", size="sm", interactive=False
                            )
                            rewrite_btn = gr.Button(
                                "✏️ Needs rewrite", size="sm", interactive=False
                            )

                    # ── Utility row: session-level actions grouped together ───
                    with gr.Row(elem_classes=["small-btn", "utility-row"]):
                        export_chat_btn = gr.Button(
                            "📥 Export chat",
                            size="sm",
                            interactive=False,
                        )
                        export_answer_btn = gr.Button(
                            "📥 Export answer",
                            size="sm",
                            interactive=False,
                        )
                        clear_btn = gr.Button("🗑️ Clear", size="sm")

                    export_file = gr.File(label="Download", visible=True)
                    export_status = gr.Markdown("")

                    with gr.Accordion(
                        "🔬 Agent trace", open=False, elem_classes="agent-trace"
                    ):
                        trace_display = gr.Markdown(
                            "_Ask a question to see agent activity._"
                        )

                # RIGHT: PDF viewer
                with gr.Column(scale=1):
                    pdf_gallery = gr.Gallery(
                        label="📄 Cited PDF pages",
                        show_label=True,
                        height=700,
                        object_fit="contain",
                        preview=True,
                    )

        # TAB 2: Analytics
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

    inline_upload_btn.click(
        handle_upload_file_inline,
        inputs=[inline_upload_file, inline_upload_source, chatbot],
        outputs=[chatbot, add_knowledge_panel, inline_upload_file, inline_add_status, add_knowledge_mode_state],
    )

    analytics_tab.select(
        fn=load_analytics_data,
        outputs=[stats_output, topic_plot, score_plot, question_table],
    )

    refresh_btn.click(
        fn=load_analytics_data,
        outputs=[stats_output, topic_plot, score_plot, question_table],
    )

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
        add_knowledge_panel,
        add_knowledge_mode_state,
    ]

    def update_mode_indicator(agentic):
        if agentic:
            return (
                '<span class="mode-pill agentic">Agentic mode (3 agents + web)</span>'
            )
        return '<span class="mode-pill simple">RAG mode</span>'

    agentic_mode.change(
        update_mode_indicator,
        inputs=[agentic_mode],
        outputs=[mode_indicator],
    )

    msg.submit(
        chat,
        inputs=[msg, chatbot, last_exchange_state, agentic_mode, add_knowledge_mode_state],
        outputs=chat_outputs,
    )
    send_btn.click(
        chat,
        inputs=[msg, chatbot, last_exchange_state, agentic_mode, add_knowledge_mode_state],
        outputs=chat_outputs,
    )

    msg.change(
        lambda text: gr.update(interactive=bool(text.strip())),
        inputs=[msg],
        outputs=[send_btn],
    )

    review_hint.change(
        lambda text: (
            gr.update(interactive=bool(text.strip())),
            gr.update(interactive=bool(text.strip())),
        ),
        inputs=[review_hint],
        outputs=[more_research_btn, rewrite_btn],
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
            gr.update(value="", placeholder="Ask a question..."),
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
            gr.update(visible=False),
            False,
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
    demo.launch(
        theme=gr.themes.Soft(
            primary_hue="emerald",
            secondary_hue="slate",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Nunito"), "sans-serif"],
        ).set(
            # Dark backgrounds
            body_background_fill="*neutral_950",
            body_background_fill_dark="*neutral_950",
            background_fill_primary="*neutral_900",
            background_fill_primary_dark="*neutral_900",
            background_fill_secondary="*neutral_800",
            background_fill_secondary_dark="*neutral_800",
            block_background_fill="*neutral_900",
            block_background_fill_dark="*neutral_900",
            # Text
            body_text_color="white",
            body_text_color_dark="white",
            body_text_color_subdued="*neutral_400",
            body_text_color_subdued_dark="*neutral_400",
            # Borders — soft and rounded
            block_border_width="1px",
            block_border_color="*neutral_800",
            block_border_color_dark="*neutral_800",
            block_radius="12px",
            # Buttons
            button_primary_background_fill="*primary_500",
            button_primary_background_fill_hover="*primary_400",
            button_primary_text_color="white",
        ),
        css=CUSTOM_CSS,
    )
