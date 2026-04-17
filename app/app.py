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
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from threading import Thread


api = FastAPI()


class QueryRequest(BaseModel):
    question: str
    mode: str = "simple"


@api.post("/query")
async def query(req: QueryRequest):
    standalone = rewrite_query(req.question, [])
    if req.mode == "agentic":
        result = run_agents(standalone, index, chunks)
    else:
        result = run_simple_rag(standalone, index, chunks)
    return {"answer": result["draft"]}


def run_api():
    uvicorn.run(api, host="127.0.0.1", port=5000)


Thread(target=run_api, daemon=True).start()


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from agents import (
    run_agents,
    run_simple_rag,
    is_refusal,
    rerun_with_more_research,
    rerun_with_rewrite,
)
from rag_pipeline import (
    add_to_database,
    check_content_scope,
    classify_add_mode_message,
    client,
    generate_answer,
    load_index,
    retrieve,
    safe_chat,
)
from mcp_server import mcp

Thread(
    target=lambda: mcp.run(transport="http", host="127.0.0.1", port=8765), daemon=True
).start()

USAGE_LOG = BASE_DIR / "logs" / "usage_log.jsonl"
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
.mode-pill.simple {
    background: var(--background-fill-secondary);
    color: var(--body-text-color);
}
.mode-pill.agentic {
    background: var(--primary-500);
    color: white;
}

.hitl-panel {
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 12px;
    margin-top: 8px;
}

.agent-trace {
    background: var(--background-fill-secondary) !important;
    border-left: 3px solid var(--primary-500) !important;
    padding: 8px 12px !important;
    font-size: 0.9rem !important;
    border-radius: 12px !important;
}

.small-btn button {
    font-size: 0.85rem !important;
    padding: 4px 12px !important;
}

.header-row {
    align-items: center;
    gap: 8px;
}
.mode-label {
    flex-grow: 0 !important;
    min-width: fit-content !important;
}
.mode-label p {
    margin: 0 !important;
    padding: 0 4px !important;
    white-space: nowrap;
}

.total-cost-display {
    text-align: right;
}
.total-cost-display p {
    margin: 0 !important;
}

footer { display: none !important; }
"""


# ---------- PDF Highlighting Logic ----------


def build_chat_response(
    history,
    last_exchange=None,
    msg_value="",
    msg_placeholder="Ask a question...",
    feedback_enabled=False,
    export_chat_enabled=None,
    export_answer_enabled=False,
    send_enabled=False,
    gallery=None,
    trace="",
    pending=None,
    hitl_visible=False,
    hint="",
    add_panel_visible=False,
    add_mode=False,
    hitl_btns_enabled=False,
    feedback_status="",
):
    """Build the 16-element tuple returned by chat() and related handlers."""
    if export_chat_enabled is None:
        export_chat_enabled = bool(history)
    return (
        history,
        last_exchange,
        gr.update(value=msg_value, placeholder=msg_placeholder),
        feedback_status,
        gr.update(interactive=feedback_enabled),
        gr.update(interactive=feedback_enabled),
        gr.update(interactive=export_chat_enabled),
        gr.update(interactive=export_answer_enabled),
        gr.update(interactive=send_enabled),
        gallery or [],
        trace,
        pending,
        gr.update(visible=hitl_visible),
        hint,
        gr.update(interactive=hitl_btns_enabled),
        gr.update(interactive=hitl_btns_enabled),
        gr.update(visible=add_panel_visible),
        add_mode,
    )


def extract_text_from_file(path: Path) -> tuple[str, str]:
    """Return (text, error_message). Exactly one will be non-empty."""
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(str(path))
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        if not text.strip():
            return "", "Could not extract any text from the file."
        return text, ""
    if path.suffix.lower() == ".txt":
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return "", "Could not extract any text from the file."
        return text, ""
    return "", "Unsupported file type. Please upload a PDF or .txt file."


def write_export(content: str, prefix: str, status_msg: str):
    """Write a markdown export file with a timestamped name and return (path, status)."""
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    filepath = EXPORT_DIR / filename
    filepath.write_text(content, encoding="utf-8")
    return str(filepath), status_msg


def format_scope_refusal(source_label: str, reason: str) -> str:
    """Standard refusal message when content is out of scope."""
    return (
        f"Sorry, {source_label} was not added.\n\n"
        f"**Reason:** {reason}\n\n"
        "This assistant only stores content related to first aid, CPR, and emergency medicine."
    )


def _run_rerun(
    pending,
    hint,
    history,
    rerun_fn,
    rerun_args,
    intro_verb,
    update_retrieved,
    rehighlight,
):
    """Shared body for on_needs_more_research / on_needs_rewrite."""
    if not pending:
        return history, None, gr.update(), None, [], "", ""

    print(f"\n[HITL] '{intro_verb}' clicked. Hint: {hint!r}")

    intro = f"{intro_verb} based on your feedback"
    if hint:
        intro += f" (hint: _{hint}_)"
    intro += "…"

    result = rerun_fn(*rerun_args)
    revised = result["draft"]

    pending["draft"] = revised
    pending["a3"] = result["a3"]
    if update_retrieved:
        pending["a2"] = result["a2"]
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
        if rehighlight and pending["retrieved"]
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
    response = safe_chat(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    if response is None:
        return []
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

    def fmt_time(d):
        return f"{d:.2f}s" if isinstance(d, (int, float)) else "—"

    total = sum(
        x.get("duration", 0)
        for x in (a1, a2, a3)
        if isinstance(x.get("duration"), (int, float))
    )

    lines = []
    lines.append("### Agent 1 — Internal Researcher")
    lines.append(f"- Duration: **{fmt_time(a1.get('duration'))}**")
    lines.append(f"- Domain check: **{a1.get('domain_check', '?')}**")
    lines.append(f"- Status: **{a1.get('status', '?')}**")
    n_sources = len(a1.get("sources", []))
    lines.append(f"- Internal sources retrieved: {n_sources}")
    if a1.get("answer"):
        preview = a1["answer"][:300]
        lines.append(f"- Draft: > {preview}{'…' if len(a1['answer']) > 300 else ''}")
    lines.append("")

    lines.append("### Agent 2 — External Fact-Checker")
    tc = a2.get("tool_calls", [])
    lines.append(f"- Duration: **{fmt_time(a2.get('duration'))}**")
    lines.append(f"- Tools called: {', '.join(tc) if tc else 'none'}")
    lines.append(f"- External sources found: {len(a2.get('sources', []))}")
    if a2.get("web_findings"):
        lines.append(f"- Findings: > {a2['web_findings']}")
    lines.append("")

    lines.append("### Agent 3 — Synthesizer + Router")
    lines.append(f"- Duration: **{fmt_time(a3.get('duration'))}**")
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
    response = safe_chat(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
    )
    if response is None:
        print(f"[rewrite] API failed, using original: {message!r}")
        return message
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
        return (
            "No usage data yet.",
            "### Total Cost: **$0.0000**",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

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
    summary = f"### **Total Questions:** {total_q} | **Avg Score:** {avg_quality:.3f}"
    total_cost_md = f"### Total Cost: **${total_cost:.4f}**"

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

    return summary, total_cost_md, df_topics, df, df_table


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
        lines.append(f"## {turn['role'].capitalize()}")
        lines.append(to_text(turn["content"]))
        lines.append("")

    return write_export(
        "\n".join(lines), "conversation", f"Exported {len(history)} messages."
    )


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
    return write_export(md, "answer", "Exported last answer.")


# ---------- chat handler ----------
def chat(message, history, last_exchange, mode_choice, add_knowledge_mode):
    """Produce an answer and send it directly to chat. Keep state for HITL follow-ups."""
    message = to_text(message)
    agentic_mode = mode_choice == "Agentic (internal + web search)"

    # In add-knowledge mode: classify before doing anything
    if add_knowledge_mode:
        classification = classify_add_mode_message(message)

        # ── User said something unrelated / wants to exit ────────────────
        if classification == "exit":
            bot_reply = (
                "No problem, I've closed the knowledge panel. "
                "Feel free to ask me anything, or say _'I want to add info'_ any time to reopen it."
            )
            new_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": bot_reply},
            ]
            return build_chat_response(new_history)

        # ── User asked a question — exit mode and answer normally ────────
        if classification == "question":
            add_knowledge_mode = False  # fall through to normal RAG below

        # ── User provided content to index ───────────────────────────────
        else:
            preview = message[:80] + ("…" if len(message) > 80 else "")
            in_scope, reason = check_content_scope(message)
            if not in_scope:
                refusal = (
                    format_scope_refusal("that content", reason)
                    + "\n\nPlease paste relevant content, or say _'cancel'_ to exit."
                )
                new_history = history + [
                    {
                        "role": "user",
                        "content": f"[Adding to knowledge base]\n_{preview}_",
                    },
                    {"role": "assistant", "content": refusal},
                ]
                return build_chat_response(
                    new_history,
                    msg_placeholder="Paste text to add, then press Send…",
                    add_panel_visible=True,
                    add_mode=True,
                )
            result_msg = handle_add_to_database(message, "chat_input")
            new_history = history + [
                {"role": "user", "content": f"[Adding to knowledge base]\n_{preview}_"},
                {"role": "assistant", "content": f"✅ {result_msg}"},
            ]
            return build_chat_response(new_history, export_chat_enabled=True)

    # Detect "add knowledge" intent — open inline panel
    if detect_add_knowledge_intent(message):
        bot_reply = (
            "Sure! The **Add Knowledge** panel is now open below.\n\n"
            "- **Upload a file** — use the file row to attach a PDF or .txt\n"
            "- **Type or paste text** — just type your content in this chat box and press **Send**\n\n"
            "Once added, the content will be immediately available for future questions."
        )
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": bot_reply},
        ]
        return build_chat_response(
            new_history,
            msg_placeholder="Paste text to add, then press Send…",
            trace="_No agent trace for knowledge addition._",
            add_panel_visible=True,
            add_mode=True,
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
        "a1": result.get("a1", {}),
        "a2": result.get("a2", {}),
        "a3": result.get("a3", {}),
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

    return build_chat_response(
        new_history,
        last_exchange=new_last,
        feedback_enabled=not is_refusal_answer,
        export_chat_enabled=True,
        export_answer_enabled=True,
        send_enabled=False,
        gallery=highlighted,
        trace=format_trace(pending),
        pending=pending,
        hitl_visible=agentic_mode and not is_refusal_answer,
        hitl_btns_enabled=agentic_mode and not is_refusal_answer,
    )


def on_needs_more_research(pending, hint, history, last_exchange):
    if not pending:
        return history, last_exchange, gr.update(), None, [], "", ""
    return _run_rerun(
        pending,
        hint,
        history,
        rerun_fn=lambda: rerun_with_more_research(
            pending["rewritten_query"], pending["a1"], hint=hint
        ),
        rerun_args=(),
        intro_verb="Revising",
        update_retrieved=True,
        rehighlight=True,
    )


def on_needs_rewrite(pending, hint, history, last_exchange):
    if not pending:
        return history, last_exchange, gr.update(), None, [], "", ""
    return _run_rerun(
        pending,
        hint,
        history,
        rerun_fn=lambda: rerun_with_rewrite(
            pending["rewritten_query"], pending["a1"], pending["a2"], hint=hint
        ),
        rerun_args=(),
        intro_verb="Rewriting",
        update_retrieved=False,
        rehighlight=False,
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

    text, error = extract_text_from_file(path)
    if error:
        return error

    result = add_to_database(text, source, index, chunks)
    return result["message"]


def handle_upload_file_inline(file, source_name: str, history: list):
    """Upload file, scope-check, add to database, confirm in chat, and exit add-knowledge mode."""
    if file is None:
        return history, gr.update(visible=True), None, "No file uploaded.", True

    path = Path(file.name)
    source = source_name.strip() or path.name

    extracted, error = extract_text_from_file(path)
    if error:
        return history, gr.update(visible=True), None, error, True

    in_scope, reason = check_content_scope(extracted)
    if not in_scope:
        refusal = format_scope_refusal(f"**{source}**", reason)
        new_history = history + [{"role": "assistant", "content": refusal}]
        return new_history, gr.update(visible=True), None, "", True

    result = add_to_database(extracted, source, index, chunks)
    msg = result["message"]
    success = result["added"] > 0
    new_history = history + [
        {"role": "assistant", "content": f"{'✅' if success else '❌'} {msg}"}
    ]
    return new_history, gr.update(visible=not success), None, "", not success


# ---------- Inline Add Knowledge helpers ----------
ADD_KNOWLEDGE_PHRASES = [
    "add knowledge",
    "add new",
    "upload pdf",
    "upload file",
    "add information",
    "new information",
    "add text",
    "update the database",
    "add to the database",
    "i want to add",
    "can i add",
    "how to add",
    "add data",
    "insert knowledge",
    "teach you",
    "teach the bot",
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
        return (
            history,
            gr.update(visible=True),
            None,
            "Unsupported file type. Please upload a PDF or .txt file.",
            True,
        )

    if not extracted.strip():
        return (
            history,
            gr.update(visible=True),
            None,
            "Could not extract any text from the file.",
            True,
        )

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
    new_history = history + [
        {"role": "assistant", "content": f"{'✅' if success else '❌'} {msg}"}
    ]
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
    with gr.Row(elem_classes="header-row"):
        theme_toggle = gr.Button("Toggle theme", size="sm", scale=0)
        mode_dropdown = gr.Dropdown(
            choices=[
                "Rag (internal search)",
                "Agentic (internal + web search)",
            ],
            value="Rag (internal search)",
            show_label=False,
            container=False,
            interactive=True,
            scale=0,
            min_width=320,
        )
    theme_toggle.click(
        fn=None,
        inputs=[],
        outputs=[],
        js="""
        () => {
            document.body.classList.toggle('dark');
        }
        """,
    )

    with gr.Tabs() as tabs:

        # ========== TAB 1: Chatbot ==========
        with gr.Tab("Chat", id=0):

            last_exchange_state = gr.State(None)
            pending_state = gr.State(None)
            add_knowledge_mode_state = gr.State(False)

            with gr.Row():

                # LEFT: chat column
                with gr.Column(scale=1):
                    chatbot = gr.Chatbot(
                        height=550,
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
                            "**Add Knowledge** — type or paste text in the box below and press **Send**, "
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
                                "Add file",
                                variant="secondary",
                                size="sm",
                                scale=1,
                                min_width=80,
                            )
                        inline_add_status = gr.Markdown("")

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
                                "Needs more research", size="sm", interactive=False
                            )
                            rewrite_btn = gr.Button(
                                "Needs rewrite", size="sm", interactive=False
                            )

                    # ── Utility row: session-level actions grouped together ───
                    with gr.Row(elem_classes=["small-btn", "utility-row"]):
                        export_chat_btn = gr.Button(
                            "Export chat",
                            size="sm",
                            interactive=False,
                        )
                        export_answer_btn = gr.Button(
                            "Export answer",
                            size="sm",
                            interactive=False,
                        )
                        clear_btn = gr.Button("Clear", size="sm")

                    # export_file = gr.File(label="Download", visible=False)
                    export_file = gr.DownloadButton("Download", visible=False)
                    export_status = gr.Markdown("")

                    with gr.Accordion(
                        "Agent trace", open=False, elem_classes="agent-trace"
                    ):
                        trace_display = gr.Markdown(
                            "_Ask a question to see agent activity._"
                        )

                # RIGHT: PDF viewer
                with gr.Column(scale=1):
                    pdf_gallery = gr.Gallery(
                        label="Cited PDF pages",
                        show_label=True,
                        height=700,
                        object_fit="contain",
                        preview=True,
                    )

        # TAB 2: Analytics
        with gr.Tab("Analytics", id=1) as analytics_tab:
            with gr.Row():
                gr.Markdown("**Usage Analytics**")
                total_cost_output = gr.Markdown(
                    "### Total Cost: **$0.0000**",
                    elem_classes="total-cost-display",
                )
            stats_output = gr.Markdown("Loading...")

            with gr.Row():
                topic_plot = gr.BarPlot(x="Topic", y="Count", title="Common Topics")
                score_plot = gr.LinePlot(
                    x="short_time",
                    y="avg_score",
                    title="Quality Trend",
                    x_title="timestamp",
                    x_label_angle=-90,
                )

            gr.Markdown("### Detailed Question Log")
            question_table = gr.Dataframe(
                headers=["Time", "Question", "Score", "Cost ($)"],
                datatype=["str", "str", "number", "number"],
                wrap=True,
            )

    # ---------- Event Handlers ----------

    inline_upload_btn.click(
        handle_upload_file_inline,
        inputs=[inline_upload_file, inline_upload_source, chatbot],
        outputs=[
            chatbot,
            add_knowledge_panel,
            inline_upload_file,
            inline_add_status,
            add_knowledge_mode_state,
        ],
    )

    analytics_tab.select(
        fn=load_analytics_data,
        outputs=[
            stats_output,
            total_cost_output,
            topic_plot,
            score_plot,
            question_table,
        ],
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
        more_research_btn,
        rewrite_btn,
        add_knowledge_panel,
        add_knowledge_mode_state,
    ]

    msg.submit(
        chat,
        inputs=[
            msg,
            chatbot,
            last_exchange_state,
            mode_dropdown,
            add_knowledge_mode_state,
        ],
        outputs=chat_outputs,
    )
    send_btn.click(
        chat,
        inputs=[
            msg,
            chatbot,
            last_exchange_state,
            mode_dropdown,
            add_knowledge_mode_state,
        ],
        outputs=chat_outputs,
    )

    msg.change(
        lambda text: gr.update(interactive=bool(text.strip())),
        inputs=[msg],
        outputs=[send_btn],
    )

    review_hint.change(
        lambda text: (gr.update(), gr.update()),
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
        lambda: build_chat_response([]),
        outputs=chat_outputs,
    )

    def export_conversation_with_visibility(history):
        filepath, status = export_conversation(history)
        return gr.update(value=filepath, visible=filepath is not None), status

    def export_last_answer_with_visibility(last_exchange):
        filepath, status = export_last_answer(last_exchange)
        return gr.update(value=filepath, visible=filepath is not None), status

    export_chat_btn.click(
        export_conversation_with_visibility,
        inputs=[chatbot],
        outputs=[export_file, export_status],
    )
    export_answer_btn.click(
        export_last_answer_with_visibility,
        inputs=[last_exchange_state],
        outputs=[export_file, export_status],
    )


if __name__ == "__main__":
    demo.launch(
        css=CUSTOM_CSS,
    )
