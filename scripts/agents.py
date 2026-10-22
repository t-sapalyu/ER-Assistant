import asyncio
import json
import time
from cost_tracker import track_cost
from fastmcp import Client
from mcp_server import TOOL_SCHEMAS
from rag_pipeline import best_score, client, generate_answer, retrieve, safe_chat

MCP_SERVER_URL = "http://127.0.0.1:8765/mcp"


async def _call_mcp_tool(tool_name: str, args: dict):
    async with Client(MCP_SERVER_URL) as mcp_client:
        result = await mcp_client.call_tool(tool_name, args)
        if hasattr(result, "data") and result.data is not None:
            return result.data
        if hasattr(result, "content") and result.content:
            return [block.text for block in result.content if hasattr(block, "text")]
        return []


def call_mcp_tool_sync(tool_name: str, args: dict):
    return asyncio.run(_call_mcp_tool(tool_name, args))


DOMAIN_DESCRIPTION = (
    "first aid, CPR, emergency medical response, injuries, " "and pre-hospital care"
)

REFUSAL_MARKERS = (
    "don't have",
    "do not have",
    "cannot answer",
    "no information",
    "not enough information",
)

MAX_AGENT_ITERATIONS = 2  # total Agent 2 + Agent 3 loops before forced finalization


def is_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(m in low for m in REFUSAL_MARKERS)


# ----------------------------------------------------------------------
# Agent 1 — Internal Researcher
# ----------------------------------------------------------------------
def agent_1_researcher(query, index, chunks) -> dict:
    t0 = time.perf_counter()
    print(f"\n{'=' * 60}")
    print(f"[Agent 1] Starting. Query: {query!r}")

    classifier_prompt = (
        f"Decide if the user's question is about the TOPIC of {DOMAIN_DESCRIPTION}, "
        f"regardless of whether specific information is available. "
        f"A question about recent updates, news, or changes to this topic is still "
        f"IN domain. Only answer 'out' if the question is about a completely "
        f"unrelated topic (weather, sports, cooking, finance, etc.).\n\n"
        f"Respond with exactly one word: 'in' or 'out'.\n\n"
        f"Question: {query}"
    )
    resp = safe_chat(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": classifier_prompt}],
        temperature=0,
        max_tokens=5,
    )
    if resp is None:
        return {
            "status": "out_of_domain",
            "answer": "",
            "sources": [],
            "domain_check": "error",
            "duration": time.perf_counter() - t0,
        }
    verdict = resp.choices[0].message.content.strip().lower()
    in_domain = verdict.startswith("in")
    print(f"[Agent 1] Domain check: {'IN' if in_domain else 'OUT'} (raw: {verdict!r})")

    if not in_domain:
        print(f"[Agent 1] Returning: out_of_domain")
        return {
            "status": "out_of_domain",
            "answer": "",
            "sources": [],
            "domain_check": "out",
            "duration": time.perf_counter() - t0,
        }

    retrieved = retrieve(query, index, chunks)
    top_score = best_score(retrieved)
    print(f"[Agent 1] Retrieved {len(retrieved)} chunks. Top score: {top_score:.3f}")

    draft = generate_answer(query, retrieved)
    refused = is_refusal(draft)
    print(f"[Agent 1] Draft generated. Refused: {refused}")
    if not refused:
        print(
            f"[Agent 1] Draft preview: {draft[:140]}{'...' if len(draft) > 140 else ''}"
        )

    if refused:
        print(f"[Agent 1] Returning: no_knowledge")
        return {
            "status": "no_knowledge",
            "answer": "",
            "sources": retrieved,
            "domain_check": "in",
            "duration": time.perf_counter() - t0,
        }

    print(f"[Agent 1] Returning: has_answer")
    return {
        "status": "has_answer",
        "answer": draft,
        "sources": retrieved,
        "domain_check": "in",
        "duration": time.perf_counter() - t0,
    }


# ----------------------------------------------------------------------
# Agent 2 — External Fact-Checker
# ----------------------------------------------------------------------
def agent_2_fact_checker(query, agent_1_output, extra_hint: str = "") -> dict:
    t0 = time.perf_counter()
    """Use MCP tools to fact-check or supplement agent 1's answer.

    extra_hint: optional instruction from Agent 3's router or the human user
    to guide the retry (e.g. 'focus on 2025 guideline changes').
    """
    status = agent_1_output.get("status")
    draft = agent_1_output.get("answer", "")
    print(f"\n[Agent 2] Starting. Agent 1 status: {status}")
    if extra_hint:
        print(f"[Agent 2] Retry hint received: {extra_hint!r}")

    if status == "no_knowledge":
        system = (
            "You are a research assistant. The internal knowledge base has no "
            "information about the user's question. Use the available tools to "
            "find reliable external information. Choose the tool best suited "
            "to the question — Wikipedia for established medical facts, "
            "web_search for recent news or updates."
        )
        user = f"Question: {query}\n\nSearch for relevant information."
        print(f"[Agent 2] Mode: search from scratch")
    else:
        system = (
            "You are a fact-checker. An internal knowledge base has produced "
            "a draft answer. Verify whether it is still current and accurate, "
            "and find updates or corrections. Use the available tools. "
            "Choose wisely — Wikipedia for established facts, web_search for "
            "recent changes. If the draft is fine, you can skip tool calls."
        )
        user = (
            f"Question: {query}\n\n"
            f"Draft answer from internal corpus:\n{draft}\n\n"
            f"Verify this against external sources."
        )
        print(f"[Agent 2] Mode: fact-check existing draft")

    if extra_hint:
        user += f"\n\nAdditional guidance for this search:\n{extra_hint}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    resp = safe_chat(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOL_SCHEMAS,
        temperature=0,
        max_tokens=500,
    )
    if resp is None:
        return {
            "web_findings": "",
            "sources": [],
            "tool_calls": [],
            "duration": time.perf_counter() - t0,
        }
    message = resp.choices[0].message
    tool_calls = message.tool_calls or []

    if not tool_calls:
        print(f"[Agent 2] LLM chose NOT to use any tool.")
        return {
            "web_findings": "",
            "sources": [],
            "tool_calls": [],
            "duration": time.perf_counter() - t0,
        }

    print(f"[Agent 2] LLM requested {len(tool_calls)} tool call(s)")

    all_sources = []
    tool_names_used = []
    messages.append(message.model_dump())

    for tc in tool_calls:
        fn_name = tc.function.name
        tool_names_used.append(fn_name)
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            args = {}
        print(f"[Agent 2]   → calling {fn_name}({args})")

        try:
            result = call_mcp_tool_sync(fn_name, args)
        except Exception as e:
            print(f"[Agent 2]   ✗ MCP call failed: {e}")
            result = []
        print(f"[Agent 2]   ← {fn_name} returned {len(result)} result(s)")

        all_sources.extend([{**item, "tool": fn_name} for item in result])

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            }
        )

    if not all_sources:
        print(f"[Agent 2] All tools returned empty.")
        return {
            "web_findings": "",
            "sources": [],
            "tool_calls": tool_names_used,
            "duration": time.perf_counter() - t0,
        }

    messages.append(
        {
            "role": "user",
            "content": (
                "Based on the tool results above, write a brief summary "
                "(2-3 sentences) of what external sources say. If they confirm "
                "the draft, say so. If they suggest corrections, highlight those."
            ),
        }
    )
    resp2 = safe_chat(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
        max_tokens=300,
    )
    if resp2 is None:
        return {
            "web_findings": "",
            "sources": all_sources,
            "tool_calls": tool_names_used,
            "duration": time.perf_counter() - t0,
        }
    findings = resp2.choices[0].message.content.strip()

    print(f"[Agent 2] Findings: {findings[:140]}{'...' if len(findings) > 140 else ''}")
    print(f"[Agent 2] Total external sources: {len(all_sources)}")

    return {
        "web_findings": findings,
        "sources": all_sources,
        "tool_calls": tool_names_used,
        "duration": time.perf_counter() - t0,
    }


# ----------------------------------------------------------------------
# Agent 3 — Synthesizer + Router
# ----------------------------------------------------------------------
def agent_3_synthesizer_router(
    query, agent_1_output, agent_2_output, rewrite_hint: str = ""
) -> dict:
    t0 = time.perf_counter()
    """Synthesize a draft AND decide if more research is needed.

    Returns:
        {
            "action": "answer" | "retry_agent_2",
            "draft": str,              # the synthesized answer
            "reason": str,             # LLM's explanation
            "retry_hint": str,         # guidance for Agent 2 retry (if action=="retry")
        }
    """
    a1_status = agent_1_output.get("status")
    a1_answer = agent_1_output.get("answer", "")
    a2_findings = agent_2_output.get("web_findings", "")
    a2_sources = agent_2_output.get("sources", [])
    has_a2 = bool(a2_findings)

    print(f"\n[Agent 3] Starting. A1 status: {a1_status} | A2 has findings: {has_a2}")
    if rewrite_hint:
        print(f"[Agent 3] Rewrite hint received: {rewrite_hint!r}")

    # Trivial paths — no routing decision needed
    if a1_status == "out_of_domain":
        print(f"[Agent 3] Path: out_of_domain refusal (no routing)")
        return {
            "action": "answer",
            "draft": (
                "I don't have information about this — it falls outside my "
                "domain (first aid and emergency response)."
            ),
            "reason": "Out-of-domain question.",
            "retry_hint": "",
            "duration": time.perf_counter() - t0,
        }

    if a1_status == "no_knowledge" and not has_a2:
        print(f"[Agent 3] Path: no info in corpus or web (no routing)")
        return {
            "action": "answer",
            "draft": (
                "I don't have information about this in my internal corpus, "
                "and external sources didn't return useful results either."
            ),
            "reason": "No information found anywhere.",
            "retry_hint": "",
            "duration": time.perf_counter() - t0,
        }

    # Build the draft (same as before)
    if not has_a2:
        draft = a1_answer
        synth_type = "internal_only"
    elif a1_status == "no_knowledge":
        sources_block = _format_web_sources(a2_sources)
        draft = (
            f"My internal corpus doesn't cover this, but external sources indicate:"
            f"\n\n{a2_findings}\n\n{sources_block}"
        )
        synth_type = "external_only"
    else:
        draft = _synthesize_combined(
            query, a1_answer, a2_findings, a2_sources, rewrite_hint
        )
        synth_type = "combined"

    print(f"[Agent 3] Draft produced ({synth_type})")
    print(f"[Agent 3] Draft preview: {draft[:140]}{'...' if len(draft) > 140 else ''}")

    # Routing decision — skip for out_of_domain / no_info which already returned above
    router_decision = _agent_3_router_decide(
        query, a1_status, draft, a2_findings, a2_sources
    )
    print(
        f"[Agent 3] Router decision: {router_decision['action']} — {router_decision['reason']}"
    )

    return {
        "action": router_decision["action"],
        "draft": draft,
        "reason": router_decision["reason"],
        "retry_hint": router_decision.get("retry_hint", ""),
        "duration": time.perf_counter() - t0,
    }


def _synthesize_combined(query, a1_answer, a2_findings, a2_sources, rewrite_hint):
    """Produce a final combined answer when both internal and external exist."""
    internal_block = f"**From the Red Cross Guide:**\n{a1_answer}"
    external_block = (
        f"**External verification:**\n{a2_findings}\n\n"
        f"{_format_web_sources(a2_sources)}"
    )

    extra = ""
    if rewrite_hint:
        extra = (
            f"\n\nThe previous draft needs revision. Apply this guidance: "
            f"{rewrite_hint}"
        )

    prompt = (
        "You are combining two sources of information into a single clear answer. "
        "Start with the internal (Red Cross Guide) information, then mention any "
        "external updates or confirmations. Preserve all citations from the internal "
        "answer — they look like (Source: file.pdf, Page N). Do NOT alter them. "
        "After the main answer, list external sources with their URLs clearly "
        "labeled. Keep it concise.\n\n"
        f"Question: {query}\n\n{internal_block}\n\n{external_block}{extra}\n\nFinal answer:"
    )
    resp = safe_chat(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=600,
    )
    if resp is None:
        return a1_answer  # fall back to internal-only answer
    return resp.choices[0].message.content.strip()


def _agent_3_router_decide(query, a1_status, draft, a2_findings, a2_sources):
    """LLM decides: is this draft ready, or does Agent 2 need to retry?"""
    summary_parts = [f"Question: {query}"]
    summary_parts.append(f"Internal corpus status: {a1_status}")
    summary_parts.append(f"External findings present: {'yes' if a2_findings else 'no'}")
    if a2_findings:
        summary_parts.append(f"External findings: {a2_findings[:300]}")
        summary_parts.append(f"Number of external sources: {len(a2_sources)}")
    summary_parts.append(f"Current draft answer:\n{draft}")

    prompt = (
        "You are a quality router. Given the question, the synthesis so far, and "
        "what was (or wasn't) found externally, decide if the draft answer is "
        "ready OR if the external fact-checker should retry with a refined query.\n\n"
        "Answer is READY when:\n"
        "- It addresses the question directly.\n"
        "- Internal and external sources agree, OR internal is authoritative and "
        "  external didn't add value.\n"
        "- External sources (if any) are from credible domains (Wikipedia, gov, "
        "  well-known medical orgs).\n\n"
        "Retry external research when:\n"
        "- The draft is vague or incomplete for a time-sensitive question.\n"
        "- External findings were empty or came from low-quality domains.\n"
        "- Internal and external sources contradict and need reconciliation.\n\n"
        "Respond in JSON only, with this exact schema:\n"
        '{"action": "answer" OR "retry_agent_2", "reason": "<one short sentence>", '
        '"retry_hint": "<if retry, a specific guidance string; else empty>"}\n\n'
        f"--- context ---\n" + "\n\n".join(summary_parts)
    )

    resp = safe_chat(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=200,
    )
    if resp is None:
        return {
            "action": "answer",
            "reason": "API error; defaulting to answer.",
            "retry_hint": "",
        }
    try:
        decision = json.loads(resp.choices[0].message.content)
        if decision.get("action") not in ("answer", "retry_agent_2"):
            decision["action"] = "answer"
        return decision
    except json.JSONDecodeError:
        # If the router misbehaves, default to answering (safe fallback)
        return {
            "action": "answer",
            "reason": "Router response parse failure; defaulting to answer.",
            "retry_hint": "",
        }


def _format_web_sources(sources: list[dict]) -> str:
    if not sources:
        return ""
    lines = ["**External sources:**"]
    for s in sources[:5]:
        title = s.get("title", "(untitled)")
        url = s.get("url", "")
        tool = s.get("tool", "")
        lines.append(f"- [{title}]({url}) — _via {tool}_")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Orchestrator with auto-retry loop
# ----------------------------------------------------------------------
def run_agents(query, index, chunks) -> dict:
    """Run agents 1 -> 2 -> 3, with Agent 3 able to trigger a single auto-retry.

    Returns a dict with the draft, retrieved sources, and agent trace.
    The human-in-the-loop review step happens *after* this function returns,
    in the UI layer.
    """
    print(f"\n{'#' * 60}")
    print(f"# run_agents: {query!r}")
    print(f"{'#' * 60}")

    a1 = agent_1_researcher(query, index, chunks)

    # Out-of-domain: skip straight to refusal
    if a1["status"] == "out_of_domain":
        a3 = agent_3_synthesizer_router(query, a1, {"web_findings": "", "sources": []})
        print(f"\n[run_agents] Done (out_of_domain short-circuit)")
        print(f"{'#' * 60}\n")
        return {
            "draft": a3["draft"],
            "retrieved": [],
            "a1": a1,
            "a2": {"web_findings": "", "sources": [], "tool_calls": [], "duration": 0},
            "a3": a3,
            "iterations": 0,
            "auto_retry_triggered": False,
        }

    a2 = agent_2_fact_checker(query, a1)
    a3 = agent_3_synthesizer_router(query, a1, a2)

    iteration = 1
    auto_retry_triggered = False

    while a3["action"] == "retry_agent_2" and iteration < MAX_AGENT_ITERATIONS:
        print(f"\n[Orchestrator] Agent 3 requested retry (iteration {iteration + 1})")
        auto_retry_triggered = True
        a2 = agent_2_fact_checker(query, a1, extra_hint=a3.get("retry_hint", ""))
        a3 = agent_3_synthesizer_router(query, a1, a2)
        iteration += 1

    if a3["action"] == "retry_agent_2":
        print(f"\n[Orchestrator] Max iterations hit. Forcing 'answer' action.")
        a3["action"] = "answer"
        a3["reason"] += " (forced after max retries)"

    print(f"\n[run_agents] Done after {iteration} iteration(s)")
    print(f"{'#' * 60}\n")

    return {
        "draft": a3["draft"],
        "retrieved": a1["sources"],
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "iterations": iteration,
        "auto_retry_triggered": auto_retry_triggered,
    }


def run_simple_rag(query, index, chunks) -> dict:
    """RAG mode: no agents, no web search. Just retrieve + generate."""
    t0 = time.perf_counter()
    print(f"\n{'#' * 60}")
    print(f"# run_simple_rag: {query!r}")
    print(f"{'#' * 60}")

    retrieved = retrieve(query, index, chunks)
    top_score = best_score(retrieved)
    print(f"[Simple RAG] Retrieved {len(retrieved)} chunks. Top score: {top_score:.3f}")

    answer = generate_answer(query, retrieved)
    duration = time.perf_counter() - t0
    print(f"[Simple RAG] Answer: {answer[:140]}{'...' if len(answer) > 140 else ''}")
    print(f"{'#' * 60}\n")

    return {
        "draft": answer,
        "retrieved": retrieved,
        "a1": {
            "status": "simple_mode",
            "answer": answer,
            "sources": retrieved,
            "domain_check": "n/a",
            "duration": duration,
        },
        "a2": {"web_findings": "", "sources": [], "tool_calls": []},
        "a3": {
            "action": "answer",
            "reason": "Simple RAG mode — no routing.",
            "retry_hint": "",
        },
        "iterations": 0,
        "auto_retry_triggered": False,
    }


# ----------------------------------------------------------------------
# Human-in-the-loop feedback functions (called from the UI)
# ----------------------------------------------------------------------
def rerun_with_more_research(query, a1, hint: str = "") -> dict:

    print(f"\n[HITL] User requested more research. Hint: {hint!r}")
    a2 = agent_2_fact_checker(query, a1, extra_hint=hint)
    a3 = agent_3_synthesizer_router(query, a1, a2)
    if a3["action"] == "retry_agent_2":
        a3["action"] = "answer"  # don't retry again after human already intervened
    return {"draft": a3["draft"], "a2": a2, "a3": a3, "retrieved": a1["sources"]}


def rerun_with_rewrite(query, a1, a2, hint: str = "") -> dict:

    print(f"\n[HITL] User requested rewrite. Hint: {hint!r}")
    a3 = agent_3_synthesizer_router(query, a1, a2, rewrite_hint=hint)
    if a3["action"] == "retry_agent_2":
        a3["action"] = "answer"  # don't trigger research retry on a rewrite request
    return {"draft": a3["draft"], "a2": a2, "a3": a3, "retrieved": a1["sources"]}
