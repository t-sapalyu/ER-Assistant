"""Three-agent system: Internal Researcher -> Web Fact-Checker -> Synthesizer."""

import json

from cost_tracker import track_cost
from mcp_server import TOOL_FUNCTIONS, TOOL_SCHEMAS
from rag_pipeline import client, generate_answer, retrieve


DOMAIN_DESCRIPTION = (
    "first aid, CPR, emergency medical response, injuries, " "and pre-hospital care"
)

# DOMAIN_DESCRIPTION = (
#     "first aid, CPR, emergency medical response, injuries, and "
#     "pre-hospital care topics covered in the Canadian Red Cross "
#     "Comprehensive Guide for First Aid & CPR"
# )

REFUSAL_MARKERS = (
    "don't have",
    "do not have",
    "cannot answer",
    "no information",
    "not enough information",
)


def is_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(m in low for m in REFUSAL_MARKERS)


# ----------------------------------------------------------------------
# Agent 1 — Internal Researcher
# ----------------------------------------------------------------------
def agent_1_researcher(query, index, chunks) -> dict:
    """Classify domain, retrieve from corpus, draft answer if possible."""
    print(f"\n{'=' * 60}")
    print(f"[Agent 1] Starting. Query: {query!r}")

    # Step 1: domain classification
    classifier_prompt = (
        f"Decide if the user's question is about the TOPIC of {DOMAIN_DESCRIPTION}, "
        f"regardless of whether specific information is available. "
        f"A question about recent updates, news, or changes to this topic is still "
        f"IN domain. Only answer 'out' if the question is about a completely "
        f"unrelated topic (weather, sports, cooking, finance, etc.).\n\n"
        f"Respond with exactly one word: 'in' or 'out'.\n\n"
        f"Question: {query}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": classifier_prompt}],
        temperature=0,
        max_tokens=5,
    )
    track_cost(resp)
    verdict = resp.choices[0].message.content.strip().lower()
    in_domain = verdict.startswith("in")
    print(f"[Agent 1] Domain check: {'IN' if in_domain else 'OUT'} (raw: {verdict!r})")

    if not in_domain:
        print(f"[Agent 1] Returning: out_of_domain (skipping retrieval)")
        return {
            "status": "out_of_domain",
            "answer": "",
            "sources": [],
            "domain_check": "out",
        }

    # Step 2: retrieve from corpus
    retrieved = retrieve(query, index, chunks)
    top_score = retrieved[0].get("score", 0.0) if retrieved else 0.0
    print(f"[Agent 1] Retrieved {len(retrieved)} chunks. Top score: {top_score:.3f}")

    # Step 3: draft an answer from the retrieved context
    draft = generate_answer(query, retrieved)
    refused = is_refusal(draft)
    print(f"[Agent 1] Draft generated. Refused: {refused}")
    if not refused:
        print(
            f"[Agent 1] Draft preview: {draft[:140]}{'...' if len(draft) > 140 else ''}"
        )

    if refused:
        print(f"[Agent 1] Returning: no_knowledge (corpus silent on this)")
        return {
            "status": "no_knowledge",
            "answer": "",
            "sources": retrieved,
            "domain_check": "in",
        }

    print(f"[Agent 1] Returning: has_answer")
    return {
        "status": "has_answer",
        "answer": draft,
        "sources": retrieved,
        "domain_check": "in",
    }


# ----------------------------------------------------------------------
# Agent 2 — External Fact-Checker (uses MCP tools via function-calling)
# ----------------------------------------------------------------------
def agent_2_fact_checker(query, agent_1_output) -> dict:
    """Use MCP tools to fact-check or supplement agent 1's answer."""
    status = agent_1_output.get("status")
    draft = agent_1_output.get("answer", "")
    print(f"\n[Agent 2] Starting. Agent 1 status: {status}")

    if status == "no_knowledge":
        system = (
            "You are a research assistant. The internal knowledge base has no "
            "information about the user's question. Use the available tools to "
            "find reliable external information. Choose the tool best suited "
            "to the question — Wikipedia for established medical facts, "
            "web_search for recent news or updates."
        )
        user = f"Question: {query}\n\nSearch for relevant information."
        print(f"[Agent 2] Mode: search from scratch (no internal knowledge)")
    else:
        system = (
            "You are a fact-checker. An internal knowledge base has produced "
            "a draft answer. Your job is to verify whether this information "
            "is still current and accurate, and find any updates or corrections. "
            "Use the available tools. Choose wisely — Wikipedia for established "
            "facts, web_search for recent changes or news. If the draft seems "
            "fine and nothing has changed, you can skip tool calls."
        )
        user = (
            f"Question: {query}\n\n"
            f"Draft answer from internal corpus:\n{draft}\n\n"
            f"Verify this against external sources."
        )
        print(f"[Agent 2] Mode: fact-check existing draft")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOL_SCHEMAS,
        temperature=0,
        max_tokens=500,
    )
    track_cost(resp)

    message = resp.choices[0].message
    tool_calls = message.tool_calls or []

    if not tool_calls:
        print(f"[Agent 2] LLM chose NOT to use any tool. Returning empty.")
        return {"web_findings": "", "sources": [], "tool_calls": []}

    print(f"[Agent 2] LLM requested {len(tool_calls)} tool call(s)")

    # Execute each tool call and collect results
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

        fn = TOOL_FUNCTIONS.get(fn_name)
        result = fn(**args) if fn else []
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
        print(f"[Agent 2] All tools returned empty. No findings to synthesize.")
        return {"web_findings": "", "sources": [], "tool_calls": tool_names_used}

    # Summarize what was found
    messages.append(
        {
            "role": "user",
            "content": (
                "Based on the tool results above, write a brief summary "
                "(2-3 sentences) of what external sources say about the "
                "question. If the sources confirm the draft answer, say so. "
                "If they suggest corrections or updates, highlight those."
            ),
        }
    )
    resp2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
        max_tokens=300,
    )
    track_cost(resp2)
    findings = resp2.choices[0].message.content.strip()

    print(f"[Agent 2] Findings: {findings[:140]}{'...' if len(findings) > 140 else ''}")
    print(f"[Agent 2] Total external sources: {len(all_sources)}")

    return {
        "web_findings": findings,
        "sources": all_sources,
        "tool_calls": tool_names_used,
    }


# ----------------------------------------------------------------------
# Agent 3 — Synthesizer
# ----------------------------------------------------------------------
def agent_3_synthesizer(query, agent_1_output, agent_2_output) -> str:
    """Merge internal and external findings into a labeled final answer."""
    a1_status = agent_1_output.get("status")
    a1_answer = agent_1_output.get("answer", "")
    a2_findings = agent_2_output.get("web_findings", "")
    a2_sources = agent_2_output.get("sources", [])

    has_a2 = bool(a2_findings)
    print(f"\n[Agent 3] Starting. A1 status: {a1_status} | A2 has findings: {has_a2}")

    if a1_status == "out_of_domain":
        print(f"[Agent 3] Path: out_of_domain refusal")
        return (
            "I don't have information about this — it falls outside my "
            "domain (first aid and emergency response)."
        )

    if a1_status == "no_knowledge" and not has_a2:
        print(f"[Agent 3] Path: no info in corpus or web")
        return (
            "I don't have information about this in my internal corpus, "
            "and external sources didn't return useful results either."
        )

    if not has_a2:
        print(f"[Agent 3] Path: internal only (pass-through, no external findings)")
        return a1_answer

    if a1_status == "no_knowledge":
        print(f"[Agent 3] Path: external only (corpus was silent)")
        sources_block = _format_web_sources(a2_sources)
        return (
            f"My internal corpus doesn't cover this, but external sources "
            f"indicate:\n\n{a2_findings}\n\n{sources_block}"
        )

    # Both internal and external — synthesize with LLM
    print(f"[Agent 3] Path: synthesizing internal + external")
    internal_block = f"**From the Red Cross Guide:**\n{a1_answer}"
    external_block = (
        f"**External verification:**\n{a2_findings}\n\n"
        f"{_format_web_sources(a2_sources)}"
    )

    prompt = (
        "You are an assistant combining two sources of information into a "
        "single clear answer. Start with the internal (Red Cross Guide) "
        "information, then mention any external updates or confirmations. "
        "Preserve all citations from the internal answer — they look like "
        "(Source: file.pdf, Page N). Do NOT remove or alter them. "
        "After the main answer, list external sources with their URLs "
        "clearly labeled. Keep it concise.\n\n"
        f"Question: {query}\n\n"
        f"{internal_block}\n\n"
        f"{external_block}\n\n"
        "Final answer:"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=600,
    )
    track_cost(resp)
    final = resp.choices[0].message.content.strip()
    print(f"[Agent 3] Final answer: {final[:140]}{'...' if len(final) > 140 else ''}")
    return final


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
# Orchestrator
# ----------------------------------------------------------------------
def run_agents(query, index, chunks) -> dict:
    """Run all three agents and return the final answer + a trace for UI display."""
    print(f"\n{'#' * 60}")
    print(f"# run_agents: {query!r}")
    print(f"{'#' * 60}")

    a1 = agent_1_researcher(query, index, chunks)

    if a1["status"] == "out_of_domain":
        final = agent_3_synthesizer(query, a1, {"web_findings": "", "sources": []})
        print(f"\n[run_agents] Done (short-circuit after out_of_domain).")
        print(f"{'#' * 60}\n")
        return {
            "answer": final,
            "retrieved": [],
            "trace": {
                "agent_1": a1,
                "agent_2": {"status": "skipped (out of domain)"},
                "agent_3": {"final_answer": final},
            },
        }

    a2 = agent_2_fact_checker(query, a1)
    final = agent_3_synthesizer(query, a1, a2)

    print(f"\n[run_agents] Done.")
    print(f"{'#' * 60}\n")

    return {
        "answer": final,
        "retrieved": a1["sources"],
        "trace": {
            "agent_1": {
                "status": a1["status"],
                "domain_check": a1["domain_check"],
                "draft_answer": a1["answer"],
                "n_sources": len(a1["sources"]),
            },
            "agent_2": {
                "tool_calls": a2.get("tool_calls", []),
                "findings": a2.get("web_findings", ""),
                "n_external_sources": len(a2.get("sources", [])),
                "sources": a2.get("sources", []),
            },
            "agent_3": {"final_answer": final},
        },
    }
