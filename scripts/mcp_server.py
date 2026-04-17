"""FastMCP server exposing web/Wikipedia search tools for the fact-checker agent.

Designed to be used in-process: other modules import the tool functions directly.
Can also run standalone (`python mcp_server.py`) as a real MCP stdio server.
"""

from typing import Any

import wikipediaapi
from ddgs import DDGS
from fastmcp import FastMCP

mcp = FastMCP("er-assistant-tools")

_WIKI = wikipediaapi.Wikipedia(
    user_agent="ER-Assistant/1.0 (hackathon project; educational use)",
    language="en",
)


@mcp.tool
def wikipedia_search(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    """Search Wikipedia for authoritative, general-knowledge information.

    Args:
        query: Search query (keywords work best, e.g. "CPR compression depth").
        max_results: Max number of articles to return (default 3).

    Returns:
        List of {title, summary, url} dicts. Empty list if nothing found.
    """
    try:
        with DDGS() as ddgs:
            hits = list(
                ddgs.text(
                    f"site:en.wikipedia.org {query}",
                    max_results=max_results * 2,
                )
            )

        results = []
        seen_titles = set()
        for hit in hits:
            url = hit.get("href", "")
            if "en.wikipedia.org/wiki/" not in url:
                continue
            title = url.rsplit("/wiki/", 1)[-1].replace("_", " ")
            if title in seen_titles:
                continue
            seen_titles.add(title)

            page = _WIKI.page(title)
            if not page.exists():
                continue

            results.append(
                {
                    "title": page.title,
                    "summary": page.summary[:800],
                    "url": page.fullurl,
                }
            )
            if len(results) >= max_results:
                break

        return results
    except Exception as e:
        print(f"[wikipedia_search error] {e}")
        return []


@mcp.tool
def web_search(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    """Search the live web via DuckDuckGo for recent news or updated information.
    Args:
        query: Search query.
        max_results: Max number of results to return (default 3).

    Returns:
        List of {title, snippet, url} dicts. Empty list if nothing found.
    """
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "title": h.get("title", ""),
                "snippet": h.get("body", "")[:500],
                "url": h.get("href", ""),
            }
            for h in hits
        ]
    except Exception as e:
        print(f"[web_search error] {e}")
        return []


# OpenAI function-calling schemas (derived from the tool docstrings)
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": (
                "Search Wikipedia for authoritative, general-knowledge "
                "information. Best for medical guidelines, definitions, "
                "procedures, and well-established facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max articles to return.",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the live web via DuckDuckGo for recent news or "
                "updated information. Use when the topic may have changed "
                "since the internal corpus was compiled."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results to return.",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
]


if __name__ == "__main__":
    # Run as a real MCP server over HTTP on port 8765
    mcp.run(transport="http", host="127.0.0.1", port=8765)
