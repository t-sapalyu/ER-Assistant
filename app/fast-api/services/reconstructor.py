def build_context(retrieved: list[dict]) -> str:
    parts = []

    for chunk in retrieved:
        source = chunk.get("source", "unknown")
        page   = chunk.get("page", "?")
        text   = chunk.get("text", "")

        parts.append(f"[Source: {source}, p.{page}]\n{text}")

    return "\n\n".join(parts)