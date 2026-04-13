import uuid

def text_splitter(chunk_size: int, overlap: int, pages: list) -> list:
    """
    chunk_size and overlap are in CHARACTERS
    """
    chunks = []
    step = chunk_size - overlap

    from collections import defaultdict
    docs = defaultdict(list)
    for page in pages:
        docs[page["source"]].append(page)

    for source, doc_pages in docs.items():
        doc_pages.sort(key=lambda p: p["page"])

        # Merge all pages into one big text
        full_text = ""
        page_map = []
        for page in doc_pages:
            if len(page["text"]) < 100:
                continue
            page_map.append((len(full_text), page["page"]))
            full_text += page["text"] + " "

        cursor = 0
        chunk_index = 0

        while cursor < len(full_text):
            end = min(cursor + chunk_size, len(full_text))

            # Don't cut mid-word
            if end < len(full_text):
                last_space = full_text.rfind(" ", cursor, end)
                if last_space != -1:
                    end = last_space

            content = full_text[cursor:end].strip()

            if content:
                # Find which page this chunk is on
                page_num = 1
                for char_start, pg in page_map:
                    if char_start <= cursor:
                        page_num = pg

                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "source": source,
                    "page": page_num,
                    "chunk_index": chunk_index,
                    "text": content,
                    "char_count": len(content),
                })
                chunk_index += 1

            cursor += step

    return chunks


def estimate_embedding_cost(chunks: list) -> None:
    # Rough estimate: 1 token ≈ 4 characters
    total_tokens = sum(c["char_count"] for c in chunks) // 4
    cost = (total_tokens / 1_000_000) * 0.02
    print(f"Total chunks:   {len(chunks)}")
    print(f"Est. tokens:    {total_tokens:,}")
    print(f"Estimated cost: ${cost:.4f}")


if __name__ == "__main__":
    import json

    with open("ER-Assistant/data/corpus.json") as f:
        corpus = json.load(f)

    chunks = text_splitter(chunk_size=1000, overlap=200, pages=corpus)
    estimate_embedding_cost(chunks)

    with open("ER-Assistant/data/chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)

    print("chunks.json saved!")