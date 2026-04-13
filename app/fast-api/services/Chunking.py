import uuid
from collections import defaultdict
 
 
def text_splitter(chunk_size: int, overlap: int, pages: list) -> list:
    """
    chunk_size and overlap are in CHARACTERS.
 
    Returns a list of chunks. Each chunk has a `pages` field that is a list
    of 1-indexed page numbers the chunk overlaps — usually one page, but two
    or more when the chunk straddles a page boundary.
    """
    chunks = []
    step = chunk_size - overlap
 
    # Group pages by source document
    docs = defaultdict(list)
    for page in pages:
        docs[page["source"]].append(page)
 
    for source, doc_pages in docs.items():
        doc_pages.sort(key=lambda p: p["page"])
 
        # Merge all pages of this document into one big string, remembering
        # where each page starts so we can map chunk offsets back to pages.
        full_text = ""
        page_spans = []  # list of (start_char, end_char, page_number)
 
        for page in doc_pages:
            if len(page["text"]) < 100:
                continue
            start = len(full_text)
            full_text += page["text"] + " "
            end = len(full_text)
            page_spans.append((start, end, page["page"]))
 
        if not full_text:
            continue
 
        cursor = 0
        chunk_index = 0
 
        while cursor < len(full_text):
            end = min(cursor + chunk_size, len(full_text))
 
            # Don't cut mid-word
            if end < len(full_text):
                last_space = full_text.rfind(" ", cursor, end)
                if last_space != -1 and last_space > cursor:
                    end = last_space
 
            content = full_text[cursor:end].strip()
            if content:
                # A chunk [cursor, end) overlaps a page [p_start, p_end) iff
                # cursor < p_end AND end > p_start.
                pages_in_chunk = sorted({
                    pg for p_start, p_end, pg in page_spans
                    if cursor < p_end and end > p_start
                })
 
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "source": source,
                    "pages": pages_in_chunk,
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
    from pathlib import Path
 
    SCRIPT_DIR = Path(__file__).resolve().parent
    # Adjust the number of .parent calls so DATA_DIR points at ER-Assistant/data
    DATA_DIR = SCRIPT_DIR.parent.parent.parent / "data"
    CORPUS_PATH = DATA_DIR / "corpus.json"
    CHUNKS_PATH = DATA_DIR / "chunks.json"
 
    with CORPUS_PATH.open() as f:
        corpus = json.load(f)
 
    chunks = text_splitter(chunk_size=1000, overlap=200, pages=corpus)
    estimate_embedding_cost(chunks)
 
    with CHUNKS_PATH.open("w") as f:
        json.dump(chunks, f, indent=2)
    print(f"chunks.json saved to {CHUNKS_PATH}")