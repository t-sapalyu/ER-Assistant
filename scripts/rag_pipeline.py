import json
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
import uuid

# ---------------------------
# Setup
# ---------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

INDEX_PATH = DATA_DIR / "my_index.faiss"
CHUNKS_PATH = DATA_DIR / "chunks.json"

# ---------------------------
# Config
# ---------------------------
CHUNK_SIZE = 600
OVERLAP = 150
MIN_TEXT_LENGTH = 100

# ---------------------------
# Step 1 — Load corpus
# ---------------------------
def load_corpus():
    with open(f"{DATA_DIR}/corpus.json", "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------
# Helper: clean text
# ---------------------------
def clean_text(text: str) -> str:
    text = text.strip()

    # Remove excessive whitespace
    text = " ".join(text.split())

    return text


# ---------------------------
# Helper: detect noisy pages
# ---------------------------
def is_noisy_page(text: str) -> bool:
    if len(text) < MIN_TEXT_LENGTH:
        return True

    if "table of contents" in text.lower():
        return True

    # Too many dots (typical TOC pattern)
    if text.count("...") > 5:
        return True

    return False

# ---------------------------
# Helper: smart split (avoid cutting sentences)
# ---------------------------
def find_split_point(text: str, start: int, end: int) -> int:
    # Prefer newline
    newline_pos = text.rfind("\n", start, end)

    # Then sentence end
    period_pos = text.rfind(".", start, end)

    split_pos = max(newline_pos, period_pos)

    if split_pos > start:
        return split_pos + 1

    return end

# ---------------------------
# Core: chunk per page
# ---------------------------
def chunk_page(text: str, source: str, page_number: int):
    chunks = []
    cursor = 0
    chunk_index = 0

    text = clean_text(text)

    while cursor < len(text):
        end = min(cursor + CHUNK_SIZE, len(text))

        # Adjust split point
        if end < len(text):
            end = find_split_point(text, cursor, end)

        chunk_text = text[cursor:end].strip()

        if chunk_text:
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "source": source,
                "page": page_number,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "char_count": len(chunk_text),
            })
            chunk_index += 1

        cursor += (CHUNK_SIZE - OVERLAP)

    return chunks


# ---------------------------
# Main chunking function
# ---------------------------
def create_chunks(corpus: list):
    all_chunks = []

    for entry in corpus:
        text = entry["text"]
        source = entry["source"]
        page = entry["page"]

        if is_noisy_page(text):
            continue

        page_chunks = chunk_page(text, source, page)
        all_chunks.extend(page_chunks)

    return all_chunks

# ---------------------------
# Step 3 — Embed (batched)
# ---------------------------
def get_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in response.data]


def embed_chunks(chunks, batch_size=50):
    texts = [c["text"] for c in chunks]
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings.extend(get_embeddings(batch))

    return embeddings


# ---------------------------
# Step 4 — Build + Save FAISS
# ---------------------------
def build_index(chunks):
    print("🔹 Embedding chunks...")

    vectors = np.array(embed_chunks(chunks)).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # Save
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print("✅ Index + chunks saved")

    return index


def load_index():
    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks


# ---------------------------
# Step 5 — Retrieve
# ---------------------------
def retrieve(query, index, chunks, k=5):
    query_vec = np.array(get_embeddings([query])).astype("float32")
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, k)

    results = []
    for i, idx in enumerate(indices[0]):
        chunk = dict(chunks[idx])
        chunk["score"] = float(scores[0][i])
        results.append(chunk)

    return results


# ---------------------------
# Step 6 — Generate answer
# ---------------------------
def generate_answer(query, retrieved_chunks):
    context = "\n\n".join([
        f"[Source: {c['source']} p.{c['page']}]\n{c['text']}"
        for c in retrieved_chunks
    ])

    prompt = f"""
You are an Emergency Responder Assistant.

Answer ONLY using the context below.
If the answer is not in the context, say:
"I don't have information about this."

Always include source citations.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300
    )

    return response.choices[0].message.content


# ---------------------------
# Full Pipeline
# ---------------------------
def run_pipeline():
    # 1. Load
    corpus = load_corpus()

    # 2. Chunk
    chunks = create_chunks(corpus)

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"✅ Saved {len(chunks)} chunks to {CHUNKS_PATH}")

    # 3–4. Embed + Index (run once)
    #index = build_index(chunks)

    #return index, chunks


# ---------------------------
# Query
# ---------------------------
def ask(query, index, chunks):
    retrieved = retrieve(query, index, chunks, k=5)
    answer = generate_answer(query, retrieved)
    return answer


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Run once (build index)
    #index, chunks = run_pipeline()
    run_pipeline()
    # Example query
    #query = "What is the depth of chest compressions in CPR?"

    #answer = ask(query, index, chunks)

    #print("\n Answer:\n")
    #print(answer)