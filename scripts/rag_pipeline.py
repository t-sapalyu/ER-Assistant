import json
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
import uuid
from cost_tracker import track_cost
import logging

# import spacy
# nlp = spacy.load("en_core_web_sm")  # pip install spacy && python -m spacy download en_core_web_sm


# ---------------------------
# Setup
# ---------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

INDEX_PATH = DATA_DIR / "my_index.faiss"
CHUNKS_PATH = DATA_DIR / "chunks.json"
CHUNK_LOG = LOG_DIR / "chunk_log.txt"

# ---------------------------
# Config
# ---------------------------
CHUNK_SIZE = 600
OVERLAP = 150
MIN_TEXT_LENGTH = 100

logging.basicConfig(
    filename=CHUNK_LOG,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


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


def sentencize(text: str) -> list[str]:
    """Split text into sentences using spaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


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
def chunk_page_old(text: str, source: str, page_number: int):
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
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "source": source,
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                }
            )
            chunk_index += 1

        cursor += CHUNK_SIZE - OVERLAP

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

        page_chunks = chunk_page_old(text, source, page)
        all_chunks.extend(page_chunks)

    return all_chunks


# ---------------------------
# Step 3 — Embed (batched)
# ---------------------------
def get_embeddings(texts):
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    track_cost(response, True)
    return [d.embedding for d in response.data]


def embed_chunks(chunks, batch_size=50):
    texts = [c["text"] for c in chunks]
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings.extend(get_embeddings(batch))

    return embeddings


# ---------------------------
# Step 4 — Build + Save FAISS
# ---------------------------
def build_index(chunks):
    print("Embedding chunks...")

    vectors = np.array(embed_chunks(chunks)).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # Save
    faiss.write_index(index, str(INDEX_PATH))

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print("Index + chunks saved")

    return index


def load_index():
    index = faiss.read_index(str(INDEX_PATH))

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks


# ---------------------------
# Step 5 — Retrieve
# ---------------------------
def retrieve(query, index, chunks, k=20, fetch_k=20, rerank=True):
    """Retrieve top-k chunks. With rerank=True, fetches fetch_k from FAISS then
    reranks with a cross-encoder to return the best k."""
    query_vec = np.array(get_embeddings([query])).astype("float32")
    faiss.normalize_L2(query_vec)

    # Stage 1: FAISS retrieval (over-fetch)
    # Clamp to index size so FAISS never pads results with -1 or repeats
    n_fetch = min(fetch_k if rerank else k, index.ntotal)
    if n_fetch == 0:
        return []
    scores, indices = index.search(query_vec, n_fetch)

    candidates = []
    for i, idx in enumerate(indices[0]):
        # FAISS pads with -1 when k > ntotal; skip invalid entries
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = dict(chunks[idx])
        chunk["faiss_score"] = float(scores[0][i])
        candidates.append(chunk)

    if not rerank:
        for c in candidates:
            c["score"] = c["faiss_score"]
        return candidates[:k]

    # Stage 2: cross-encoder reranking
    # reranker = get_reranker()
    # pairs = [(query, c["text"]) for c in candidates]
    # rerank_scores = reranker.predict(pairs)

    # for c, s in zip(candidates, rerank_scores):
    #     c["rerank_score"] = float(s)
    #     c["score"] = float(s)  # overwrite so downstream code uses rerank score

    # candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    for c in candidates:
        c["score"] = c.get("rerank_score", c["faiss_score"])

    # LOGGING 
    logger.info("---- RETRIEVED CHUNKS ----")
    logger.info(f"query: {query}")
    logger.info(f"total_candidates: {len(candidates)}")

    for i, chunk in enumerate(candidates):
        logger.info(f"[Rank {i+1}] score: {chunk.get('score')}")
        logger.info(f"id: {chunk.get('chunk_id')}")        
        logger.info(f"source: {chunk.get('source')}")
        logger.info(f"pages: {chunk.get('page')}")
        logger.info(f"text preview: {chunk.get('text', '')[:200]}")
        logger.info("-" * 50)
    return candidates[:k]


# ---------------------------
# Step 6 — Generate answer
# ---------------------------
def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(
        [
            f"\n\n[Source: {c['source']} | Page {c['page']}]\n{c['text']}"
            for c in retrieved_chunks
        ]
    )

    system_prompt = """You are a helpful first-aid assistant.

    Your job is to answer the question using the provided context. Do NOT invent facts.

    Rules:
    - Read ALL context chunks carefully before deciding you cannot answer.
    - Use reasoning and inference from the context — the answer may not use the exact same words as the question.
      For example: if asked about a pregnant person and the context describes chest thrusts as an alternative
      to abdominal thrusts for certain patients, that IS the answer.
    - Combine information across multiple chunks when needed to form a complete answer.
    - Always cite your source using: (Source: <filename>, Page <number>).
    - Only say 'I don't have enough information to answer this.' if, after reading all chunks,
      there is genuinely no information that could reasonably answer the question.
    - If the question is yes/no, answer with only 'Yes' or 'No'.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0,
        max_tokens=500,
    )
    track_cost(response)
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

    print(f"Saved {len(chunks)} chunks to {CHUNKS_PATH}")

    # 3–4. Embed + Index (run once)
    index = build_index(chunks)

    return index, chunks


# ---------------------------
# Query
# ---------------------------
def ask(query, index, chunks):
    retrieved = retrieve(query, index, chunks)
    answer = generate_answer(query, retrieved)
    return answer


# ---------------------------
# Step 6b — Scope validation
# ---------------------------
CLASSIFY_ADD_MODE_PROMPT = (
    "The user is in 'Add Knowledge' mode, meaning they are supposed to be providing "
    "new factual content about first aid or emergency medicine to store in a knowledge base.\n\n"
    "Read their message and decide what they actually intend:\n"
    "  CONTENT  — the message genuinely looks like knowledge to store: "
    "it reads as informational, instructional, or reference text about a medical/first-aid topic.\n"
    "  QUESTION — the message is clearly asking for information or an answer.\n"
    "  EXIT     — everything else: the user seems to have lost interest, is confused, "
    "is saying something unrelated, is expressing an emotion, is chatting casually, "
    "or is otherwise not providing content and not asking a question. "
    "Use your judgment — if it does not clearly fit CONTENT or QUESTION, choose EXIT.\n\n"
    "Reply with ONLY one word: CONTENT, QUESTION, or EXIT."
)


def classify_add_mode_message(message: str) -> str:
    """Return 'content', 'question', or 'exit'. Uses LLM intent understanding, not keywords."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": CLASSIFY_ADD_MODE_PROMPT},
            {"role": "user", "content": message},
        ],
        temperature=0,
        max_tokens=10,
    )
    label = response.choices[0].message.content.strip().upper()
    if "CONTENT" in label:
        return "content"
    if "QUESTION" in label:
        return "question"
    return "exit"


SCOPE_SYSTEM_PROMPT = (
    "You are a strict content gatekeeper for an ER / First-Aid assistant. "
    "Decide whether the supplied text is relevant to any of these topics: "
    "first aid, CPR, AED, choking, bleeding control, burns, fractures, "
    "stroke, cardiac arrest, emergency medicine, tourniquets, wound care, "
    "shock, poisoning, allergic reactions, or similar emergency / pre-hospital care.\n\n"
    "Reply with EXACTLY one line in this format:\n"
    "RELEVANT: <yes|no>\n"
    "REASON: <one short sentence>\n\n"
    "Do not add anything else."
)


def check_content_scope(text: str) -> tuple[bool, str]:
    """Return (is_relevant, reason). Uses the first 1500 chars as a sample."""
    sample = text.strip()[:1500]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SCOPE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Text to evaluate:\n\n{sample}"},
        ],
        temperature=0,
        max_tokens=80,
    )
    reply = response.choices[0].message.content.strip()
    relevant = "yes" in reply.lower().split("relevant:")[-1].split("\n")[0].lower()
    reason_line = [l for l in reply.splitlines() if l.upper().startswith("REASON:")]
    reason = reason_line[0].split(":", 1)[-1].strip() if reason_line else "Out of scope."
    return relevant, reason


# ---------------------------
# Step 7 — Add to database (Agentic AI Step 3)
# ---------------------------
def add_to_database(text: str, source_name: str, index, chunks: list) -> dict:
    """Chunk, embed, and add new text to the live FAISS index + chunks list.

    Args:
        text: Raw text to add.
        source_name: Label used as the 'source' field in each chunk (e.g. filename).
        index: Live faiss.Index object (mutated in place).
        chunks: Live chunks list (mutated in place).

    Returns:
        Dict with keys: added (int), source (str), message (str).
    """
    new_chunks = chunk_page_old(text, source_name, 0)
    if not new_chunks:
        return {"added": 0, "source": source_name, "message": "Text too short to index."}

    embeddings = embed_chunks(new_chunks)
    vectors = np.array(embeddings).astype("float32")
    faiss.normalize_L2(vectors)
    index.add(vectors)
    chunks.extend(new_chunks)

    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    msg = f"Added {len(new_chunks)} chunk(s) from '{source_name}' to the database."
    return {"added": len(new_chunks), "source": source_name, "message": msg}


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Run once (build index)
    index, chunks = run_pipeline()
    # run_pipeline()
    # Example query
    query = "What is the depth of chest compressions in CPR?"

    answer = ask(query, index, chunks)

    print("\n Answer:\n")
    print(answer)

