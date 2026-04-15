import json
import os
from pathlib import Path
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError
 
# ---------------------------------------------------------------------------

# Setup

# ---------------------------------------------------------------------------

load_dotenv()

client = OpenAI()
 
# Paths are resolved relative to this script, not the current working directory.

# Script lives at: ER-Assistant/app/fast-api/services/Embbeding.py

# Data lives at:   ER-Assistant/data/

# So we go up three levels from the script to reach ER-Assistant/, then into data/.

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"

CHUNKS_PATH = DATA_DIR / "chunks.json"

INDEX_PATH = DATA_DIR / "my_index.faiss"
 
EMBEDDING_MODEL = "text-embedding-3-small"

MAX_CHARS_PER_CHUNK = 30000  # ~8000 tokens, safely under the 8191 model limit
 
 
# ---------------------------------------------------------------------------

# Embedding

# ---------------------------------------------------------------------------

def embed_chunks(chunks: list, batch_size: int = 100) -> np.ndarray:

    """Embed a list of chunks in batches, returning a float32 ndarray."""

    # Sanity check: flag any chunk that's too long for the model.

    oversized = [i for i, c in enumerate(chunks) if len(c["text"]) > MAX_CHARS_PER_CHUNK]

    if oversized:

        raise ValueError(

            f"{len(oversized)} chunk(s) exceed {MAX_CHARS_PER_CHUNK} chars "

            f"(first offender: index {oversized[0]}). Re-chunk before embedding."

        )
 
    all_embeddings: list[list[float]] = []
 
    for i in range(0, len(chunks), batch_size):

        batch = chunks[i : i + batch_size]

        texts = [c["text"] for c in batch]
 
        try:

            response = client.embeddings.create(

                model=EMBEDDING_MODEL,

                input=texts,

            )

        except (RateLimitError, APIError) as e:

            raise RuntimeError(

                f"OpenAI API failed on batch starting at index {i}: {e}"

            ) from e
 
        all_embeddings.extend(e.embedding for e in response.data)

        print(f"Embedded {i + len(batch)}/{len(chunks)} chunks...")
 
    return np.array(all_embeddings, dtype="float32")
 
 
# ---------------------------------------------------------------------------

# Indexing

# ---------------------------------------------------------------------------

def build_and_save_index(embeddings: np.ndarray, index_path: Path = INDEX_PATH) -> None:

    """Build a cosine-similarity FAISS index and save it to disk.
 
    OpenAI embeddings are designed for cosine similarity, not L2 distance.

    FAISS doesn't have a direct cosine index, so we L2-normalize the vectors

    and use inner product — the two are mathematically equivalent on unit vectors.
 
    IMPORTANT: the query side must also L2-normalize query embeddings before

    calling index.search(), or retrieval will be wrong.

    """

    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    index.add(embeddings)
 
    index_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))

    print(f"FAISS index saved to {index_path}")
 
 
# ---------------------------------------------------------------------------

# Entry point

# ---------------------------------------------------------------------------

if __name__ == "__main__":

    if not os.getenv("OPENAI_API_KEY"):

        raise RuntimeError(

            "OPENAI_API_KEY is not set. Add it to your .env file before running."

        )
 
    if not CHUNKS_PATH.exists():

        raise FileNotFoundError(

            f"Could not find chunks file at {CHUNKS_PATH}. "

            f"Expected project layout: ER-Assistant/data/chunks.json"

        )
 
    with CHUNKS_PATH.open() as f:

        chunks = json.load(f)
 
    print(f"Total chunks to embed: {len(chunks)}")

    embeddings = embed_chunks(chunks)

    build_and_save_index(embeddings)

    print("Done! Share my_index.faiss and chunks.json with your team.")
 