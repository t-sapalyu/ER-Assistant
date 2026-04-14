import faiss
from fastapi import HTTPException

index: faiss.Index | None = None
chunks: list[dict] = []

def load_index(index_path: str, chunks_path: str):
    global index, chunks

    index = faiss.read_index(index_path)

    import json
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

def search_chunks(query_vector, k: int):
    if index is None or not chunks:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded",
        )

    scores, indices = index.search(query_vector, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        chunk = dict(chunks[idx])
        chunk["score"] = float(score)
        results.append(chunk)

    return results