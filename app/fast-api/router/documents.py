from fastapi import APIRouter, HTTPException
from schemas.documents import QueryRequest, QueryResponse, SourceMeta
from services.embedder import embed_prompt
from services.searcher import search_chunks
from services.reconstructor import build_context
from services.llm import generate_answer
router = APIRouter()

@router.post("/documents/query", response_model=QueryResponse)
def queryDocument(request: QueryRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Empty prompt")

    query_vector = embed_prompt(request.prompt)
    retrieved = search_chunks(query_vector, request.k)
    context = build_context(retrieved)
    answer = generate_answer(request.prompt, context)

    sources = [
        SourceMeta(
            source=c.get("source", "unknown"),
            page=int(c.get("page", 0)),
            score=c["score"],
        )
        for c in retrieved
    ]

    return QueryResponse(answer=answer, sources=sources)