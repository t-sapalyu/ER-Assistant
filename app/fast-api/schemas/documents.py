from pydantic import BaseModel

class QueryRequest(BaseModel):
    prompt: str
    k: int = 5

class SourceMeta(BaseModel):
    source: str
    page: int
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceMeta]