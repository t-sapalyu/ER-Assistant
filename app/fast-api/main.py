import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from router.documents import router as documents_router
from services.searcher import load_index

load_dotenv()

INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "D:/Study/Kaggle Week/project/data/my_index.faiss")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "D:/Study/Kaggle Week/project/data/chunks.json")

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)


@app.on_event("startup")
def startup():
    load_index(INDEX_PATH, CHUNKS_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}