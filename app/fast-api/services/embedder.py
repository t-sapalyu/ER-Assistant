import time
import os
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError
from fastapi import HTTPException
load_dotenv()

INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "D:/Study/Kaggle Week/project/data/my_index.faiss")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "D:/Study/Kaggle Week/project/data/chunks.json")
client = OpenAI()

def embed_prompt(text: str) -> np.ndarray:
    for attempt in range(3):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            vector = np.array(
                response.data[0].embedding, dtype="float32"
            ).reshape(1, -1)

            faiss.normalize_L2(vector)
            return vector

        except RateLimitError:
            time.sleep(2 ** attempt)

        except APIError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Embedding API error: {exc}",
            )

    raise HTTPException(status_code=429, detail="Embedding rate limit exceeded")