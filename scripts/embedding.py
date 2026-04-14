import numpy as np

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings(texts):
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in response.data]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example: compare 4 sentences
sentences = [
    "The cat sat on the mat",
    "A kitten rested on the rug",
    "Stock prices rose sharply today",
    "The dog lay on the carpet"
]
embeddings = get_embeddings(sentences)

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"{sentences[i][:20]:20s} vs {sentences[j][:20]:20s} → {sim:.3f}")