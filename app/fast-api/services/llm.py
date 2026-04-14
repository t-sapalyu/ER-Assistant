import time
from openai import OpenAI, RateLimitError, APIError
from fastapi import HTTPException



def get_client(api_key: str):
    return OpenAI(api_key=api_key)

def generate_answer(query: str, context: str) -> str:
    system_prompt = (
        "You are a helpful assistant. Answer ONLY using the provided context. "
        "If missing, say you don't know. Always cite sources."
    )

    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
                max_tokens=500,
            )
            return response.choices[0].message.content

        except RateLimitError:
            time.sleep(2 ** attempt)

        except APIError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Chat API error: {exc}",
            )

    raise HTTPException(status_code=429, detail="Chat rate limit exceeded")