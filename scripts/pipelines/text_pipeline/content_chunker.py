# scripts/content_chunker.py

import tiktoken
from typing import List
import uuid
from datetime import datetime

#langchain chunking fucntion 

def chunk_text_tokenwise(text: str, url:str, query:str,COLLECTION_NAME:uuid.UUID, max_tokens: int = 500, overlap: int = 50, model: str = "gpt-3.5-turbo") -> List[dict[str, str]]:
    """
    Token-aware text chunking. Compatible with OpenAI models.
    """

    query_id=COLLECTION_NAME,

    # Current timestamp
    timestamp = datetime.utcnow().isoformat()

    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens).strip()
        if chunk_text:
            chunks.append({
                "query_id": query_id,
                "query": query,
                "timestamp": timestamp,
                "text": chunk_text,
                "url": url
            })
        start += max_tokens - overlap

    return chunks
