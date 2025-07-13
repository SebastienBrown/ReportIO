# scripts/content_chunker.py

import tiktoken
from typing import List

#langchain chunking fucntion 

def chunk_text_tokenwise(text: str, max_tokens: int = 500, overlap: int = 50, model: str = "gpt-3.5-turbo") -> List[str]:
    """
    Token-aware text chunking. Compatible with OpenAI models.
    """
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens).strip()
        if chunk_text:
            chunks.append(chunk_text)
        start += max_tokens - overlap

    return chunks
