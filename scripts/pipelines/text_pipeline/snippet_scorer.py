import numpy as np
from typing import List, Dict
from scripts.llm.embed import embed_llm  # LangChain embedding model

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def score_snippets(query: str, articles: List[Dict], top_k: int = 5) -> List[Dict]:
    query_embedding = embed_llm.embed_query(query)

    # Prepare batch of texts to embed
    texts = [f"{article['title']} {article['snippet']}" for article in articles]
    embeddings = embed_llm.embed_documents(texts)  # List of vectors

    # Score and store results (without storing embeddings)
    scored_articles = []
    for article, embedding in zip(articles, embeddings):
        score = cosine_similarity(np.array(query_embedding), np.array(embedding))
        scored_articles.append({
            "title": article["title"],
            "url": article["url"],
            "snippet": article["snippet"],
            "score": score
        })

    # Return top K by score
    return sorted(scored_articles, key=lambda x: x["score"], reverse=True)[:top_k]