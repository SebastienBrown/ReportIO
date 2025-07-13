# scripts/pipelines/video_pipeline/vector_store.py

from qdrant_client.http.models import PointStruct
from scripts.llm.embed import embed_llm
from scripts.llm.qdrant_client import client, init_qdrant_collection

COLLECTION_NAME = "video_chunks"

def init_vector_collection():
    init_qdrant_collection(COLLECTION_NAME)

def embed_and_upsert_chunks(chunks: list[dict]):
    """
    Embeds and stores video transcript chunks with metadata.
    Each chunk must include a 'chunk' key (text), plus metadata like video_id, start, url, title.
    """
    texts = [c["chunk"] for c in chunks]
    vectors = embed_llm.embed_documents(texts)

    points = [
        PointStruct(
            id=i,
            vector=vectors[i],
            payload=chunks[i]  # include full dict: text, start, video_id, etc.
        )
        for i in range(len(chunks))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)

def search_similar_chunks(query: str, top_k: int = 5) -> list[dict]:
    """
    Embeds a query and returns the top matching video chunks, including metadata (start time, title, etc.)
    """
    query_vector = embed_llm.embed_query(query)
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
    )
    return [hit.payload for hit in results]
