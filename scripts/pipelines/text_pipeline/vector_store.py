# scripts/pipelines/text_pipeline/vector_store.py

from qdrant_client.http.models import PointStruct
from scripts.llm.embed import embed_llm
from scripts.llm.qdrant_client import client, init_qdrant_collection
import time

COLLECTION_NAME = "text_chunks"
def init_vector_collection():
    init_qdrant_collection(COLLECTION_NAME)


def embed_and_upsert_chunks(chunks: list[str], batch_size: int = 40, delay: float = 1.5):
 
    """
    Embed chunks in batches and upsert to Qdrant.
    """
    all_vectors = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            batch_vectors = embed_llm.embed_documents(batch)
            all_vectors.extend(batch_vectors)
            print(f"[Vector Store] Embedded batch {i // batch_size + 1}")
        except Exception as e:
            print(f"[Vector Store] Failed to embed batch {i}: {e}")
            raise
        time.sleep(delay)  # prevent hitting Azure rate limits

    if not all_vectors:
        print("[Vector Store] Embedding returned no vectors.")
        return

    points = [
        PointStruct(id=i, vector=all_vectors[i], payload={"text": chunks[i]})
        for i in range(len(all_vectors))
    ]

    if not points:
        print("[Vector Store] No points to upsert to Qdrant.")
        return

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[Vector Store] Upserted {len(points)} points to Qdrant.")


def search_similar_chunks(query: str, top_k: int = 5) -> list[str]:
    """
    Embeds a query and returns the top matching chunk texts (no metadata).
    """
    query_vector = embed_llm.embed_query(query)
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
    )
    return [hit.payload["text"] for hit in results]
