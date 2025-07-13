# scripts/pipelines/text_pipeline/vector_store.py

from qdrant_client.http.models import PointStruct
from scripts.llm.embed import embed_llm
from scripts.llm.qdrant_client import client, init_qdrant_collection

COLLECTION_NAME = "text_chunks"
def init_vector_collection():
    init_qdrant_collection(COLLECTION_NAME)


def embed_and_upsert_chunks(chunks: list[str]):
 
    vectors = embed_llm.embed_documents(chunks)
    points = [
        PointStruct(id=i, vector=vectors[i], payload={"text": chunks[i]})
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)


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
