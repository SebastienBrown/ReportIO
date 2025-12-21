# scripts/pipelines/text_pipeline/vector_store.py

from qdrant_client.http.models import PointStruct
from scripts.llm.embed import embed_llm
from scripts.llm.qdrant_client_init import client, init_qdrant_collection
import time
import uuid

def init_vector_collection(COLLECTION_NAME):
    init_qdrant_collection(COLLECTION_NAME)
    


def embed_and_upsert_chunks(chunks: list[dict],COLLECTION_NAME:uuid.UUID, batch_size: int = 40, delay: float = 1.5):
    
    """
    Embed chunks in batches and upsert to Qdrant.
    """

    all_vectors = []
    
    # Calculate and print total token estimate before embedding
    try:
        import tiktoken
        from collections import defaultdict
        enc = tiktoken.get_encoding("cl100k_base") # standard for ada-002
        
        url_stats = defaultdict(lambda: {"tokens": 0, "chunks": 0})
        total_tokens = 0
        
        for c in chunks:
            tokens = len(enc.encode(c["text"]))
            url = c.get("url", "Unknown Source")
            url_stats[url]["tokens"] += tokens
            url_stats[url]["chunks"] += 1
            total_tokens += tokens

        print(f"\n[DEBUG] PRE-EMBEDDING DIAGNOSTICS (Breakdown by Source):")
        for url, stats in url_stats.items():
            print(f"  - {url[:80]}{'...' if len(url) > 80 else ''}")
            print(f"    -> {stats['chunks']} chunks | {stats['tokens']} tokens")
            
        print(f"\n[DEBUG] TOTAL SUMMARY:")
        print(f"  - Total Chunks: {len(chunks)}")
        print(f"  - Est. Total Tokens: {total_tokens}")
        print(f"  - Est. Cost (ada-002): ${ (total_tokens / 1_000_000) * 0.10:.6f}") # Approx cost
        print(f"  - TPM Demand: {total_tokens} tokens will be sent in { (len(chunks) + batch_size - 1) // batch_size } batches\n")
    except Exception as te:
        print(f"[DEBUG] Could not calculate token estimate: {te}")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk["text"] for chunk in batch]

        try:
            batch_vectors = embed_llm.embed_documents(texts)
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
        PointStruct(
            id=i,
            vector=all_vectors[i],
            payload={
                "text": chunks[i]["text"],
                "url": chunks[i]["url"],
                "query_id": chunks[i]["query_id"],
                "timestamp": chunks[i]["timestamp"],
                "query": chunks[i]["query"]
            }
        )
        for i in range(len(all_vectors))
    ]

    if not points:
        print("[Vector Store] No points to upsert to Qdrant.")
        return

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[Vector Store] Upserted {len(points)} points to Qdrant.")


def search_similar_chunks(query: str,COLLECTION_NAME:uuid.UUID, top_k: int = 5) -> list[dict[str, str]]:
    """
    Embeds a query and returns the top matching chunk texts (no metadata).
    """
    query_vector = embed_llm.embed_query(query)
    results = client.search(    
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
    )
    return [
        {
            "text": hit.payload.get("text", ""),
            "url": hit.payload.get("url", "")
        }
        for hit in results
    ]
