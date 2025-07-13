# scripts/pipelines/video_pipeline/orchestrator.py

from asyncio.log import logger
from scripts.pipelines.video_pipeline.video_search import search_youtube_videos
from scripts.pipelines.video_pipeline.transcript_chunker import fetch_and_chunk_transcripts
from scripts.pipelines.video_pipeline.vector_store import (
    init_vector_collection,
    embed_and_upsert_chunks,
    search_similar_chunks,
)
from scripts.pipelines.video_pipeline.generation import group_and_summarize_video_chunks


def run_video_pipeline(query: str, top_k: int = 5, logger=print):
    logger("[DEBUG] Step 1: Searching YouTube videos...")
    search_results = search_youtube_videos(query)
    logger(f"[DEBUG] Step 1 done → Found {len(search_results)} videos.")

    logger("[DEBUG] Step 2–3: Fetching and chunking transcripts...")
    all_chunks = fetch_and_chunk_transcripts(search_results, logger=logger)
    logger(f"[DEBUG] Step 3 done → {len(all_chunks)} chunks created.")

    logger("[DEBUG] Step 4: Initializing and embedding into Qdrant...")
    init_vector_collection()
    embed_and_upsert_chunks(all_chunks)
    logger("[DEBUG] Step 4 done → Chunks upserted.")

    logger("[DEBUG] Step 5: Retrieving similar chunks...")
    retrieved_chunks = search_similar_chunks(query, top_k=top_k)
    logger(f"[DEBUG] Step 5 done → Retrieved {len(retrieved_chunks)} chunks.")

    if retrieved_chunks:
            logger(repr(retrieved_chunks[0]))
    else:
            logger("None retrieved.")

    
    grouped_videos = group_and_summarize_video_chunks(retrieved_chunks)
    logger(f"[DEBUG] Step 6 done → Grouped videos and summarized key moments")

    logger("[DEBUG] Sample retrieved video chunk:")
  
    return {
        "status": "ok",
        "video_results": search_results,
        "retrieved_chunks": grouped_videos,
        # "llm_answer": llm_answer,
    }

if __name__ == "__main__":
    query = "Attention is all you need"
    result = run_video_pipeline(query)

    print("\n[DEBUG] Final retrieved chunks:")
    for chunk in result["retrieved_chunks"][:3]:
        print(chunk)
