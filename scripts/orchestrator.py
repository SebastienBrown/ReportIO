# scripts/orchestrator.py

from scripts.search_module import search_web_articles
from scripts.snippet_scorer import score_snippets
from scripts.content_loader import load_and_chunk_content
from scripts.vector_store import embed_and_upsert_chunks, search_similar_chunks, init_qdrant_collection
from scripts.generation import generate_answer_from_context, generate_gpt_answer
from scripts.query_classifier import classify_query



def run_orchestration_pipeline(query: str, num_results: int = 10, top_k: int = 5, logger=print):
    try:
        #classifier 
        # logger("[DEBUG] Step 0: Classifying query...")
        # result = classify_query(query)
        # logger(f"[DEBUG] Step 0 done → Route: {result['route']} (Confidence: {result['confidence']:.2f})")

        # if result["route"] == "GPT":
        #     logger("[DEBUG] GPT route selected → Skipping RAG pipeline")
        #     llm_answer = generate_gpt_answer(query)  # or whatever fallback you use
        #     return {
        #         "llm_answer": llm_answer,
        #         "status": "ok",
        #         "top_snippets": [],
        #         "retrieved_chunks": [],
        #         "route": "GPT",
        #         "confidence": result["confidence"]
        #     }

        logger("[DEBUG] Step 1: Searching web articles...")
        raw_articles = search_web_articles(query, num_results=num_results)
        logger(f"[DEBUG] Step 1 done → Found {len(raw_articles)} articles")

        logger("[DEBUG] Step 2: Scoring snippets...")
        top_articles = score_snippets(query, raw_articles, top_k=top_k)
        logger(f"[DEBUG] Step 2 done → Top {len(top_articles)} articles scored")

        logger("[DEBUG] Step 3: Extracting URLs...")
        top_urls = [article["url"] for article in top_articles]
        logger(f"[DEBUG] Step 3 done → URLs: {top_urls}")

        logger("[DEBUG] Step 4: Chunking content from URLs...")
        chunks = load_and_chunk_content(top_urls)
        logger(f"[DEBUG] Step 4 done → {len(chunks)} chunks loaded")

        logger("[DEBUG] Step 5: Embedding and upserting chunks to Qdrant...")
        init_qdrant_collection()
        embed_and_upsert_chunks(chunks)
        logger("[DEBUG] Step 5 done")

        logger("[DEBUG] Step 6: Searching Qdrant for similar chunks...")
        retrieved_chunks = search_similar_chunks(query, top_k=top_k)
        logger(f"[DEBUG] Step 6 done → Retrieved {len(retrieved_chunks)} results")

        llm_answer = generate_answer_from_context(query, retrieved_chunks)

        logger("[DEBUG] Step 7: Final answer generated.")

        return {
            "llm_answer": llm_answer,
            "status": "ok",
            "top_snippets": top_articles,
            "retrieved_chunks": retrieved_chunks,
        }

    except Exception as e:
        logger(f"[ERROR] Exception during orchestration: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }
