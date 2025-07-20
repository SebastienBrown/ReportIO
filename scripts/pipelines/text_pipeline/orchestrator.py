# scripts/orchestrator.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.pipelines.text_pipeline.snippet_scorer import score_snippets
from scripts.pipelines.text_pipeline.vector_store import embed_and_upsert_chunks, search_similar_chunks, init_vector_collection
from scripts.pipelines.text_pipeline.generation import generate_answer_from_context, generate_gpt_answer
from scripts.pipelines.text_pipeline.process_query import QueryTransformer
from scripts.pipelines.text_pipeline.process_query import generate_sections


USE_SEB_SEARCH = True

if USE_SEB_SEARCH:
    from scripts.pipelines.text_pipeline.seb.wrappers import search_web_articles_seb as search_web_articles
else:
    from scripts.pipelines.text_pipeline.search_module import search_web_articles


USE_SEB_SCRAPER_AND_CHUNKER = False

if USE_SEB_SCRAPER_AND_CHUNKER:
    from scripts.pipelines.text_pipeline.seb.wrappers import load_and_chunk_content_seb as load_and_chunk_content
else:
    from scripts.pipelines.text_pipeline.content_loader import load_and_chunk_content



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

        final_report=""

        #Include classifier from Seb's code
        logger("[DEBUG] Step 0: Processing query for enhanced search")
        oldQuery=query
        queryProcessor=QueryTransformer()
        query = queryProcessor.transform_query(query)
        logger(f"[DEBUG] Step 0 done → From {oldQuery} to {query}")

        logger("[DEBUG] Step 0.5: Extracting underlying prompts")
        topicList=generate_sections(query)
        logger(f"[DEBUG] Step 0.5 done → Topic list is {topicList}")
    
        for i, titleQuery in enumerate(topicList, start=1):
            compositeQuery=f"{query}:{titleQuery}"
            logger(f"[DEBUG] Step 1: Searching web articles for query={compositeQuery}")
            raw_articles = search_web_articles(compositeQuery, num_results=num_results)
            logger(f"[DEBUG] Step 1 done → Found {len(raw_articles)} articles")

            

            logger("[DEBUG] Step 2: Scoring snippets...")
            top_articles = score_snippets(compositeQuery, raw_articles, top_k=top_k)
            logger(f"[DEBUG] Step 2 done → Top {len(top_articles)} articles scored")



            logger("[DEBUG] Step 3: Extracting URLs...")
            top_urls = [article["url"] for article in top_articles]
            logger(f"[DEBUG] Step 3 done → URLs: {top_urls}")


            logger("[DEBUG] Step 4: Chunking content from URLs...")
            chunks = load_and_chunk_content(top_urls)
            logger(f"[DEBUG] Step 4 done → {len(chunks)} chunks loaded")

            #Include chunking from Seb's code (good practice)

            logger("[DEBUG] Step 5: Embedding and upserting chunks to Qdrant...")
            init_vector_collection()
            embed_and_upsert_chunks(chunks)
            logger("[DEBUG] Step 5 done")

            logger("[DEBUG] Step 6: Searching Qdrant for similar chunks...")
            retrieved_chunks = search_similar_chunks(query, top_k=top_k)
            logger(f"[DEBUG] Step 6 done → Retrieved {len(retrieved_chunks)} results")

            llm_answer = generate_answer_from_context(query, retrieved_chunks)
            final_report+=llm_answer

            logger("[DEBUG] Step 7: Final answer generated.")

        return {
            "llm_answer": final_report,
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


if __name__ == "__main__":
    test_query = "latest breakthroughs in clean energy 2025"
    result = run_orchestration_pipeline(test_query)
    import json
    print(json.dumps(result, indent=2))
