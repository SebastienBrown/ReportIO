# scripts/seb/wrappers.py

from scripts.pipelines.text_pipeline.seb.search import google_search
import os
from dotenv import load_dotenv

load_dotenv(override=True)

API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")

def search_web_articles_seb(query: str, num_results: int = 10) -> list[dict]:
    raw_results = google_search(query, API_KEY, CSE_ID, num_results=num_results)

    return [
        {
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", "")
        }
        for item in raw_results
    ]




from scripts.pipelines.text_pipeline.seb.scraper import WebScrapingService
from scripts.pipelines.text_pipeline.content_chunker import chunk_text_tokenwise
import json

def load_and_chunk_content_seb(urls, max_tokens=500, overlap=50):
    service = WebScrapingService()
    scraped_chunks = []

    for url in urls:
        try:
            result = service.scrape(url)
            if result.success and result.content:
                chunks = chunk_text_tokenwise(result.content, max_tokens=max_tokens, overlap=overlap)
                scraped_chunks.extend(chunks)
            print(f"[DEBUG] Scraped: {url} → {result.word_count} words → {len(chunks)} chunks")

        except Exception as e:
            print(f"[ERROR] Failed to scrape {url}: {e}")
    debug_json = [{"chunk_id": i, "length": len(c), "preview": c[:100]} for i, c in enumerate(scraped_chunks)]
    print(json.dumps(debug_json, indent=2))
    return scraped_chunks


