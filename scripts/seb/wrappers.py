# scripts/seb/wrappers.py

from scripts.seb.search import google_search
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




from scripts.seb.scraper import WebScrapingService
from scripts.content_chunker import chunk_text_tokenwise

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
    

    return scraped_chunks


