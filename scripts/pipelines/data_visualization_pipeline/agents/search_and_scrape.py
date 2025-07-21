import asyncio
from crawl4ai import AsyncWebCrawler
from scripts.pipelines.text_pipeline.seb.wrappers import search_web_articles_google_API
from typing import List


async def scrape_urls_async(urls: List[str]) -> List[str]:
    """Scrape URLs and return raw text content without chunking"""
    all_content = []
    
    async with AsyncWebCrawler() as crawler:
        for url in urls:
            try:
                result = await crawler.arun(url=url)
                text = result.markdown.strip() if hasattr(result, "markdown") else ""
                if text:
                    all_content.append(text)
            except Exception as e:
                print(f"[ERROR] Failed to process {url}: {e}")
    
    return all_content

def scrape_urls(urls: List[str]) -> List[str]:
    """Sync wrapper for scraping URLs"""
    return asyncio.run(scrape_urls_async(urls))

def search_and_scrape(state: dict) -> dict:
    print("[Search+Scrape] Running...")
    
    subqueries = state.get("rephrased_queries", [])
    all_urls = []
    for q in subqueries:
        results = search_web_articles_google_API(q)
        urls = [r["url"] for r in results if r.get("url")] # Limit to 10 URLs per query
        all_urls.extend(urls)
    
    # Optional: remove duplicates
    all_urls = list(set(all_urls))[:10]
    
    print(f"[Search+Scrape] Scraping {len(all_urls)} URLs...")
    # Use the new scraper without chunking
    full_contents = scrape_urls(all_urls)
    
    state["urls"] = all_urls
    state["raw_documents"] = full_contents
    return state