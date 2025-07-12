import asyncio
from crawl4ai import AsyncWebCrawler
from typing import List
from scripts.content_chunker import chunk_text_tokenwise

async def load_and_chunk_content_async(urls: List[str], max_tokens: int = 500, overlap: int = 50) -> List[str]:
    all_chunks = []

    async with AsyncWebCrawler() as crawler:
        for url in urls:
            try:
                result = await crawler.arun(url=url)
                text = result.markdown.strip() if hasattr(result, "markdown") else ""
                if text:
                    chunks = chunk_text_tokenwise(text, max_tokens=max_tokens, overlap=overlap)
                    all_chunks.extend(chunks)
            except Exception as e:
                print(f"[ERROR] Failed to process {url}: {e}")

    return all_chunks

# Optional sync wrapper for Flask or testing
def load_and_chunk_content(urls: List[str], max_tokens: int = 500, overlap: int = 50) -> List[str]:
    return asyncio.run(load_and_chunk_content_async(urls, max_tokens, overlap))
