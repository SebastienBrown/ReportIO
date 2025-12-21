import asyncio
from crawl4ai import AsyncWebCrawler
from typing import List
from scripts.pipelines.text_pipeline.content_chunker import chunk_text_tokenwise
import uuid
import re
from datetime import datetime

def strip_urls(text: str, strip_titles: bool = False) -> str:
    """
    Strips the URL part from Markdown links (e.g., [text](url) -> [text])
    or removes the entire link construct if strip_titles is True.
    Also removes raw http/https/www URLs from the text.
    """
    if not text:
        return ""
    
    if strip_titles:
        # 1. Strip the entire Markdown link: [Title](https://...) -> ""
        text = re.sub(r'\[([^\]]+)\]\((https?://[^\s\)]+)\)', '', text)
    else:
        # 1. Strip only the URL part: [Title](https://...) -> [Title]
        text = re.sub(r'\[([^\]]+)\]\((https?://[^\s\)]+)\)', r'[\1]', text)
    
    # 2. Strip raw URLs: https://... or http://... or www...
    text = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+', '', text)
    
    # 3. Clean up double spaces but PRESERVE newlines
    # [ \t]+ matches one or more spaces or tabs
    text = re.sub(r'[ \t]+', ' ', text)
    # 4. Remove triple+ newlines to keep it compact but readable
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

async def load_and_chunk_content_async(urls: List[str],query:str,COLLECTION_NAME: uuid.UUID, max_tokens: int = 500, overlap: int = 50, strip_titles: bool = False) -> List[dict[str, str]]:
    all_chunks = []
    debug_file = "debug_crawled_content.txt"
    
    # Clear the debug file for a new run
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write(f"DEBUG CRAWL LOG - {datetime.now().isoformat()}\n")
        f.write("="*80 + "\n\n")

    async with AsyncWebCrawler() as crawler:
        for url in urls:
            try:
                result = await crawler.arun(url=url)
                raw_text = result.markdown.strip() if hasattr(result, "markdown") else ""
                
                if raw_text:
                    # Filter out URLs to save tokens
                    clean_text = strip_urls(raw_text, strip_titles=strip_titles)
                    
                    # Log the clean content to a debug file for analysis
                    with open(debug_file, "a", encoding="utf-8") as f:
                        f.write(f"URL_TESTER: {url}\n")
                        f.write("-" * 40 + "\n")
                        f.write(f"[CLEANED CONTENT - URLs STRIPPED (Titles Removed: {strip_titles})]\n")
                        f.write(clean_text)
                        f.write("\n\n" + "="*80 + "\n\n")
                    
                    chunks = chunk_text_tokenwise(clean_text,url,query,COLLECTION_NAME, max_tokens=max_tokens, overlap=overlap)
                    all_chunks.extend(chunks)
            except Exception as e:
                print(f"[ERROR] Failed to process {url}: {e}")

    return all_chunks

# Optional sync wrapper for Flask or testing
def load_and_chunk_content(urls: List[str],query:str,COLLECTION_NAME:uuid.UUID, max_tokens: int = 500, overlap: int = 50, strip_titles: bool = False) -> List[str]:
    return asyncio.run(load_and_chunk_content_async(urls,query,COLLECTION_NAME, max_tokens, overlap, strip_titles))
