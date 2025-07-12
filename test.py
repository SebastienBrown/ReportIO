from scripts.seb.scraper import WebScrapingService

url = "https://news.mit.edu/2025/taking-training-wheels-off-clean-energy-0402"
service = WebScrapingService()
result = service.scrape(url)

print(f"✅ Success: {result.success}")
print(f"🔤 Word count: {result.word_count}")
print(f"📝 Title: {result.title}")
print(f"📄 Content Preview:\n{result.content[:500]}")
