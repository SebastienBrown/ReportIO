# scripts/search_module.py
import os
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../backend/.env'))

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

def search_web_articles(query: str, num_results: int = 5):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY}
    payload = {"q": query}

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Search failed: {response.status_code} - {response.text}")

    data = response.json()

    articles = []
    for result in data.get("organic", [])[:num_results]:
        articles.append({
            "title": result.get("title", ""),
            "url": result.get("link", ""),
            "snippet": result.get("snippet", "")
        })

    return articles
