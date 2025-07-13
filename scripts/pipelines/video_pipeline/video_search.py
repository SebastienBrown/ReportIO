# scripts/pipelines/video_pipeline/video_search.py

import os
import requests
from dotenv import load_dotenv

load_dotenv()  # Load from .env

YOUTUBE_SEARCH_API_KEY = os.getenv("YOUTUBE_SEARCH_API_KEY")

def search_youtube_videos(query, max_results=5):
    if not YOUTUBE_SEARCH_API_KEY:
        raise EnvironmentError("Missing YOUTUBE_SEARCH_API_KEY in environment variables.")

    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "key": YOUTUBE_SEARCH_API_KEY
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    results = response.json()

    videos = []
    for item in results.get("items", []):
        videos.append({
            "title": item["snippet"]["title"],
            "videoId": item["id"]["videoId"],
            "channelTitle": item["snippet"]["channelTitle"],
            "publishedAt": item["snippet"]["publishedAt"],
            "description": item["snippet"]["description"],
            "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        })

    return videos
