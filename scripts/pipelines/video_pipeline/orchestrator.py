# scripts/pipelines/video_pipeline/orchestrator.py

from .video_search import search_youtube_videos

def run_video_pipeline(query: str, logger=print):
    logger("[DEBUG] [Video] Running video pipeline...")

    try:
        results = search_youtube_videos(query)
        logger(f"[DEBUG] [Video] Found {len(results)} results.")
    except Exception as e:
        logger(f"[ERROR] [Video] Failed to run search: {e}")
        return {"status": "error", "error": str(e)}

    return {
        "status": "ok",
        "video_results": results
    }


if __name__ == "__main__":
    query = "Attention is all you need"
    response = run_video_pipeline(query)

    print("\n[DEBUG] Raw results:")
    print(response)
# <-- This shows us what's really inside
    
    for video in response["video_results"]:
        print(f"\n📺 {video['title']}\n🔗 {video['url']}")
