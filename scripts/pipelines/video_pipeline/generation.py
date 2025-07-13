from collections import defaultdict
from scripts.llm.chat import chat_llm


def summarize_chunk(chunk_text: str) -> str:
    """Summarize a video moment in 2–5 words."""
    prompt = (
        "Summarize the following video moment in 2 to 5 words. "
        "Just return the concise summary, no punctuation."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": chunk_text}
    ]
    response = chat_llm.invoke(messages)
    return response.content.strip()


def summarize_video(chunks: list[str]) -> str:
    """Summarize the whole video in one sentence using its chunks."""
    joined_text = "\n".join(chunks[:3])[:1000]  # limit to first ~1k chars
    prompt = (
        "Summarize the overall content of this video in one clear sentence. "
        "Focus on what the video teaches or explains."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": joined_text}
    ]
    response = chat_llm.invoke(messages)
    return response.content.strip()


def group_and_summarize_video_chunks(chunks: list[dict]) -> list[dict]:
    """
    Groups video chunks by video_id, summarizes each key moment (2–5 words),
    and generates a 1-sentence summary of the overall video.
    """
    grouped = defaultdict(list)

    for chunk in chunks:
        grouped[chunk["video_id"]].append(chunk)

    results = []
    for video_id, moments in grouped.items():
        if not moments:
            continue

        base = moments[0]
        key_moments = []

        for m in moments:
            summary = summarize_chunk(m["chunk"])
            key_moments.append({
                "start": m["start"],
                "summary": summary,
                "text": m["chunk"],
                "url": m["url"]
            })

        video_summary = summarize_video([m["chunk"] for m in moments])

        results.append({
            "video_id": video_id,
            "title": base["title"],
            "url": base["url"].split("&t=")[0],
            "summary": video_summary,
            "moments": sorted(key_moments, key=lambda x: x["start"])
        })

    return results
