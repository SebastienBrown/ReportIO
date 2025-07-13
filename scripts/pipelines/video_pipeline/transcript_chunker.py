import tiktoken
from scripts.pipelines.video_pipeline.transcript_loader import fetch_transcripts_bulk

def chunk_transcript_segments(transcript, max_tokens=500, overlap=50):
    """
    Chunk transcript segments (with timestamps) into token-aware chunks.
    Returns list of dicts with chunk text and start time.
    """
    enc = tiktoken.get_encoding("cl100k_base")

    chunks = []
    current_tokens = []
    current_texts = []
    current_start = None

    for segment in transcript:
        segment_text = segment["text"].strip()
        segment_start = segment.get("start", 0)  # default to 0 if missing

        encoded = enc.encode(segment_text)

        if not current_tokens:
            current_start = segment_start

        current_tokens.extend(encoded)
        current_texts.append(segment_text)

        if len(current_tokens) >= max_tokens:
            chunk_text = enc.decode(current_tokens[:max_tokens]).strip()
            chunks.append({
                "chunk": chunk_text,
                "start": current_start
            })
            overlap_tokens = current_tokens[-overlap:] if overlap > 0 else []
            current_tokens = list(overlap_tokens)
            current_texts = []
            current_start = segment_start

    if current_tokens:
        chunk_text = enc.decode(current_tokens).strip()
        chunks.append({
            "chunk": chunk_text,
            "start": current_start
        })

    return chunks


def fetch_and_chunk_transcripts(video_infos, logger=print):
    logger("[DEBUG] Step 2: Fetching transcripts...")
    transcripts = fetch_transcripts_bulk(video_infos)
    logger(f"[DEBUG] Step 2 done → {len(transcripts)} transcripts loaded.")

    all_chunks = []
    for video in video_infos:
        vid = video["videoId"]
        if vid in transcripts:
            chunks = chunk_transcript_segments(transcripts[vid])
        else:
            fallback_text = f"{video['title']}\n{video['description']}".strip()
            if fallback_text:
                chunks = chunk_transcript_segments([{
                    "text": fallback_text,
                    "start": 0
                }])
            else:
                continue

        for chunk in chunks:
            all_chunks.append({
                "video_id": vid,
                "chunk": chunk["chunk"],
                "start": chunk["start"],
                "title": video.get("title"),
                "url": f"{video.get('url')}&t={int(chunk['start'])}"

            })

    logger(f"[DEBUG] Step 3 done → {len(all_chunks)} chunks created.")
    return all_chunks
