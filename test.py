from scripts.pipelines.video_pipeline.transcript_scraper import fetch_transcript_from_html

video_id = "mvORZvd3QsM"
segments = fetch_transcript_from_html(video_id)

print(f"\n[TEST] Retrieved {len(segments) if segments else 0} segments:")
if segments:
    for s in segments[:3]:
        print(f"→ {s['start']}s: {s['text'][:60]}")
