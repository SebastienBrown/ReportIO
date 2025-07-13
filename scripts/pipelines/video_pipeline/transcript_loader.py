from scripts.pipelines.video_pipeline.transcript_fallback import fetch_transcript_from_yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    RequestBlocked,
    IpBlocked
)

def fetch_transcript(video_id: str):
    try:
        return YouTubeTranscriptApi.get_transcript(video_id)

    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(f"[⚠️] Transcript not available for {video_id}: {str(e)}")
        return fetch_transcript_from_yt_dlp(video_id)

    except (RequestBlocked, IpBlocked) as e:
        print(f"[⛔️] IP blocked for {video_id}, using yt-dlp fallback...")
        return fetch_transcript_from_yt_dlp(video_id)

    except Exception as e:
        print(f"[❌] Unexpected error for {video_id}: {str(e)}")
        return None


def fetch_transcripts_bulk(video_infos):
    """
    Fetch transcripts for each video.
    Falls back to title + description when transcript fails.
    """
    transcripts = {}
    for video in video_infos:
        video_id = video["videoId"]
        transcript = fetch_transcript(video_id)

        if transcript:
            transcripts[video_id] = transcript
        else:
            print(f"[📝] Falling back to title + description for {video_id}")
            fallback_text = f"{video['title']}\n{video['description']}"
            transcripts[video_id] = [{
                "start": 0,
                "text": fallback_text
            }]
    return transcripts
