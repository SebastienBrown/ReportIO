import os
import tempfile
import subprocess
from datetime import datetime


def timestamp_to_seconds(ts):
    t = datetime.strptime(ts, "%H:%M:%S.%f")
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def fetch_transcript_from_yt_dlp(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_dir = tempfile.mkdtemp()
    output_template = os.path.join(output_dir, "%(id)s")

    command = [
        "yt-dlp",
        "--quiet",
        "--write-auto-subs",
        "--sub-lang", "en",
        "--skip-download",
        "--output", output_template,
        url
    ]

    try:
        subprocess.run(command, check=True)
        subtitle_file = f"{output_template % {'id': video_id}}.en.vtt"

        if not os.path.exists(subtitle_file):
            print(f"[❌] .vtt file not found for {video_id}")
            return None

        with open(subtitle_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        segments = []
        for line in lines:
            if "-->" in line:
                try:
                    start = line.split("-->")[0].strip()
                    h, m, s = start.split(":")
                    seconds = int(h) * 3600 + int(m) * 60 + float(s.replace(",", "."))
                    segments.append({"start": seconds, "text": ""})
                except:
                    continue
            elif line.strip() and segments:
                segments[-1]["text"] += " " + line.strip()

        return [
            {"start": seg["start"], "duration": 0, "text": seg["text"].strip()}
            for seg in segments if seg["text"].strip()
        ]

    except subprocess.CalledProcessError as e:
        print(f"[❌] yt-dlp failed for {video_id}: {e}")
        return None
