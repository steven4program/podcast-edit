"""ElevenLabs Scribe -> canonical transcript.json. normalize_scribe is the testable core."""
import json, os, subprocess, tempfile
import requests

# Scribe emits audio-event labels in the transcription language. For zh audio these
# are Chinese (e.g. "(清嗓声)" throat clear, "(笑声)" laughter). Substring-matched in order;
# English keywords kept for non-zh audio. Order matters: specific before generic.
_EVENT_KEYWORDS = [
    ("咳", "cough"), ("cough", "cough"),
    ("清嗓", "throat_clear"), ("清喉", "throat_clear"), ("throat", "throat_clear"),
    ("笑", "laughter"), ("laugh", "laughter"),
    ("吸气", "breath"), ("呼吸", "breath"), ("喘", "breath"), ("breath", "breath"),
    ("停顿", "pause"), ("沉默", "pause"), ("静默", "pause"), ("无语", "pause"),
    ("pause", "pause"), ("silen", "pause"),
    ("噪", "noise"), ("背景", "noise"), ("按键", "noise"), ("键盘", "noise"), ("noise", "noise"),
]


def _classify_event(text):
    low = text.lower()
    for key, label in _EVENT_KEYWORDS:
        if key.lower() in low:
            return label
    return "other"


def _nearest_speaker(words, t):
    return min(words, key=lambda w: abs(w["start"] - t))["speaker"] if words else None


def speaker_from_filename(path):
    return os.path.splitext(os.path.basename(path))[0].split("--")[-1]


def normalize_scribe(raw, speaker=None):
    words, events, next_id = [], [], 0
    for e in raw.get("words", []):
        if e.get("type", "word") == "word":
            words.append({"id": next_id, "text": e["text"], "start": e["start"],
                          "end": e["end"], "speaker": e.get("speaker_id")})
            next_id += 1
        elif e.get("type") == "audio_event":
            events.append({"type": _classify_event(e["text"]), "start": e["start"],
                           "end": e["end"], "speaker": e.get("speaker_id")})
    for ev in events:
        if ev["speaker"] is None:
            ev["speaker"] = _nearest_speaker(words, ev["start"])
    if speaker is not None:
        for w in words:
            w["speaker"] = speaker
        for ev in events:
            ev["speaker"] = speaker
    return {"audio": raw.get("audio", ""), "duration": words[-1]["end"] if words else 0.0,
            "words": words, "events": events}


def merge_tracks(transcripts):
    all_words = [w for t in transcripts for w in t["words"]]
    all_events = [ev for t in transcripts for ev in t["events"]]
    all_words.sort(key=lambda w: (w["start"], w["speaker"] or ""))
    for i, w in enumerate(all_words):
        w["id"] = i
    all_events.sort(key=lambda ev: ev["start"])
    duration = max((w["end"] for w in all_words), default=0.0)
    return {"audio": "", "tracks": [], "duration": duration, "words": all_words, "events": all_events}


def _extract_wav(audio_path):
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", wav],
                   check=True, capture_output=True)
    return wav


def _scribe(wav_path, language, diarize):
    key = os.environ["ELEVENLABS_API_KEY"]
    with open(wav_path, "rb") as f:
        resp = requests.post("https://api.elevenlabs.io/v1/speech-to-text",
                             headers={"xi-api-key": key},
                             data={"model_id": "scribe_v1", "language_code": language,
                                   "diarize": "true" if diarize else "false",
                                   "tag_audio_events": "true",
                                   "timestamps_granularity": "word"},
                             files={"file": f}, timeout=1800)
    resp.raise_for_status()
    return resp.json()


def transcribe(audio_path, out_path, language="zho"):
    wav = _extract_wav(audio_path)
    try:
        raw = _scribe(wav, language, diarize=True)
    finally:
        os.remove(wav)
    raw["audio"] = audio_path
    with open(out_path.replace(".json", "_raw.json"), "w") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    transcript = normalize_scribe(raw)
    transcript["audio"] = audio_path
    with open(out_path, "w") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    return transcript


def transcribe_multitrack(track_paths, out_path, language="zho"):
    base = out_path.replace(".json", "")
    normalized = []
    for path in track_paths:
        speaker = speaker_from_filename(path)
        wav = _extract_wav(path)
        try:
            raw = _scribe(wav, language, diarize=False)
        finally:
            os.remove(wav)
        raw["audio"] = path
        raw_out = f"{base}_{speaker}_raw.json"
        with open(raw_out, "w") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
        normalized.append(normalize_scribe(raw, speaker=speaker))
    merged = merge_tracks(normalized)
    merged["tracks"] = list(track_paths)
    with open(out_path, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return merged


_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}

if __name__ == "__main__":
    import sys
    inp, out = sys.argv[1], sys.argv[2]
    if os.path.isdir(inp):
        paths = sorted(
            p for p in (os.path.join(inp, n) for n in os.listdir(inp))
            if os.path.splitext(p)[1].lower() in _AUDIO_EXTS
        )
        transcribe_multitrack(paths, out)
    else:
        transcribe(inp, out)
