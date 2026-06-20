"""ElevenLabs Scribe -> canonical transcript.json. normalize_scribe is the testable core."""
import json, os, subprocess, tempfile
import requests

_EVENT_MAP = {"cough": "cough", "throat": "throat_clear", "laugh": "laughter", "breath": "breath"}


def _classify_event(text):
    low = text.lower()
    for key, label in _EVENT_MAP.items():
        if key in low:
            return label
    return "other"


def _nearest_speaker(words, t):
    return min(words, key=lambda w: abs(w["start"] - t))["speaker"] if words else None


def normalize_scribe(raw):
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
    return {"audio": raw.get("audio", ""), "duration": words[-1]["end"] if words else 0.0,
            "words": words, "events": events}


def _extract_wav(audio_path):
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", wav],
                   check=True, capture_output=True)
    return wav


def transcribe(audio_path, out_path, language="zho"):
    key = os.environ["ELEVENLABS_API_KEY"]
    wav = _extract_wav(audio_path)
    try:
        with open(wav, "rb") as f:
            resp = requests.post("https://api.elevenlabs.io/v1/speech-to-text",
                                 headers={"xi-api-key": key},
                                 data={"model_id": "scribe_v1", "language_code": language,
                                       "diarize": "true", "tag_audio_events": "true",
                                       "timestamps_granularity": "word"},
                                 files={"file": f}, timeout=1800)
        resp.raise_for_status()
        raw = resp.json()
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


if __name__ == "__main__":
    import sys
    transcribe(sys.argv[1], sys.argv[2])
