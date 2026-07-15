"""ElevenLabs Scribe -> canonical transcript.json. normalize_scribe is the testable core."""
import json, os, re, subprocess, sys, tempfile
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


def _posix(path):
    # Store source paths with forward slashes so a transcript.json written on Windows
    # (backslash paths) still resolves when rendered on macOS/Linux. Both OSes accept '/'.
    return path.replace("\\", "/")


def _clamp_word_end(text, start, end):
    # Scribe occasionally emits absurd word durations (a single 股 spanning 98s, an
    # OK-backchannel run spanning 12s). Such a word "covers" a minute of timeline and
    # render's mid-word guard then blocks every cut inside it. Cap the duration at a
    # generous per-character budget; real zh words run ~0.15-0.3s per character.
    budget = 0.5 + 0.45 * max(1, len(re.findall(r"\w", text)))
    return min(end, start + budget)


def normalize_scribe(raw, speaker=None):
    words, events, next_id = [], [], 0
    for e in raw.get("words", []):
        if e.get("type", "word") == "word":
            # Scribe v2 marks hesitation with ellipsis runs (e.g. "......data"). Strip those —
            # real pauses live in the word-timestamp gaps. Keep sentence punctuation (。，) and
            # decimals (single "."). Drop tokens that were nothing but ellipsis.
            text = re.sub(r"\.{2,}|。{2,}|…", "", e["text"])
            if not text:
                continue
            words.append({"id": next_id, "text": text, "start": e["start"],
                          "end": _clamp_word_end(text, e["start"], e["end"]),
                          "speaker": e.get("speaker_id")})
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


def merge_tracks(transcripts, duration=None):
    """`duration` is the real audio length (ffprobe); the word-end fallback truncates
    the tail after the last word (render would cut room tone / outro sounds there)."""
    all_words = [w for t in transcripts for w in t["words"]]
    all_events = [ev for t in transcripts for ev in t["events"]]
    all_words.sort(key=lambda w: (w["start"], w["speaker"] or ""))
    for i, w in enumerate(all_words):
        w["id"] = i
    all_events.sort(key=lambda ev: ev["start"])
    duration = max(duration or 0.0, max((w["end"] for w in all_words), default=0.0))
    return {"audio": "", "tracks": [], "duration": duration, "words": all_words, "events": all_events}


def _extract_wav(audio_path):
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", wav],
                   check=True, capture_output=True)
    return wav


# ── VAD pre-pass ─────────────────────────────────────────────────────────────
# Scribe hallucinates coherent-but-fake text on silent audio. In a multitrack
# recording each mic is silent whenever its owner isn't talking, so we strip
# silence before sending the track to Scribe, then remap the trimmed timestamps
# back onto the original timeline so the merged tracks stay aligned.
# ponytail: energy-based ffmpeg silencedetect, enough for these well-isolated
# mics; swap for Silero VAD if noisy/borderline tracks start leaking.
_SILENCE_DB = -40     # quieter than this counts as silence
_MIN_SILENCE = 0.3    # seconds of quiet before it's a real gap
_SPEECH_PAD = 0.2     # seconds kept around each voiced region (don't clip word edges)


def _duration(audio_path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", audio_path],
                         check=True, capture_output=True, text=True).stdout
    return float(out.strip())


def _detect_silence(audio_path, duration):
    out = subprocess.run(["ffmpeg", "-i", audio_path, "-af",
                          f"silencedetect=noise={_SILENCE_DB}dB:d={_MIN_SILENCE}", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", out)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", out)]
    silences = list(zip(starts, ends))
    if len(starts) > len(ends):          # trailing silence runs to EOF
        silences.append((starts[-1], duration))
    return silences


def voiced_segments(silences, duration, pad=_SPEECH_PAD):
    """Complement of the silence intervals, padded and merged. Original-timeline seconds."""
    voiced, prev = [], 0.0
    for s, e in silences:
        if s > prev:
            voiced.append((prev, s))
        prev = max(prev, e)
    if prev < duration:
        voiced.append((prev, duration))
    merged = []
    for s, e in voiced:
        s, e = max(0.0, s - pad), min(duration, e + pad)
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def remap_to_original(t, segments):
    """Map a timestamp on the trimmed (voiced-only) timeline back to original time."""
    c = 0.0
    for s, e in segments:
        length = e - s
        if t <= c + length:
            return s + (t - c)
        c += length
    return segments[-1][1] if segments else t


def _extract_voiced_wav(audio_path, segments):
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    trims = "".join(f"[0:a]atrim={s}:{e},asetpts=PTS-STARTPTS[s{i}];"
                    for i, (s, e) in enumerate(segments))
    labels = "".join(f"[s{i}]" for i in range(len(segments)))
    fc = f"{trims}{labels}concat=n={len(segments)}:v=0:a=1[out]"
    subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-filter_complex", fc,
                    "-map", "[out]", "-ac", "1", "-ar", "16000", wav],
                   check=True, capture_output=True)
    return wav


def _remap_transcript(transcript, segments):
    """Map the trimmed-timeline transcript back onto the original timeline.

    A token whose trimmed span straddles a VAD seam gets its END stretched across the
    REMOVED silence once remapped (measured: a 嗯 became 10s, a ， 54s — such a token
    pollutes the packed view and its word-guard span blocks every cut under it). The
    speaker is silent past the seam by VAD's own determination, so the sound cannot
    extend beyond the voiced segment the token STARTS in: cap the end there, then
    re-apply the per-character budget (junk durations inside a single long segment)."""
    def _seg_end(t):
        return next((e for s, e in segments if s <= t <= e), None)
    for w in transcript["words"]:
        w["start"], w["end"] = remap_to_original(w["start"], segments), remap_to_original(w["end"], segments)
        cap = _seg_end(w["start"])
        if cap is not None:
            w["end"] = min(w["end"], cap)
        w["end"] = max(w["start"], _clamp_word_end(w["text"], w["start"], w["end"]))
    for ev in transcript["events"]:
        ev["start"], ev["end"] = remap_to_original(ev["start"], segments), remap_to_original(ev["end"], segments)
        cap = _seg_end(ev["start"])
        if cap is not None:
            ev["end"] = max(ev["start"], min(ev["end"], cap))
    transcript["duration"] = max((w["end"] for w in transcript["words"]), default=0.0)
    return transcript


def _scribe(wav_path, language, diarize):
    key = os.environ["ELEVENLABS_API_KEY"]
    with open(wav_path, "rb") as f:
        resp = requests.post("https://api.elevenlabs.io/v1/speech-to-text",
                             headers={"xi-api-key": key},
                             data={"model_id": "scribe_v2", "language_code": language,
                                   "diarize": "true" if diarize else "false",
                                   "tag_audio_events": "true",
                                   "timestamps_granularity": "word"},
                             files={"file": f}, timeout=1800)
    resp.raise_for_status()
    return resp.json()


def _raw_path(out_path, suffix):
    """Raw Scribe dumps are debug intermediates → a work/ subdir next to the transcript."""
    work = os.path.join(os.path.dirname(out_path) or ".", "work")
    os.makedirs(work, exist_ok=True)
    stem = os.path.splitext(os.path.basename(out_path))[0]
    return os.path.join(work, f"{stem}{suffix}")


def transcribe(audio_path, out_path, language="zho"):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    wav = _extract_wav(audio_path)
    try:
        raw = _scribe(wav, language, diarize=True)
    finally:
        os.remove(wav)
    raw["audio"] = audio_path
    with open(_raw_path(out_path, "_raw.json"), "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    transcript = normalize_scribe(raw)
    transcript["audio"] = _posix(audio_path)
    # real audio length, not last-word end — keep the tail after the last word
    transcript["duration"] = max(transcript["duration"], _duration(audio_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    return transcript


def transcribe_multitrack(track_paths, out_path, language="zho"):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    normalized, track_durations = [], []
    for path in track_paths:
        speaker = speaker_from_filename(path)
        duration = _duration(path)
        track_durations.append(duration)
        segments = voiced_segments(_detect_silence(path, duration), duration)
        if not segments:
            print(f"skip {speaker}: no speech detected", file=sys.stderr)
            continue
        wav = _extract_voiced_wav(path, segments)
        try:
            raw = _scribe(wav, language, diarize=False)
        finally:
            os.remove(wav)
        raw["audio"] = path
        with open(_raw_path(out_path, f"_{speaker}_raw.json"), "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
        norm = normalize_scribe(raw, speaker=speaker)
        normalized.append(_remap_transcript(norm, segments))
    merged = merge_tracks(normalized, duration=max(track_durations, default=0.0))
    merged["tracks"] = [_posix(p) for p in track_paths]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return merged


_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Scribe transcription -> canonical transcript.json")
    ap.add_argument("input", help="tracks dir (multitrack) or a single audio file")
    ap.add_argument("out", help="output transcript.json path")
    ap.add_argument("--language", default="zho", help="Scribe language code (default: zho)")
    a = ap.parse_args()
    if os.path.isdir(a.input):
        paths = sorted(
            p for p in (os.path.join(a.input, n) for n in os.listdir(a.input))
            if os.path.splitext(p)[1].lower() in _AUDIO_EXTS
        )
        transcribe_multitrack(paths, a.out, language=a.language)
    else:
        transcribe(a.input, a.out, language=a.language)
