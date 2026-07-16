"""Audio-LLM clip classifier + throat-clear discovery sweep for the cough-prone host.

The host's throat/nose-clears are the headline edit (most of an editor's time). Scribe's
event tagger has poor recall on these soft sounds and RMS energy can't tell a throat-clear
from a voiced syllable — so the recall mechanism is an audio-LLM sweep over the host's OWN
track: slice into windows, ask the model to timestamp every throat/nose-clear/cough, then
split the hits into (a) ones in a gap in his own speech -> safe to mute, and (b) ones
co-articulated with his own words -> flag (cut/mute can't remove those without losing words).

The model is a pluggable provider (Gemini / OpenAI / …) — see ai_providers.py. sweep_track
and classify take a provider object (dependency injection) so the detection logic here is
backend-agnostic and unit-testable with a fake provider (no network in tests).
NOTE: both audio-LLM backends are currently commented out in ai_providers.py (user
request 2026-07-11) — the CLI therefore defaults to the keyless `scribe` path below.

Laughter is NEVER a candidate: any 'laughter' label means keep.
"""
import json, os, re, subprocess, sys, tempfile, time

from .render import probe_duration, _protected, frame_rms
from .ai_providers import get_provider, _PROVIDERS

_KEYWORDS = [("throat", "throat_clear"), ("nose", "throat_clear"), ("sniff", "throat_clear"),
             ("cough", "cough"), ("laugh", "laughter"), ("breath", "breath"),
             ("speech", "speech"), ("talk", "speech")]
_PROMPT = ("Classify this short audio clip as exactly one of: cough, throat clearing, "
           "laughter, breath, speech. Answer with the single best label.")
_SWEEP_PROMPT = (
    "This is a short audio clip of ONE Mandarin-Chinese speaker. List every throat-clearing, "
    "nose-clearing/sniff, or cough sound. Do NOT list laughter, breaths, or normal speech. "
    "For each, give start and end time in SECONDS from the start of THIS clip and a type of "
    '"throat_clear" or "cough". Respond ONLY as a JSON array like '
    '[{"start":1.2,"end":1.6,"type":"throat_clear"}]. If there are none, respond [].')


def parse_label(text):
    low = text.lower()
    for key, label in _KEYWORDS:
        if key in low:
            return label
    return "other"


def _retry_after(exc, default):
    # 429s carry the server's wait ("Please retry in 50.6s" / "retryDelay': '50s'") — honor it.
    m = re.search(r"retry[^0-9]*(\d+(?:\.\d+)?)\s*s", str(exc))
    return float(m.group(1)) + 1.0 if m else default


def _extract_clip(audio_path, start, end):
    fd, clip = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(["ffmpeg", "-y", "-ss", str(start), "-t", str(end - start), "-i", audio_path,
                    "-ac", "1", "-ar", "16000", clip], check=True, capture_output=True)
    return clip


def classify(audio_path, start, end, provider=None, tries=3):
    provider = provider or get_provider()
    clip = _extract_clip(audio_path, start, end)
    try:
        last = None
        for i in range(tries):
            try:
                return parse_label(provider.generate(_PROMPT, clip))
            except Exception as e:  # transient API/file-state errors -> back off and retry
                last = e
                time.sleep(_retry_after(e, 2.0))
        raise last
    finally:
        os.remove(clip)


# ── discovery sweep (pure helpers are unit-tested; Gemini calls are live) ──────
def _windows(duration, window, hop):
    out, t = [], 0.0
    while t < duration:
        out.append((round(t, 3), round(min(t + window, duration), 3)))
        if t + window >= duration:
            break
        t += hop
    return out


def _parse_events(text, offset, win_end):
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for e in arr if isinstance(arr, list) else []:
        try:
            s, en = float(e["start"]) + offset, float(e["end"]) + offset
        except (KeyError, TypeError, ValueError):
            continue
        lab = parse_label(str(e.get("type", "")))
        if lab not in ("throat_clear", "cough"):   # drops laughter/breath/speech/other
            continue
        out.append({"start": round(max(s, offset), 3),
                    "end": round(min(en, win_end), 3), "type": lab})
    return [e for e in out if e["end"] > e["start"]]


def _merge(events, gap=0.4):
    merged = []
    for e in sorted(events, key=lambda x: x["start"]):
        if merged and e["start"] <= merged[-1]["end"] + gap:
            merged[-1]["end"] = max(merged[-1]["end"], e["end"])
        else:
            merged.append(dict(e))
    return merged


def verify_on_track(events, host_track, k=5.0, pad=0.06):
    """Drop events with no real energy on the host's OWN track. Gemini (like Scribe)
    hallucinates non-verbal events in near-silent windows, and another speaker talking on
    THEIR track must not count as the host clearing his throat. Keep an event only if its
    peak RMS on the host track is >= k x the track's noise floor."""
    import numpy as np, soundfile as sf
    rms, sr = frame_rms(host_track, 0.02)   # streamed — never the whole track in RAM
    if not len(rms):
        return list(events)                 # unreadable/empty track: fail open
    fr = int(0.02 * sr)
    thresh = k * float(np.percentile(rms, 10))
    out = []
    with sf.SoundFile(host_track) as f:
        for ev in events:
            # clamp at 0: a negative start index would slice from the track's END
            # (Python negative indexing) and silently drop a real event at t≈0
            a = max(0, int((ev["start"] - pad) * sr))
            b = min(len(f), int((ev["end"] + pad) * sr))
            peak = 0.0
            if b > a:
                f.seek(a)
                s = f.read(b - a, dtype="float64")
                s = s if s.ndim == 1 else s.mean(1)
                m = len(s) // fr
                if m:
                    peak = float(np.sqrt((s[:m * fr].reshape(m, fr) ** 2).mean(1)).max())
            if peak >= thresh:
                out.append(ev)
    return out


def split_events(events, words, host, eps=0.05):
    """gap -> mute (safe), co-articulated with host's own content word -> flag."""
    hw = [w for w in words if w.get("speaker") == host and _protected(w)]
    mutes, flagged = [], []
    for ev in events:
        coarticulated = any(w["start"] + eps < ev["end"] and ev["start"] < w["end"] - eps
                            for w in hw)
        if coarticulated:
            flagged.append(ev)
        else:
            mutes.append({"speaker": host, "start": ev["start"], "end": ev["end"]})
    return mutes, flagged


def scribe_events(transcript, host, pad=0.3):
    """Keyless cough/throat-clear candidates: reuse the Scribe audio-event tags already
    in transcript.json (tag_audio_events comes free with transcription). Much lower
    recall than an audio-LLM sweep — on a test episode Scribe tagged 13 clears where the
    sweep found 38, with almost no overlap — so this is the fallback when no provider
    key is available (or the user asks for Scribe-only). Scribe often parks a tag at
    zero width, so events are padded — but the padding is clipped at the host's own word
    boundaries AND his own laughter events: only the RAW tag span decides mute-vs-flag
    in split_events (padding into a neighbouring word must not fake a co-articulation),
    and a mute reaching into a laugh would behead it — laughter is never collateral.
    A tag overlapping his laughter at all is no candidate (any detector saying laughter
    means keep)."""
    hw = [w for w in transcript.get("words", [])
          if w.get("speaker") == host and _protected(w)]
    hl = [ev for ev in transcript.get("events", [])
          if ev.get("speaker") == host and ev.get("type") == "laughter"]

    def _overlaps_laugh(s, e):
        return any(l["start"] < e and s < l["end"] for l in hl)

    out = []
    for ev in transcript.get("events", []):
        if ev.get("speaker") != host or ev.get("type") not in ("throat_clear", "cough"):
            continue
        if _overlaps_laugh(ev["start"], ev["end"]):
            continue
        prev_end = max((t for t in [w["end"] for w in hw] + [l["end"] for l in hl]
                        if t <= ev["start"]), default=0.0)
        next_start = min((t for t in [w["start"] for w in hw] + [l["start"] for l in hl]
                          if t >= ev["end"]), default=float("inf"))
        s = min(max(ev["start"] - pad, prev_end, 0.0), ev["start"])
        e = max(min(ev["end"] + pad, next_start), ev["end"])
        out.append({"start": round(s, 3), "end": round(e, 3), "type": ev["type"]})
    # two clears sandwiching a laugh shorter than the merge gap would bridge it — drop
    # any merged span that overlaps a laugh (lose two mutes, never behead the laugh)
    return [e for e in _merge(out) if not _overlaps_laugh(e["start"], e["end"])]


def sweep_track(audio_path, provider=None, window=12.0, hop=11.0,
                throttle=7.0, tries=4, max_requests=None):
    """Sweep the track for throat/nose-clears via `provider` (default: get_provider()).
    Throttle between windows for free-tier RPM (set 0 on paid tier); on a 429 honor the
    server's retryDelay. max_requests is a hard cost cap on model calls per run."""
    provider = provider or get_provider()
    wins = _windows(probe_duration(audio_path), window, hop)
    if max_requests is not None and len(wins) > max_requests:
        print(f"sweep: capping {len(wins)} windows to max_requests={max_requests} "
              f"(only first {round(wins[max_requests - 1][1])}s swept)", file=sys.stderr)
        wins = wins[:max_requests]
    events = []
    for wi, (s, e) in enumerate(wins):
        if wi and throttle:
            time.sleep(throttle)
        clip = _extract_clip(audio_path, s, e)
        try:
            last = None
            for i in range(tries):
                try:
                    events += _parse_events(provider.generate(_SWEEP_PROMPT, clip, json_mode=True), s, e)
                    break
                except Exception as ex:
                    last = ex
                    time.sleep(_retry_after(ex, 8.0))
            else:
                raise last
        finally:
            os.remove(clip)
    return _merge(events)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Sweep the host's track for throat/nose-clears.")
    p.add_argument("host_track"); p.add_argument("transcript"); p.add_argument("host")
    # no audio-LLM backend enabled -> scribe is the only working path, make it the default
    # (re-enabling a provider in ai_providers.py flips the default back automatically)
    p.add_argument("--provider", default=("scribe" if not _PROVIDERS else None),
                   help="scribe = keyless, reuse transcript.json's Scribe event tags "
                        "(low recall; the CURRENT default — Gemini/OpenAI are commented "
                        "out in ai_providers.py) | or an enabled audio-LLM provider name")
    p.add_argument("--model", default=None, help="override the provider's default model")
    p.add_argument("--throttle", type=float, default=7.0, help="sleep between windows; 0 on paid tier")
    p.add_argument("--window", type=float, default=12.0)
    p.add_argument("--hop", type=float, default=11.0)
    p.add_argument("--max-requests", type=int, default=None, help="hard cap on model calls (cost)")
    a = p.parse_args()
    t = json.load(open(a.transcript, encoding="utf-8"))
    if a.provider == "scribe":
        events = scribe_events(t, a.host)
    else:
        provider = get_provider(a.provider, a.model)
        events = sweep_track(a.host_track, provider, window=a.window, hop=a.hop,
                             throttle=a.throttle, max_requests=a.max_requests)
    events = verify_on_track(events, a.host_track)   # drop silent-window hallucinations
    mutes, flagged = split_events(events, t["words"], a.host)
    print(json.dumps({"found": len(events), "mutes": mutes, "flagged": flagged},
                     ensure_ascii=False, indent=2))
