"""Gemini clip classifier + throat-clear discovery sweep for the cough-prone host.

The host's throat/nose-clears are the headline edit (most of an editor's time). Scribe's
event tagger has poor recall on these soft sounds and RMS energy can't tell a throat-clear
from a voiced syllable — so the recall mechanism is a Gemini sweep over the host's OWN track:
slice into windows, ask Gemini to timestamp every throat/nose-clear/cough, then split the
hits into (a) ones in a gap in his own speech -> safe to mute, and (b) ones co-articulated
with his own words -> flag (cut/mute can't remove those without losing the words).

Laughter is NEVER a candidate: any 'laughter' label means keep.
"""
import json, os, re, subprocess, sys, tempfile, time

from .render import probe_duration, _protected

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


def classify(audio_path, start, end, model="gemini-2.5-flash", tries=3):
    from google import genai
    clip = _extract_clip(audio_path, start, end)
    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        uploaded = client.files.upload(file=clip)
        last = None
        for i in range(tries):
            try:
                resp = client.models.generate_content(model=model, contents=[_PROMPT, uploaded])
                return parse_label(resp.text)
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
    x, sr = sf.read(host_track)
    x = x if x.ndim == 1 else x.mean(1)
    fr = int(0.02 * sr)
    n = len(x) // fr
    floor = float(np.percentile(np.sqrt((x[:n * fr].reshape(n, fr) ** 2).mean(1)), 10))
    thresh = k * floor
    out = []
    for ev in events:
        s = x[int((ev["start"] - pad) * sr):int((ev["end"] + pad) * sr)]
        m = len(s) // fr
        peak = float(np.sqrt((s[:m * fr].reshape(m, fr) ** 2).mean(1)).max()) if m else 0.0
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


def sweep_track(audio_path, window=12.0, hop=11.0, model="gemini-2.5-flash",
                throttle=7.0, tries=4, max_requests=None):
    """Throttle between windows for free-tier RPM (set 0 on paid tier); on a 429 honor the
    server's retryDelay. max_requests is a hard cost cap on Gemini calls per run."""
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
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
            uploaded = client.files.upload(file=clip)
            last = None
            for i in range(tries):
                try:
                    resp = client.models.generate_content(
                        model=model, contents=[_SWEEP_PROMPT, uploaded],
                        config={"response_mime_type": "application/json"})
                    events += _parse_events(resp.text, s, e)
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
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--throttle", type=float, default=7.0, help="sleep between windows; 0 on paid tier")
    p.add_argument("--window", type=float, default=12.0)
    p.add_argument("--hop", type=float, default=11.0)
    p.add_argument("--max-requests", type=int, default=None, help="hard cap on Gemini calls (cost)")
    a = p.parse_args()
    t = json.load(open(a.transcript))
    events = sweep_track(a.host_track, window=a.window, hop=a.hop, model=a.model,
                         throttle=a.throttle, max_requests=a.max_requests)
    events = verify_on_track(events, a.host_track)   # drop silent-window hallucinations
    mutes, flagged = split_events(events, t["words"], a.host)
    print(json.dumps({"found": len(events), "mutes": mutes, "flagged": flagged},
                     ensure_ascii=False, indent=2))
