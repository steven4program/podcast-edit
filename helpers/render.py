"""Render: turn a cut-list into a smooth output audio file.

Pure timeline math (resolve / kept-segments / remap / verify) is the testable core.
Audio is rendered per-track and mixed down: every track is cut at the SAME merged-
timeline boundaries, so the tracks stay aligned and Phase 5 can mute a single track
(e.g. host cough) without touching the others. Single-file input still works.
"""
import json, os, re, subprocess


# ── pure data core ───────────────────────────────────────────────────────────
def resolve_cut_times(cuts, words):
    by_id = {w["id"]: w for w in words}
    out = []
    for c in cuts:
        if c.get("start_word") is not None:
            start, end = by_id[c["start_word"]]["start"], by_id[c["end_word"]]["end"]
            # Scribe word spans micro-overlap, so end_word's end can run a few ms past
            # the next (kept) word's start and clip it (verify uses a 50ms eps for the
            # same reason; remap doesn't). Clamp the cut to that word's start.
            nxt = by_id.get(c["end_word"] + 1)
            if nxt is not None and nxt["start"] > start:
                end = min(end, nxt["start"])
        else:
            start, end = c["start"], c["end"]
        out.append({**c, "start": start, "end": end})
    return out


def compute_kept_segments(cuts, duration):
    intervals = sorted((c["start"], c["end"]) for c in cuts)
    merged = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    segs, pos = [], 0.0
    for s, e in merged:
        if s > pos:
            segs.append((pos, s))
        pos = max(pos, e)
    if pos < duration:
        segs.append((pos, duration))
    return segs


def _removed_before(t, segments):
    kept = 0.0
    for s, e in segments:
        if e <= t:
            kept += e - s
        elif s < t < e:
            kept += t - s
    return kept


def remap_words(words, segments):
    kept = []
    for w in words:
        if not any(s <= w["start"] < e for s, e in segments):
            continue
        kept.append({**w, "start": _removed_before(w["start"], segments),
                     "end": _removed_before(w["end"], segments)})
    return kept


_FILLERS = ("呃", "嗯", "啊", "欸")


def _is_word(w):
    return bool(re.search(r"\w", w["text"]))  # punctuation-only tokens (。，？…) are cuttable


def _protected(w):
    # A cut must not slice a CONTENT word. Punctuation and fillers (呃嗯啊欸) are removable,
    # so they never block a cut — otherwise the mid-word guard refuses to cut the very
    # filler we're trying to remove.
    return _is_word(w) and w["text"].strip() not in _FILLERS


_MIDWORD_EPS = 0.05  # a boundary within 50ms of a word edge counts as "at the edge"


def _inside_word(t, real, eps=_MIDWORD_EPS):
    # Scribe word timestamps are imprecise and micro-overlap each other, so only flag a
    # boundary that is clearly INSIDE a word (>eps from both edges), not edge-touches.
    return any(w["start"] + eps < t < w["end"] - eps for w in real)


def verify_no_midword(boundaries, words):
    # Only real words block a cut. Punctuation tokens don't — and Scribe sometimes gives
    # them junk durations (a trailing ？ spanning tens of seconds), which would falsely
    # block every cut in that span.
    real = [w for w in words if _protected(w)]
    return [t for t in boundaries if _inside_word(t, real)]


# ── audio: silence detection, snapping, render ───────────────────────────────
def detect_silences(path, thresh_db=-35.0, min_len=0.3):
    proc = subprocess.run(
        ["ffmpeg", "-i", path, "-af",
         f"silencedetect=noise={thresh_db}dB:d={min_len}", "-f", "null", "-"],
        capture_output=True, text=True)
    starts, sils = [], []
    for line in proc.stderr.splitlines():
        if "silence_start:" in line:
            starts.append(float(line.split("silence_start:")[1].strip()))
        elif "silence_end:" in line:
            end = float(line.split("silence_end:")[1].split("|")[0].strip())
            if starts:
                sils.append((starts.pop(), end))
    return sils


def snap_point(t, silences, window=0.3):
    best, best_d = None, window
    for s, e in silences:
        for edge in (s, e):
            if abs(edge - t) <= best_d:
                best, best_d = edge, abs(edge - t)
    return (best, True) if best is not None else (t, False)


def safe_snap(t, silences, words, window=0.3):
    """Snap to silence, but never onto a position inside a real word (acoustic silence
    edges don't always align with Scribe's word timings). Falls back to the original t."""
    nt, snapped = snap_point(t, silences, window)
    if snapped and _inside_word(nt, [w for w in words if _protected(w)]):
        return t, False
    return nt, snapped


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", path], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def _snap_silences(sources, scratch):
    """Silences to snap cut points to = where NOBODY is talking = silence in the mix."""
    if len(sources) == 1:
        return detect_silences(sources[0])
    mix = scratch + ".snapmix.wav"
    inputs = [x for src in sources for x in ("-i", src)]
    subprocess.run(["ffmpeg", "-y", *inputs, "-filter_complex",
                    f"amix=inputs={len(sources)}:normalize=0[m]", "-map", "[m]", mix],
                   check=True, capture_output=True)
    try:
        return detect_silences(mix)
    finally:
        os.remove(mix)


def _filtergraph(n_tracks, segments, track_mutes=None):
    """Cut every track at `segments` (3ms fades per piece), concat per track, then mix.
    track_mutes[ti] = [(s,e),...] silences that track in those original-timeline spans
    (the multitrack cough superpower: kill a cough on the host's track without removing time
    or touching other speakers)."""
    track_mutes = track_mutes or {}
    parts, track_labels = [], []
    for ti in range(n_tracks):
        src = f"[{ti}:a]"
        if track_mutes.get(ti):
            en = "+".join(f"between(t,{s},{e})" for s, e in track_mutes[ti])
            parts.append(f"{src}volume=0:enable='{en}'[m{ti}]")
            src = f"[m{ti}]"
        seg_labels = []
        for si, (s, e) in enumerate(segments):
            lbl = f"a{ti}_{si}"
            parts.append(f"{src}atrim={s}:{e},asetpts=PTS-STARTPTS,"
                         f"afade=t=in:st=0:d=0.003,"
                         f"afade=t=out:st={max(e - s - 0.003, 0)}:d=0.003[{lbl}]")
            seg_labels.append(f"[{lbl}]")
        parts.append("".join(seg_labels) + f"concat=n={len(segments)}:v=0:a=1[t{ti}]")
        track_labels.append(f"[t{ti}]")
    loud = "loudnorm=I=-16:TP=-1.5:LRA=11[out]"
    if n_tracks == 1:
        parts.append(f"{track_labels[0]}{loud}")
    else:
        parts.append("".join(track_labels) + f"amix=inputs={n_tracks}:normalize=0,{loud}")
    return ";".join(parts)


def _speaker(path):
    return os.path.splitext(os.path.basename(path))[0].split("--")[-1]


def render(transcript, cuts, audio_path, out_path, snap_window=0.3, mutes=None):
    words, duration = transcript["words"], transcript["duration"]
    sources = transcript.get("tracks") or [audio_path]
    resolved = resolve_cut_times(cuts, words)

    silences = _snap_silences(sources, out_path)
    snapped_count = 0
    for c in resolved:
        # Word-id cuts (repeats/stutters) are already on word edges — snapping a short
        # intra-speech cut to a nearby silence drags it onto adjacent words. The mid-word
        # guard + 3ms fades keep the seam clean. Only acoustic/segment cuts snap.
        if c.get("start_word") is not None:
            continue
        ns, s1 = safe_snap(c["start"], silences, words, snap_window)
        ne, s2 = safe_snap(c["end"], silences, words, snap_window)
        c["start"], c["end"] = ns, ne
        snapped_count += int(s1) + int(s2)

    segments = compute_kept_segments(resolved, duration)
    if not segments:
        raise ValueError("cuts remove the entire audio")
    boundaries = [b for seg in segments for b in seg if 0 < b < duration]
    flagged = verify_no_midword(boundaries, words)
    if flagged:
        raise ValueError(f"mid-word boundaries after snap: {flagged}")

    track_mutes = {}
    for m in mutes or []:
        for ti, src in enumerate(sources):
            if _speaker(src) == m["speaker"]:
                track_mutes.setdefault(ti, []).append((m["start"], m["end"]))
    inputs = [x for src in sources for x in ("-i", src)]
    fc = _filtergraph(len(sources), segments, track_mutes)
    subprocess.run(["ffmpeg", "-y", *inputs, "-filter_complex", fc, "-map", "[out]", out_path],
                   check=True, capture_output=True)

    kept = {"words": remap_words(words, segments), "segments": segments}
    with open(out_path.replace(".mp3", "_kept_transcript.json"), "w") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    return {"segments": segments, "snapped": snapped_count, "flagged": flagged}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("transcript"); p.add_argument("cuts"); p.add_argument("out")
    p.add_argument("--audio", help="single-file source (ignored if transcript has tracks)")
    a = p.parse_args()
    t = json.load(open(a.transcript))
    spec = json.load(open(a.cuts))  # {"cuts": [...], "mutes": [...]} (mutes optional)
    res = render(t, spec["cuts"], a.audio, a.out, mutes=spec.get("mutes"))
    print(json.dumps(res, ensure_ascii=False))
