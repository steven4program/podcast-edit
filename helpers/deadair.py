"""Dead-air detection: word-union gaps where NOBODY talks -> macro-cut proposals.

A recall aid like repeats.py — deterministic scan, the LLM reviews each proposal
(drop a pause that is doing dramatic work). A gap that overlaps a laughter event is
never proposed: laughter is content (hard rule). Each qualifying gap is shortened to
`keep` seconds — half left after the last word, half before the next — so the cut sits
in the middle of the silence and speech rhythm stays natural. Proposals are macro cuts
(start/end seconds); refine_boundaries then slides each boundary off any REAL sound on
any track — token times lie (Scribe parks filler tokens away from their sound, and a
too-quiet sound may have no token at all), and the mid-word guard can't see what was
never tokenized. render's snap + mid-word guard keep the seams safe after that.
"""
import json, sys

from .fillers import _envelope, _voiced_thresh


def find_dead_air(words, events=None, min_gap=1.2, keep=0.8):
    """-> macro cut dicts for every interior gap >= min_gap in the WORD-UNION timeline
    (a moment nobody is talking, across all speakers)."""
    laughs = [(e["start"], e["end"]) for e in (events or []) if e.get("type") == "laughter"]
    out, reach = [], None
    for w in sorted(words, key=lambda w: w["start"]):
        if reach is not None:
            gap = w["start"] - reach
            if gap >= min_gap:
                s, e = reach + keep / 2, w["start"] - keep / 2
                if any(ls < e and s < le for ls, le in laughs):
                    pass  # laughter lives in this pause -> keep it whole
                else:
                    out.append({"type": "macro", "start": round(s, 3), "end": round(e, 3),
                                "reason": f"dead air {gap:.1f}s → keep {keep:.1f}s"})
        reach = w["end"] if reach is None else max(reach, w["end"])
    return out


def _slide(env, thresh, t, forward):
    """t inside a voiced run (frame RMS > thresh) -> the run's outer edge in the given
    direction; else t unchanged."""
    rms, step = env
    i = int(t / step)
    if i < 0 or i >= len(rms) or rms[i] <= thresh:
        return t
    if forward:
        while i < len(rms) and rms[i] > thresh:
            i += 1
        return i * step
    while i >= 0 and rms[i] > thresh:
        i -= 1
    return (i + 1) * step


def refine_boundaries(cuts, track_paths, pad=0.05, min_cut=0.3):
    """Slide every proposal boundary off real sound, checked on every track.
    find_dead_air trusts token times, but Scribe parks fillers' tokens away from their
    sound and never tokenizes some quiet sounds at all — so a boundary can land INSIDE
    a sound, where the mid-word guard (token-based, fillers exempt) cannot see it and
    render's 0.3s snap window may not reach silence. A cut START inside a voiced run
    slides forward past it, an END slides back; a proposal that shrinks under min_cut
    is dropped (the 'gap' was really sound). Boundaries already in silence — including
    gaps that merely CONTAIN a breath without touching a boundary — pass through
    untouched, so recall is unaffected."""
    envs = []
    for p in track_paths:
        env = _envelope(p)
        envs.append((env, _voiced_thresh(env[0])))
    if not envs:
        return list(cuts)
    out = []
    for c in cuts:
        s, e = c["start"], c["end"]
        for _ in range(4):  # sliding off one track's sound can land in another's
            ns = max(_slide(env, th, s, True) for env, th in envs)
            ne = min(_slide(env, th, e, False) for env, th in envs)
            if ns == s and ne == e:
                break
            s, e = ns, ne
        s = s + pad if s != c["start"] else s
        e = e - pad if e != c["end"] else e
        if e - s >= min_cut:
            # snap: false — boundaries just verified against every track's own audio;
            # deadair cuts sit mid-silence BY DESIGN (keep/2 each side), and render's
            # snap would drag them to the silence EDGES, distorting the kept pause.
            out.append({**c, "start": round(s, 3), "end": round(e, 3), "snap": False})
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Propose cuts that shorten long dead air.")
    p.add_argument("transcript")
    p.add_argument("--min-gap", type=float, default=1.2, help="propose only gaps >= this (s)")
    p.add_argument("--keep", type=float, default=0.8, help="pause length left in place (s)")
    a = p.parse_args()
    t = json.load(open(a.transcript, encoding="utf-8"))
    cuts = find_dead_air(t["words"], t.get("events"), a.min_gap, a.keep)
    tracks = t.get("tracks") or ([t["audio"]] if t.get("audio") else [])
    if tracks:
        try:
            cuts = refine_boundaries(cuts, tracks)
        except Exception as e:  # missing/undecodable audio -> ship the raw proposals
            print(f"deadair: acoustic boundary check skipped ({e})", file=sys.stderr)
    print(json.dumps({"dead_air": cuts}, ensure_ascii=False, indent=2))
