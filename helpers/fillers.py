"""Acoustic filler-cut proposal.

Scribe's filler (呃/嗯/啊/欸) timestamps are unreliable — often near-zero width and parked
in the wrong silence (verified: a 呃 whose sound is at 47.2s gets a token at 49.0s). So we
ignore the token's span and instead locate the actual filler SOUND acoustically, inside the
gap between the two surrounding CONTENT words, on that speaker's own track.

Fillers embedded in continuous speech/stutter (no surrounding silence to bound them) or in
cross-talk are flagged for manual review, not cut — forcing them would remove real content.
A filler whose voiced extent overlaps a laughter event is flagged too: the extent is
first-to-last sound in the gap, so a laugh beside the 呃 would be swallowed by the same
cut — and laughter is never collateral (hard rule).
"""
import numpy as np

from .render import _protected, verify_no_midword, frame_rms
from .transcribe import speaker_from_filename

_FILLERS = ("呃", "嗯", "啊", "欸")
_FLOOR_K = 5.0      # voiced = RMS above k x the track's own noise floor
_THRESH_MIN = 1e-4  # never below this (digitally-silent tracks have a ~0 floor)
_MIN_LEN = 0.05     # ignore voiced runs shorter than this (s)
_MAX_EXTENT = 1.6   # voiced span longer than this => filler embedded in speech => flag
_PAD = 0.04


def _envelope(path, frame=0.01):
    rms, _sr = frame_rms(path, frame)   # streamed — never the whole track in RAM
    return rms, frame


def _voiced_thresh(rms, k=_FLOOR_K, floor_min=_THRESH_MIN):
    """Per-track voiced threshold: k x the 10th-percentile RMS (the track's own noise
    floor — same recipe as ai_listen.verify_on_track). An absolute constant misses
    fillers on a quiet mic ('already silent') and over-triggers on a hot one."""
    if not len(rms):
        return floor_min
    return max(k * float(np.percentile(rms, 10)), floor_min)


def voiced_extent(env, a, b, thresh, min_len=_MIN_LEN, pad=_PAD):
    """First-to-last voiced run within [a, b] (original-timeline seconds), padded. None if silent."""
    rms, step = env
    on = rms[int(a / step):int(b / step)] > thresh
    runs, k = [], 0
    while k < len(on):
        if on[k]:
            s = k
            while k < len(on) and on[k]:
                k += 1
            if (k - s) * step >= min_len:
                runs.append((s, k))
        else:
            k += 1
    if not runs:
        return None
    return (max(a, runs[0][0] * step + a - pad), min(b, runs[-1][1] * step + a + pad))


def propose_filler_cuts(transcript, track_paths):
    """-> (cuts, flagged). cuts: macro cut dicts on the actual filler sound. flagged: filler
    ids we couldn't safely cut (embedded in speech / cross-talk / overlapping laughter /
    already silent)."""
    words = transcript["words"]
    # ANY speaker's laughter blocks a cut: a filler cut removes timeline from every track,
    # so a guest laughing over the host's 呃 would be deleted with it.
    laughs = [(e["start"], e["end"]) for e in transcript.get("events", [])
              if e.get("type") == "laughter"]
    env = {speaker_from_filename(p): _envelope(p) for p in track_paths}
    thresh = {spk: _voiced_thresh(e[0]) for spk, e in env.items()}
    content = [w for w in words if _protected(w)]
    cuts, flagged = [], []
    for w in words:
        if w["text"].strip() not in _FILLERS:
            continue
        prev = [c for c in content if c["id"] < w["id"]]
        nxt = [c for c in content if c["id"] > w["id"]]
        if not prev or not nxt or w["speaker"] not in env:
            flagged.append({"id": w["id"], "reason": "no boundary/track"})
            continue
        ext = voiced_extent(env[w["speaker"]], prev[-1]["end"], nxt[0]["start"],
                            thresh[w["speaker"]])
        if ext is None:
            flagged.append({"id": w["id"], "reason": "already silent"})
        elif any(ls < ext[1] and ext[0] < le for ls, le in laughs):
            flagged.append({"id": w["id"], "reason": "overlaps laughter"})
        elif ext[1] - ext[0] > _MAX_EXTENT or verify_no_midword(list(ext), words):
            flagged.append({"id": w["id"], "reason": "embedded in speech / cross-talk"})
        else:
            # snap: false — the extent IS the acoustic boundary (this speaker's own
            # track, floor-relative threshold). render's silence-snap works off the
            # MIX's cruder silencedetect and would drag the edge back into the sound.
            cuts.append({"type": "macro", "start": ext[0], "end": ext[1],
                         "snap": False, "reason": f"filler {w['text'].strip()}"})
    return cuts, flagged


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser(description="Propose acoustic filler cuts (呃嗯啊欸).")
    p.add_argument("transcript", help="transcript.json (its `tracks` list locates the audio)")
    a = p.parse_args()
    t = json.load(open(a.transcript, encoding="utf-8"))
    tracks = t.get("tracks") or [t["audio"]]
    cuts, flagged = propose_filler_cuts(t, tracks)
    print(json.dumps({"cuts": cuts, "flagged": flagged}, ensure_ascii=False, indent=2))
