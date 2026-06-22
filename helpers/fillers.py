"""Acoustic filler-cut proposal.

Scribe's filler (呃/嗯/啊/欸) timestamps are unreliable — often near-zero width and parked
in the wrong silence (verified: a 呃 whose sound is at 47.2s gets a token at 49.0s). So we
ignore the token's span and instead locate the actual filler SOUND acoustically, inside the
gap between the two surrounding CONTENT words, on that speaker's own track.

Fillers embedded in continuous speech/stutter (no surrounding silence to bound them) or in
cross-talk are flagged for manual review, not cut — forcing them would remove real content.
"""
import numpy as np
import soundfile as sf

from .render import _protected, verify_no_midword
from .transcribe import speaker_from_filename

_FILLERS = ("呃", "嗯", "啊", "欸")
_THRESH = 0.012     # RMS above this = voiced
_MIN_LEN = 0.05     # ignore voiced runs shorter than this (s)
_MAX_EXTENT = 1.6   # voiced span longer than this => filler embedded in speech => flag
_PAD = 0.04


def _envelope(path, frame=0.01):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    fr = int(frame * sr)
    n = len(x) // fr
    rms = np.sqrt((x[:n * fr].reshape(n, fr) ** 2).mean(axis=1))
    return rms, frame


def voiced_extent(env, a, b, thresh=_THRESH, min_len=_MIN_LEN, pad=_PAD):
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
    ids we couldn't safely cut (embedded in speech / cross-talk / already silent)."""
    words = transcript["words"]
    env = {speaker_from_filename(p): _envelope(p) for p in track_paths}
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
        ext = voiced_extent(env[w["speaker"]], prev[-1]["end"], nxt[0]["start"])
        if ext is None:
            flagged.append({"id": w["id"], "reason": "already silent"})
        elif ext[1] - ext[0] > _MAX_EXTENT or verify_no_midword(list(ext), words):
            flagged.append({"id": w["id"], "reason": "embedded in speech / cross-talk"})
        else:
            cuts.append({"type": "macro", "start": ext[0], "end": ext[1],
                         "reason": f"filler {w['text'].strip()}"})
    return cuts, flagged
