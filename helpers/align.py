"""Re-time Scribe's words with MMS forced alignment (WIP skeleton — not yet wired into the
pipeline). Scribe gets the TEXT right but its word timestamps collapse on fast / run-together
speech (validated: 矽谷's sound is at 54.1 but Scribe parks it at 55.15, so a filler cut
swallows it). Forced alignment re-times the KNOWN text against the audio — the constraint
(text is given) is what makes the times accurate; see docs/alignment.md.

TOOL: torchaudio's MMS_FA model + torchaudio.functional.forced_align — pure pip, cross-platform.
(ctc-forced-aligner needs an MSVC C++ build on Windows.) Chinese -> toneless pinyin (pypinyin),
English kept as latin, both align against MMS's romanized latin token set.
CAVEAT: torchaudio.functional.forced_align is deprecated in torchaudio 2.8 (removed in 2.9) —
pin torchaudio<2.9 or migrate the emission->path step when we productionize.

TRUST is NOT a fixed confidence threshold. Speaking rate and loudness make raw alignment
confidence unreliable across speakers, exactly like a fixed ms/char threshold is unreliable
(see repeats collapse-guard). Instead the decision is physically grounded and speaker-relative:
keep whichever of {aligned, scribe} time lands on the speaker's OWN voiced energy, where
"voiced" is relative to that track's noise floor (fillers._voiced_thresh) — so it holds
whether they talk fast/slow or loud/soft. Hard sanity gates (monotonic, in-bounds, plausible
duration) catch gross failures with no tunable number. Confidence is used ONLY to rank what to
surface for review, never as a silent on/off gate.
"""
import re

from .fillers import _voiced_thresh, frame_rms
from .transcribe import speaker_from_filename, voiced_segments, _detect_silence, \
    _silence_thresh_db, _duration, _clamp_word_end

_PUNCT = set("，。、！？…—–-　 .,!?；：\"'「」（）()")
_MIN_ENERGY = 0.3     # a real word span is mostly voiced; below this the aligned time is on
                      # silence/noise → the alignment is invalid there, fall back to Scribe
_DRIFT = 0.3          # s; only surface an aligned-vs-scribe move larger than this for review


# ── romanization (Chinese -> pinyin, English -> latin), model-agnostic ────────
def _romanize(token):
    """Token text -> latin string for the aligner. Hanzi -> toneless pinyin, an English word
    -> lowercase; punctuation / non-spoken -> '' (caller skips or interpolates)."""
    t = token.strip()
    if not t or t in _PUNCT:
        return ""
    if re.search(r"[A-Za-z]", t):
        return re.sub(r"[^a-z]", "", t.lower())
    import pypinyin
    return re.sub(r"[^a-z]", "", "".join(pypinyin.lazy_pinyin(t)))


# ── the trust decision (pure, speaker-relative, testable without torch) ───────
def _voiced_frac(rms, step, a, b, thresh):
    seg = rms[int(max(0, a) / step):int(max(0, b) / step)]
    return float((seg > thresh).mean()) if len(seg) else 0.0


def reconcile(words, aligned, rms, step, thresh):
    """Choose aligned vs Scribe time per word by which lands on the speaker's real voiced
    energy (speed/loudness-invariant). `aligned` is a list of (start, end, conf) parallel to
    `words`, or None where the aligner produced nothing. Returns (retimed_words, report) where
    report is [(word_id, note)] for the review page. Only start/end/`t_src`/`conf` change."""
    out, report, prev = [], [], -1.0
    for w, al in zip(words, aligned):
        s_s, s_e = w["start"], w["end"]
        take_scribe = lambda why: (report.append((w["id"], why)) if why else None) or \
            {**w, "start": s_s, "end": s_e, "t_src": "scribe"}
        if al is None:
            out.append(take_scribe("no alignment")); prev = s_s; continue
        a_s, a_e, conf = al
        sane = (a_e > a_s >= 0 and a_s >= prev - 1e-6
                and a_e - a_s <= _clamp_word_end(w["text"], 0.0, 1e9))
        # small pad so a sub-100ms span still samples its neighbouring voiced frames
        a_energy = _voiced_frac(rms, step, a_s - 0.03, a_e + 0.03, thresh) if sane else 0.0
        # DEFAULT to alignment (accurate on average); fall back to Scribe ONLY when the aligned
        # time is physically INVALID — off the speaker's voice, out of order, or an implausible
        # duration. We do NOT require alignment to "beat" Scribe on energy: a collapsed Scribe
        # time often lands on a NEIGHBOUR's sound (谷 parked at 55.15 hits 最's onset), so both
        # look voiced and a tug-of-war keeps the wrong one. No confidence number is gated on.
        if sane and a_energy >= _MIN_ENERGY:
            note = f"re-timed {abs(a_s - s_s) * 1000:.0f}ms off Scribe" if abs(a_s - s_s) > _DRIFT else ""
            out.append({**w, "start": a_s, "end": a_e, "t_src": "align",
                        "conf": round(conf, 2), **({"review": note} if note else {})})
            if note:
                report.append((w["id"], note))
        else:
            out.append(take_scribe("alignment invalid (off-voice / order / duration) — kept Scribe"))
        prev = out[-1]["start"]
    return out, report


# ── orchestration (needs torch; guarded) ─────────────────────────────────────
def _align_segment(wave, tokens, model, dct):
    """Force-align one clip. `wave` is a 1xN 16kHz tensor, `tokens` the romanized strings for
    the clip. -> list of (start_s, end_s, conf) parallel to tokens (None where empty)."""
    import torch, torchaudio
    flat, lens = [], []
    for r in tokens:
        ids = [dct[c] for c in r if c in dct]
        lens.append(len(ids))
        flat += ids or [dct["*"]]
    with torch.inference_mode():
        emission, _ = model(wave)
    aligned, scores = torchaudio.functional.forced_align(
        emission, torch.tensor([flat], dtype=torch.int32), blank=0)
    spans = torchaudio.functional.merge_tokens(aligned[0], scores[0].exp())
    ratio = wave.size(1) / emission.size(1) / 16000
    res, i = [], 0
    for r, n in zip(tokens, lens):
        n = n or 1
        grp = spans[i:i + n]; i += n
        res.append(None if not r else
                   (grp[0].start * ratio, grp[-1].end * ratio,
                    sum(s.score for s in grp) / len(grp)))
    return res


_CHUNK_GAP = 1.0   # split only at silences longer than this (coarse chunks)


def _coarse_chunks(track_path, dur):
    """Speech runs split ONLY at long (> _CHUNK_GAP) silences. Forced alignment must consume
    the WHOLE clip with the tokens it is given, so a chunk has to contain a COMPLETE run of
    speech with ALL its words — a fine VAD segment that clips a phrase mid-way gets a partial
    token list and the aligner stretches those tokens across the whole clip (measured: 9 tokens
    smeared over a 28-word span). Long silences make word->chunk assignment by Scribe time safe:
    collapse errors are sub-second, far smaller than the gap, so a word's real and collapsed
    times land in the same chunk."""
    fine = voiced_segments(_detect_silence(track_path, dur, _silence_thresh_db(track_path)), dur)
    chunks = []
    for s, e in fine:
        if chunks and s - chunks[-1][1] < _CHUNK_GAP:
            chunks[-1] = (chunks[-1][0], e)
        else:
            chunks.append((s, e))
    return chunks


def _load_model():
    """MMS forced-alignment model + its romanized token dict. torch/torchaudio/pypinyin are
    core deps (alignment is a required transcription step); this hint only fires on a broken
    install."""
    try:
        import warnings, torchaudio, pypinyin  # noqa: F401  (pypinyin used by _romanize)
        warnings.filterwarnings("ignore")       # torchaudio.forced_align is deprecated in 2.8
    except ImportError as e:
        raise SystemExit(f"alignment dep missing ({e.name}) — reinstall: pip install -e \".[dev]\"")
    bundle = torchaudio.pipelines.MMS_FA
    return bundle.get_model(), bundle.get_dict()


def align_track(track_path, words, model=None, dct=None):
    """Re-time one speaker's `words` against their track. Aligns per coarse chunk (a complete
    speech run with all its tokens — see _coarse_chunks), reconciles against the speaker's own
    energy. -> (retimed_words, report). Pass a shared (model, dct) to avoid reloading per track."""
    import torchaudio
    if model is None:
        model, dct = _load_model()
    dur = _duration(track_path)
    segs = _coarse_chunks(track_path, dur)
    rms, sr = frame_rms(track_path, 0.01)
    thr, step = _voiced_thresh(rms), 0.01
    wave, wsr = torchaudio.load(track_path)
    if wsr != 16000:
        wave = torchaudio.functional.resample(wave, wsr, 16000)
    aligned = [None] * len(words)
    for s, e in segs:                                   # assign words to their voiced segment
        idx = [i for i, w in enumerate(words) if s <= w["start"] < e]
        if not idx:
            continue
        clip = wave[:, int(s * 16000):int(e * 16000)]
        toks = [_romanize(words[i]["text"]) for i in idx]
        try:
            seg = _align_segment(clip, toks, model, dct)
        except Exception:
            continue                      # one bad chunk -> those words fall back to Scribe
        for i, al in zip(idx, seg):
            aligned[i] = None if al is None else (al[0] + s, al[1] + s, al[2])
    return reconcile(words, aligned, rms, step, thr)


def align_transcript(transcript):
    """Re-time every speaker in place, then re-sort + re-id (id == time order, like
    merge_tracks). -> (transcript, report). Text/events untouched; only word times change."""
    tracks = {speaker_from_filename(p): p for p in transcript.get("tracks", [])}
    model, dct = _load_model()                       # load once, reuse across tracks
    report = []
    for spk, path in tracks.items():
        ws = [w for w in transcript["words"] if w["speaker"] == spk]
        retimed, rep = align_track(path, ws, model, dct)
        by_id = {w["id"]: w for w in retimed}
        for w in transcript["words"]:
            if w["id"] in by_id:
                w.update(by_id[w["id"]])
        report += rep
    transcript["words"].sort(key=lambda w: (w["start"], w["speaker"] or ""))
    for i, w in enumerate(transcript["words"]):
        w["id"] = i
    return transcript, report


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser(description="Re-time transcript words via MMS forced alignment.")
    p.add_argument("transcript")
    p.add_argument("--write", action="store_true", help="overwrite the transcript in place")
    a = p.parse_args()
    t = json.load(open(a.transcript, encoding="utf-8"))
    t, report = align_transcript(t)
    if a.write:
        json.dump(t, open(a.transcript, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({"retimed": sum(w.get("t_src") == "align" for w in t["words"]),
                      "kept_scribe": sum(w.get("t_src") == "scribe" for w in t["words"]),
                      "review": len(report)}, ensure_ascii=False, indent=2))
