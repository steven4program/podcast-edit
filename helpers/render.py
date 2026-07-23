"""Render: turn a cut-list into a smooth output audio file.

Pure timeline math (resolve / kept-segments / remap / verify) is the testable core.
Audio is rendered per-track and mixed down: every track is cut at the SAME merged-
timeline boundaries, so the tracks stay aligned and Phase 5 can mute a single track
(e.g. host cough) without touching the others. Single-file input still works.
"""
import json, math, os, re, subprocess, tempfile


# ── pure data core ───────────────────────────────────────────────────────────
def resolve_cut_times(cuts, words):
    by_id = {w["id"]: w for w in words}
    out = []
    for c in cuts:
        if c.get("start_word") is not None:
            try:
                start, end = by_id[c["start_word"]]["start"], by_id[c["end_word"]]["end"]
            except KeyError as e:
                raise ValueError(f"cut cites unknown word id {e.args[0]}: {c}") from None
            # Align the cut END to the next (kept) word's onset. Scribe both micro-overlaps
            # (end_word.end runs past the keep word → clips it) AND under-measures short or
            # broken syllables, parking the trailing '-' marker at zero width (end_word.end
            # falls short → the stutter's sound survives, e.g. a 我 tagged 20ms). Snapping to
            # the keep word's start fixes both; bounded to <0.5s so a real pause isn't eaten.
            nxt = by_id.get(c["end_word"] + 1)
            if nxt is not None and start < nxt["start"] < end + 0.5:
                end = nxt["start"]
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


def verify_mutes(mutes, words, eps=_MIDWORD_EPS):
    """A mute silences one speaker's track — so it must never cover that speaker's OWN
    content words (it would erase them). Same seam guarantee as verify_no_midword, for
    the mute path: split_events enforces this at proposal time, but cuts.json is written
    by hand/LLM, so render re-checks. Fillers/punctuation don't block (they're removable),
    mirroring _protected."""
    bad = []
    for m in mutes or []:
        hit = next((w for w in words
                    if w.get("speaker") == m["speaker"] and _protected(w)
                    and w["start"] + eps < m["end"] and m["start"] < w["end"] - eps), None)
        if hit is not None:
            bad.append({"mute": (m["speaker"], m["start"], m["end"]),
                        "word": (hit["id"], hit["text"])})
    return bad


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


# Per-track speech leveling. Everything here is STATIC (a constant gain and a fixed
# input->output level curve), which is the load-bearing property: any time-varying,
# silence-gated leveler manufactures a fade-out at every sentence end. dynaudnorm was
# measured doing exactly that (a 4.6 dB natural phrase tail became a 28-36 dB fade):
# its per-frame gain takes the MINIMUM over a ~3s window, the silent frames after a
# phrase are gated to unity gain, and the min-filter drags that unity back through the
# last ~1.5s of speech. Lowering the threshold doesn't help (silence is below any
# usable threshold); removing it boosts the noise floor ~20 dB. So:
# 1. speech_gain() — one constant volume per track aligns each speaker's typical
#    speech level (evens speaker gaps; replaces dynaudnorm's cross-track job).
# 2. compand, downward compression ONLY: 4:1 above the -32 dB knee squeezes sudden
#    loud bursts; below the knee the curve is exactly 1:1, so a decaying tail falls at
#    its natural rate — compression-only can never steepen a fade (a curve that BOOSTS
#    quiet speech must slope >1 below the boost region, and tails passing through that
#    zone plunge; a -35->-18 boost curve measurably doubled tail decay).
# 3. mastering (in render) anchors the mix at -16 LUFS with ONE measured static gain
#    (see _master_gain — in-graph single-pass loudnorm is itself dynamic).
_LEVEL = "compand=attacks=0.05:decays=0.3:soft-knee=6:points=-80/-80|-32/-32|0/-24"
_TARGET_SPEECH_DB = -23.0


def frame_rms(path, frame):
    """Per-frame RMS envelope of a track, streamed in blocks — a 2h 48kHz track must
    never sit in RAM whole (float64 of it is >1GB). A partial frame at a block edge is
    carried into the next block, so the envelope is identical to a whole-file read
    (only the final sub-frame tail is dropped, as before). -> (rms array, samplerate)."""
    import numpy as np, soundfile as sf
    out, rem = [], None
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        fr = int(frame * sr)
        for block in f.blocks(blocksize=fr * 2048, dtype="float64"):
            block = block if block.ndim == 1 else block.mean(1)
            if rem is not None and len(rem):
                block = np.concatenate([rem, block])
            n = len(block) // fr
            if n:
                out.append(np.sqrt((block[:n * fr].reshape(n, fr) ** 2).mean(1)))
            rem = block[n * fr:]
    return (np.concatenate(out) if out else np.empty(0)), sr


def speech_gain(path, target_db=_TARGET_SPEECH_DB, clamp_db=12.0):
    """Static per-track gain (dB) aligning this speaker's typical speech level to
    target: 70th percentile of 50ms RMS frames above -45 dBFS (his own speech; the
    track is silent when he isn't talking). Clamped so a near-silent or clipping
    track can't produce a wild gain."""
    import numpy as np
    rms, _sr = frame_rms(path, 0.05)
    if not len(rms):
        return 0.0
    voiced = rms[rms > 10 ** (-45 / 20)]
    if not len(voiced):
        return 0.0
    level = 20 * np.log10(float(np.percentile(voiced, 70)))
    return round(max(-clamp_db, min(clamp_db, target_db - level)), 1)


def _mute_pieces(spans, duration):
    """Partition [0, duration] into alternating (start, end, muted) pieces around the
    mute spans (sorted, overlaps folded, clamped). The muted track is rebuilt from
    these pieces so its length is unchanged and every mute edge gets a real fade."""
    pieces, pos = [], 0.0
    for s, e in sorted(spans):
        s, e = max(s, pos, 0.0), min(e, duration)
        if s > pos:
            pieces.append((pos, s, False))
        if e > s:
            pieces.append((s, e, True))
        pos = max(pos, e)
    if pos < duration:
        pieces.append((pos, duration, False))
    return pieces


def _filtergraph(n_tracks, segments, track_mutes=None, track_gains=None):
    """Cut every track at `segments` (3ms fades per piece), concat + level per track,
    then mix to [out]. track_mutes[ti] = [(s,e),...] silences that track in those
    original-timeline spans (the multitrack cough superpower: kill a cough on the host's
    track without removing time or touching other speakers). track_gains[ti] = static dB
    from speech_gain(), applied ahead of the compand stage. Mastering (loudness anchor)
    is NOT here — it is a separate measured static pass in render(); see _master_gain."""
    track_mutes = track_mutes or {}
    track_gains = track_gains or {}
    parts, track_labels = [], []
    n_seg = len(segments)
    for ti in range(n_tracks):
        # Each segment's atrim needs its own source pad. A raw input pad ([ti:a]) can be
        # referenced by many filters (ffmpeg auto-splits it), but a FILTER-OUTPUT label can
        # only feed one consumer — so when we mute a track, its volume=0 output must be
        # asplit into one copy per segment. Reusing a single [m{ti}] label across every atrim
        # silently drops most of that track in a multi-input graph (host vanishes from the mix).
        if track_mutes.get(ti):
            # A mute must not click: volume=0:enable=… is a hard gain step, and a
            # volume EXPRESSION only updates once per frame (a 3ms ramp quantizes into
            # 2-3 audible steps — measured). So rebuild the muted track structurally,
            # with the same idiom as the cut segments: atrim pieces, 3ms afades on the
            # speech pieces, volume=0 on the muted pieces, concat back to full length
            # (sample counts add back exactly — no timeline drift).
            total = max(e for _, e in segments)
            plabels = []
            for pi, (s, e, muted) in enumerate(_mute_pieces(track_mutes[ti], total)):
                lbl = f"mp{ti}_{pi}"
                fx = ("volume=0" if muted else
                      f"afade=t=in:st=0:d=0.003,afade=t=out:st={max(e - s - 0.003, 0)}:d=0.003")
                parts.append(f"[{ti}:a]atrim={s}:{e},asetpts=PTS-STARTPTS,{fx}[{lbl}]")
                plabels.append(f"[{lbl}]")
            outs = "".join(f"[m{ti}_{si}]" for si in range(n_seg))
            parts.append("".join(plabels) + f"concat=n={len(plabels)}:v=0:a=1,"
                         f"asplit={n_seg}{outs}")
            seg_srcs = [f"[m{ti}_{si}]" for si in range(n_seg)]
        else:
            seg_srcs = [f"[{ti}:a]"] * n_seg
        seg_labels = []
        for si, (s, e) in enumerate(segments):
            lbl = f"a{ti}_{si}"
            parts.append(f"{seg_srcs[si]}atrim={s}:{e},asetpts=PTS-STARTPTS,"
                         f"afade=t=in:st=0:d=0.003,"
                         f"afade=t=out:st={max(e - s - 0.003, 0)}:d=0.003[{lbl}]")
            seg_labels.append(f"[{lbl}]")
        lvl = f"volume={track_gains.get(ti, 0.0)}dB,{_LEVEL}"
        parts.append("".join(seg_labels) + f"concat=n={len(segments)}:v=0:a=1,{lvl}[t{ti}]")
        track_labels.append(f"[t{ti}]")
    if n_tracks == 1:
        parts.append(f"{track_labels[0]}anull[out]")
    else:
        parts.append("".join(track_labels) + f"amix=inputs={n_tracks}:normalize=0[out]")
    return ";".join(parts)


# ── mastering: measured STATIC gain + transient limiter, never in-graph loudnorm ──
# Single-pass loudnorm is a DYNAMIC normalizer (time-varying gain — the exact class of
# processing the leveling comment above rules out) and it upsamples to 192kHz internally
# without downsampling back (measured: a wav out of `-af loudnorm` is 192kHz; mp3 only
# escapes because the encoder can't take it). So render measures loudness on the finished
# mix, then applies ONE constant volume to hit -16 LUFS integrated, with alimiter as the
# peak safety. Natural speech+laughter has a ~28dB crest factor (measured on a real
# episode: isolated laughter/plosive transients), so a static gain alone can't satisfy
# -16 LUFS AND the peak ceiling — capping the gain at the ceiling left a whole episode
# 13dB under target. The limiter is time-varying, but only on millisecond transients:
# no gate, no windowed minimum, nothing that can manufacture a phrase-tail fade
# (test_render_does_not_steepen_fading_tail verifies that empirically).
_TARGET_I = -16.0  # integrated loudness target (podcast standard)
_LIMIT_DB = -2.0  # sample-peak ceiling; ~0.3dB margin keeps TRUE peak under -1.5 dBTP
_MASTER = f"alimiter=limit={10 ** (_LIMIT_DB / 20):.4f}:attack=5:release=50:level=false"


def measure_loudness(path):
    """-> (integrated LUFS, true peak dBTP) via ffmpeg loudnorm's measurement JSON."""
    proc = subprocess.run(
        ["ffmpeg", "-i", path, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
         "-f", "null", "-"], capture_output=True, text=True)
    try:
        i = proc.stderr.rindex("{")
        data = json.loads(proc.stderr[i:proc.stderr.index("}", i) + 1])
        return float(data["input_i"]), float(data["input_tp"])
    except (ValueError, KeyError):
        raise RuntimeError("loudnorm measurement failed:\n"
                           + "\n".join(proc.stderr.splitlines()[-10:]))


def _master_gain(input_i, target_i=_TARGET_I):
    """Static mastering gain (dB) to the integrated-loudness target. Peaks are the
    limiter's job, not this gain's. A non-finite measurement (silence measures -inf)
    leaves the audio alone."""
    if not math.isfinite(input_i):
        return 0.0
    return round(target_i - input_i, 2)


def _speaker(path):
    return os.path.splitext(os.path.basename(path))[0].split("--")[-1]


def _track_mutes(sources, mutes):
    out = {}
    for m in mutes or []:
        for ti, src in enumerate(sources):
            if _speaker(src) == m["speaker"]:
                out.setdefault(ti, []).append((m["start"], m["end"]))
    return out


def _ffmpeg(args):
    # check=True + capture_output swallows ffmpeg's actual error into an opaque exit
    # code; surface the stderr tail so a failed render is diagnosable.
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}):\n"
                           + "\n".join(proc.stderr.splitlines()[-15:]))


def _run_ffmpeg(sources, fc, out_path, extra=()):
    # Many cuts -> a long filtergraph. Passed inline it can exceed the OS command-line
    # limit (Windows CreateProcess caps at ~32k chars), so hand it to ffmpeg via a script
    # file (-filter_complex_script) instead of an argv string. Portable, no length ceiling.
    fd, fc_path = tempfile.mkstemp(suffix=".ffscript")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(fc)
    inputs = [x for src in sources for x in ("-i", src)]
    try:
        _ffmpeg(["ffmpeg", "-y", *inputs, "-filter_complex_script", fc_path,
                 "-map", "[out]", *extra, out_path])
    finally:
        os.remove(fc_path)


def _render_mix(sources, segments, track_mutes, track_gains, mix_path):
    """Cut + level + mix to the float intermediate at `mix_path` (pcm_f32le).

    Renders each track through its OWN small filtergraph and amixes the float stems, instead
    of one monster graph. One graph of n_tracks x len(segments) atrim/afade nodes drives
    ffmpeg's per-frame scheduler super-linearly (measured: a 525-cut, 4-track episode ~5.5h
    and climbing); n graphs of ~len(segments) nodes each run in minutes. The output is
    numerically identical (test_per_track_mix_matches_monolith null-tests it to <1e-6): each
    stem is exactly the [t{ti}] the monster graph builds — the SAME _filtergraph(1, ...) with
    that track's mute/gain — the stems are float so re-reading them is lossless, and the amix
    params (normalize=0) match. Mastering stays in render()."""
    n = len(sources)
    if n == 1:                      # already a small graph; no stems to gain from splitting
        fc = _filtergraph(1, segments, track_mutes, track_gains)
        _run_ffmpeg(sources, fc, mix_path, extra=("-c:a", "pcm_f32le"))
        return
    stems = []
    try:
        for ti, src in enumerate(sources):
            fd, stem = tempfile.mkstemp(suffix=".wav"); os.close(fd)
            stems.append(stem)
            fc = _filtergraph(1, segments,
                              {0: track_mutes[ti]} if track_mutes and ti in track_mutes else None,
                              {0: track_gains.get(ti, 0.0)})
            _run_ffmpeg([src], fc, stem, extra=("-c:a", "pcm_f32le"))
        inputs = [x for s in stems for x in ("-i", s)]
        _ffmpeg(["ffmpeg", "-y", *inputs, "-filter_complex",
                 f"amix=inputs={n}:normalize=0[out]", "-map", "[out]",
                 "-c:a", "pcm_f32le", mix_path])
    finally:
        for s in stems:
            if os.path.exists(s):
                os.remove(s)


def render(transcript, cuts, audio_path, out_path, snap_window=0.3, mutes=None,
           track_gains=None):
    words = transcript["words"]
    sources = transcript.get("tracks") or [audio_path]
    # transcript duration is derived from word timestamps, which end at the last word —
    # trust the actual audio length so the tail (room tone, decay past the last word,
    # anything Scribe didn't tokenize) survives into the output.
    duration = max(transcript["duration"], *(probe_duration(s) for s in sources))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)  # work/ or out/
    resolved = resolve_cut_times(cuts, words)

    bad_mutes = verify_mutes(mutes, words)
    if bad_mutes:
        raise ValueError(f"mutes cover the speaker's own words: {bad_mutes}")

    silences = _snap_silences(sources, out_path)
    snapped_count = 0
    for c in resolved:
        # Word-id cuts (repeats/stutters) are already on word edges — snapping a short
        # intra-speech cut to a nearby silence drags it onto adjacent words. The mid-word
        # guard + 3ms fades keep the seam clean. Only hand-placed segment cuts snap:
        # a cut carrying "snap": false has ACOUSTICALLY determined boundaries (fillers'
        # voiced extent, deadair's refined edges — found on the speaker's own track with
        # a noise-floor-relative threshold). Re-snapping those to the mix's silencedetect
        # edges (absolute -35dB, 0.3s minimum — much cruder) drags the boundary back
        # INTO the sound: measured 36-222ms of a removed 呃 surviving into the output
        # while the transcript showed it struck through.
        if c.get("start_word") is not None or c.get("snap") is False:
            continue
        ns, s1 = safe_snap(c["start"], silences, words, snap_window)
        ne, s2 = safe_snap(c["end"], silences, words, snap_window)
        if ne - ns < 0.5 * (c["end"] - c["start"]):  # both ends snapped to the same nearby
            ns, ne, s1, s2 = c["start"], c["end"], False, False  # edge → would erase the cut
        c["start"], c["end"] = ns, ne                            # (the sound); keep the span
        snapped_count += int(s1) + int(s2)

    segments = compute_kept_segments(resolved, duration)
    if not segments:
        raise ValueError("cuts remove the entire audio")
    boundaries = [b for seg in segments for b in seg if 0 < b < duration]
    flagged = verify_no_midword(boundaries, words)
    if flagged:
        raise ValueError(f"mid-word boundaries after snap: {flagged}")

    track_mutes = _track_mutes(sources, mutes)
    if track_gains is None:
        track_gains = {ti: speech_gain(src) for ti, src in enumerate(sources)}
    # Cut+level+mix to a float intermediate (no clipping before mastering), measure,
    # then apply ONE static gain — see the mastering comment above _master_gain.
    fd, mix = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        _render_mix(sources, segments, track_mutes, track_gains, mix)
        gain = _master_gain(measure_loudness(mix)[0])
        for _ in range(2):
            _ffmpeg(["ffmpeg", "-y", "-i", mix, "-af",
                     f"volume={gain}dB,{_MASTER}", out_path])
            got = measure_loudness(out_path)[0]
            # heavy limiting (a laughter-dense episode) eats into integrated loudness;
            # fold the shortfall back into the (still constant) gain and re-run once
            if not math.isfinite(got) or abs(_TARGET_I - got) <= 0.5:
                break
            gain = round(gain + (_TARGET_I - got), 2)
    finally:
        os.remove(mix)

    kept = {"words": remap_words(words, segments), "segments": segments}
    kept_path = os.path.splitext(out_path)[0] + "_kept_transcript.json"  # robust to non-.mp3 out
    with open(kept_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    return {"segments": segments, "snapped": snapped_count, "flagged": flagged,
            "master_gain_db": gain}


def render_stems(transcript, cuts, audio_path, out_dir, snap_window=0.3, mutes=None):
    """Stem export of the approved cut: out_dir/final.<ext> (the integrated mix) plus one
    final_<speaker>.<ext> per source track, in the SOURCE tracks' format (wav in → wav out,
    mp3 in → mp3 out). Every file is cut at the SAME boundaries and the host's mutes apply
    to his stem too, so the stems stay time-aligned with the mix. Stems keep the per-track
    speech leveling but skip loudnorm — normalizing each stem alone would shift the
    speaker balance when they are re-mixed downstream."""
    sources = transcript.get("tracks") or [audio_path]
    ext = os.path.splitext(sources[0])[1].lower() or ".wav"
    os.makedirs(out_dir, exist_ok=True)
    mix_path = os.path.join(out_dir, f"final{ext}")
    gains = {ti: speech_gain(src) for ti, src in enumerate(sources)}  # once — the stems
    res = render(transcript, cuts, audio_path, mix_path, snap_window, mutes,
                 track_gains=gains)                                   # reuse it below
    track_mutes = _track_mutes(sources, mutes)
    outputs = {"mix": mix_path}
    for ti, src in enumerate(sources):
        stem = os.path.join(out_dir, f"final_{_speaker(src)}{ext}")
        fc = _filtergraph(1, res["segments"],
                          {0: track_mutes[ti]} if ti in track_mutes else None,
                          {0: gains[ti]})
        _run_ffmpeg([src], fc, stem)
        outputs[_speaker(src)] = stem
    return {**res, "outputs": outputs}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("transcript"); p.add_argument("cuts"); p.add_argument("out")
    p.add_argument("--audio", help="single-file source (ignored if transcript has tracks)")
    p.add_argument("--stems", action="store_true",
                   help="OUT is a directory: write final.<ext> (mix) + one stem per speaker, "
                        "in the source tracks' format (wav in -> wav out, mp3 in -> mp3 out)")
    a = p.parse_args()
    t = json.load(open(a.transcript, encoding="utf-8"))
    spec = json.load(open(a.cuts, encoding="utf-8"))  # {"cuts": [...], "mutes": [...]} (mutes optional)
    if a.stems:
        res = render_stems(t, spec["cuts"], a.audio, a.out, mutes=spec.get("mutes"))
    else:
        res = render(t, spec["cuts"], a.audio, a.out, mutes=spec.get("mutes"))
    print(json.dumps(res, ensure_ascii=False))
