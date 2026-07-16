import numpy as np
import soundfile as sf

from helpers.deadair import find_dead_air, refine_boundaries


def _w(i, s, e, spk="a"):
    return {"id": i, "text": "字", "start": s, "end": e, "speaker": spk}


def _track(path, spans, sr=16000, dur=4.0):
    t = np.arange(int(dur * sr)) / sr
    x = np.zeros_like(t)
    for a, b in spans:
        x[(t >= a) & (t < b)] = (0.3 * np.sin(2 * np.pi * 200 * t))[(t >= a) & (t < b)]
    sf.write(path, x.astype(np.float32), sr)


def test_long_gap_proposed_with_kept_pause_split():
    words = [_w(0, 0.0, 1.0), _w(1, 3.0, 3.5)]  # 2.0s gap
    cuts = find_dead_air(words, min_gap=1.2, keep=0.8)
    assert len(cuts) == 1
    assert cuts[0]["start"] == 1.4 and cuts[0]["end"] == 2.6  # 0.4s left each side


def test_short_gap_ignored():
    words = [_w(0, 0.0, 1.0), _w(1, 2.1, 2.5)]  # 1.1s < 1.2 threshold
    assert find_dead_air(words, min_gap=1.2) == []


def test_gap_is_word_union_across_speakers():
    # b keeps talking through a's silence -> nobody-quiet gap is only 3.0-3.6 (0.6s)
    words = [_w(0, 0.0, 1.0, "a"), _w(1, 0.8, 3.0, "b"), _w(2, 3.6, 4.0, "a")]
    assert find_dead_air(words, min_gap=1.2) == []


def test_gap_with_laughter_is_protected():
    words = [_w(0, 0.0, 1.0), _w(1, 4.0, 4.5)]
    events = [{"type": "laughter", "start": 1.8, "end": 2.6, "speaker": "a"}]
    assert find_dead_air(words, events, min_gap=1.2) == []
    # a non-laughter event (e.g. a muted cough) does not protect the gap
    events = [{"type": "cough", "start": 1.8, "end": 2.2, "speaker": "a"}]
    assert len(find_dead_air(words, events, min_gap=1.2)) == 1


# ── acoustic boundary refinement (token times lie) ───────────────────────────
def test_refine_slides_boundary_off_untokenized_sound(tmp_path):
    # A sustained 呃 (real sound 1.1-2.5s) whose Scribe token is near-zero-width: the
    # token gap proposes a cut starting MID-SOUND (1.55). The mid-word guard can't see
    # it (fillers exempt) and snap can't reach silence (0.95s away) — refinement must
    # slide the start past the real sound.
    p = str(tmp_path / "x--a.wav")
    _track(p, [(0.5, 1.0), (1.1, 2.5), (3.5, 3.9)])
    words = [_w(0, 0.5, 1.0), {"id": 1, "text": "呃", "start": 1.1, "end": 1.15,
                               "speaker": "a"}, _w(2, 3.5, 3.9)]
    cuts = find_dead_air(words)
    assert cuts and cuts[0]["start"] < 2.5          # raw proposal starts inside the sound
    refined = refine_boundaries(cuts, [p])
    assert len(refined) == 1
    assert refined[0]["start"] >= 2.5               # slid past the sound (+pad)
    assert refined[0]["end"] == cuts[0]["end"]      # end already in silence: untouched
    assert refined[0]["snap"] is False              # verified boundaries: render must not re-snap


def test_refine_drops_proposal_that_is_really_sound(tmp_path):
    # tokens promise a gap but the track is voiced straight through -> nothing to cut
    p = str(tmp_path / "x--a.wav")
    _track(p, [(0.5, 3.5)])
    words = [_w(0, 0.5, 1.0), _w(1, 3.5, 3.9)]
    cuts = find_dead_air(words)
    assert len(cuts) == 1
    assert refine_boundaries(cuts, [p]) == []


def test_refine_keeps_truly_silent_gap_untouched(tmp_path):
    # a genuinely silent gap (breath-free here; a breath not touching a boundary also
    # passes) -> the proposal goes through byte-identical, recall unaffected
    p = str(tmp_path / "x--a.wav")
    _track(p, [(0.5, 1.0), (3.5, 3.9)])
    words = [_w(0, 0.5, 1.0), _w(1, 3.5, 3.9)]
    cuts = find_dead_air(words)
    out = refine_boundaries(cuts, [p])
    assert [(c["start"], c["end"]) for c in out] == [(c["start"], c["end"]) for c in cuts]
    assert all(c["snap"] is False for c in out)  # verified boundaries opt out of render's snap
