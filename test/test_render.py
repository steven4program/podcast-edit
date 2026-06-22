import json, os

import helpers.render as r
from test.fixtures import sample_transcript, make_test_wav, make_two_tracks


# ── Task 2.1: pure data core ─────────────────────────────────────────────────
def test_resolve_word_cut_uses_exact_word_times():
    cuts = [{"type": "micro", "start_word": 0, "end_word": 0, "reason": "repeat"}]
    out = r.resolve_cut_times(cuts, sample_transcript()["words"])
    assert out[0]["start"] == 0.0 and out[0]["end"] == 0.4


def test_resolve_event_cut_keeps_times():
    cuts = [{"type": "event", "start": 1.6, "end": 1.9, "reason": "cough"}]
    out = r.resolve_cut_times(cuts, sample_transcript()["words"])
    assert out[0]["start"] == 1.6 and out[0]["end"] == 1.9


def test_kept_segments_is_complement():
    cuts = [{"start": 0.0, "end": 0.4}, {"start": 1.6, "end": 1.9}]
    assert r.compute_kept_segments(cuts, 4.0) == [(0.4, 1.6), (1.9, 4.0)]


def test_kept_segments_merges_overlap():
    cuts = [{"start": 0.5, "end": 1.0}, {"start": 0.8, "end": 1.2}]
    assert r.compute_kept_segments(cuts, 2.0) == [(0.0, 0.5), (1.2, 2.0)]


def test_remap_words_shifts_onto_output_timeline():
    kept = r.remap_words(sample_transcript()["words"], [(0.4, 4.0)])
    assert kept[0]["id"] == 1
    assert abs(kept[0]["start"] - 0.0) < 1e-9
    assert abs(kept[-1]["end"] - (3.7 - 0.4)) < 1e-9


def test_verify_flags_midword_boundary():
    words = sample_transcript()["words"]
    assert r.verify_no_midword([0.2], words) == [0.2]
    assert r.verify_no_midword([0.4], words) == []


# ── Task 2.2: snap (pure) ────────────────────────────────────────────────────
def test_snap_point_moves_to_silence_within_window():
    new_t, snapped = r.snap_point(1.45, [(1.5, 2.2)], window=0.3)
    assert snapped and abs(new_t - 1.5) < 1e-9


def test_snap_point_keeps_when_no_silence_in_window():
    new_t, snapped = r.snap_point(0.5, [(1.5, 2.2)], window=0.3)
    assert not snapped and new_t == 0.5


# ── Task 2.2: render (single-file) ───────────────────────────────────────────
def test_render_output_duration_matches_kept(tmp_path):
    wav, out = str(tmp_path / "in.wav"), str(tmp_path / "out.mp3")
    make_test_wav(wav)
    t = sample_transcript(); t["audio"] = wav
    cuts = [{"type": "micro", "start_word": 0, "end_word": 0, "reason": "repeat 我覺得"}]
    result = r.render(t, cuts, wav, out)
    assert os.path.exists(out)
    kept = json.load(open(out.replace(".mp3", "_kept_transcript.json")))
    assert kept["words"][0]["id"] == 1
    expected = sum(e - s for s, e in result["segments"])
    assert abs(r.probe_duration(out) - expected) < 0.3  # no drift / no extra audio


# ── Task 2.2: render (multitrack — per-track cut then mix) ────────────────────
def test_render_multitrack_cuts_each_track_and_mixes(tmp_path):
    a, b, out = str(tmp_path / "a.wav"), str(tmp_path / "b.wav"), str(tmp_path / "out.mp3")
    make_two_tracks(a, b)
    t = sample_transcript(); t.pop("audio", None); t["tracks"] = [a, b]
    cuts = [{"type": "micro", "start_word": 0, "end_word": 0, "reason": "repeat"}]
    result = r.render(t, cuts, None, out)
    assert os.path.exists(out)
    expected = sum(e - s for s, e in result["segments"])
    assert abs(r.probe_duration(out) - expected) < 0.3
