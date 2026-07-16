import json, os

import helpers.render as r
from test.fixtures import sample_transcript, make_test_wav, make_two_tracks


# ── Task 2.1: pure data core ─────────────────────────────────────────────────
def test_resolve_word_cut_uses_exact_word_times():
    cuts = [{"type": "micro", "start_word": 0, "end_word": 0, "reason": "repeat"}]
    out = r.resolve_cut_times(cuts, sample_transcript()["words"])
    assert out[0]["start"] == 0.0 and out[0]["end"] == 0.4


def test_resolve_word_cut_clamps_to_next_word_on_overlap():
    # Scribe spans micro-overlap: end_word ends at 0.42 but the next (kept) word
    # starts at 0.40. The cut must stop at 0.40 so word 1 survives remap.
    words = [{"id": 0, "text": "你", "start": 0.0, "end": 0.42, "speaker": "a"},
             {"id": 1, "text": "你好", "start": 0.40, "end": 0.9, "speaker": "a"}]
    out = r.resolve_cut_times([{"start_word": 0, "end_word": 0}], words)
    assert out[0]["end"] == 0.40


def test_resolve_word_cut_extends_to_keep_onset_on_undermeasure():
    # Scribe under-measures the broken syllable (我 = 20ms) and parks '-' at zero width.
    # The cut must extend to the keep word's onset (0.30) so the stutter's sound is removed.
    words = [{"id": 0, "text": "我", "start": 0.20, "end": 0.22, "speaker": "a"},
             {"id": 1, "text": "-", "start": 0.22, "end": 0.22, "speaker": "a"},
             {"id": 2, "text": "我以前", "start": 0.30, "end": 0.8, "speaker": "a"}]
    out = r.resolve_cut_times([{"start_word": 0, "end_word": 1}], words)
    assert out[0]["end"] == 0.30


def test_resolve_word_cut_keeps_real_pause_after_end():
    # A genuine >0.5s pause after end_word is NOT swallowed (next word is the continuation).
    words = [{"id": 0, "text": "對", "start": 0.0, "end": 0.4, "speaker": "a"},
             {"id": 1, "text": "然後", "start": 1.2, "end": 1.6, "speaker": "a"}]
    out = r.resolve_cut_times([{"start_word": 0, "end_word": 0}], words)
    assert out[0]["end"] == 0.4


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


def test_verify_ignores_punctuation_token():
    # Scribe can give a punctuation token a junk duration; you can always cut at punctuation
    words = [{"id": 0, "text": "好", "start": 0.0, "end": 0.5, "speaker": "a"},
             {"id": 1, "text": "？", "start": 1.0, "end": 50.0, "speaker": "b"}]
    assert r.verify_no_midword([25.0], words) == []


def test_safe_snap_rejects_snap_into_word():
    words = [{"id": 0, "text": "你好", "start": 1.4, "end": 1.6, "speaker": "a"}]
    nt, snapped = r.safe_snap(1.45, [(1.5, 2.0)], words, window=0.3)  # 1.5 is inside the word
    assert not snapped and nt == 1.45


def test_safe_snap_allows_clean_snap():
    words = [{"id": 0, "text": "你好", "start": 0.0, "end": 0.5, "speaker": "a"}]
    nt, snapped = r.safe_snap(1.45, [(1.5, 2.0)], words, window=0.3)
    assert snapped and abs(nt - 1.5) < 1e-9


# ── Task 2.2: snap (pure) ────────────────────────────────────────────────────
def test_snap_point_moves_to_silence_within_window():
    new_t, snapped = r.snap_point(1.45, [(1.5, 2.2)], window=0.3)
    assert snapped and abs(new_t - 1.5) < 1e-9


def test_snap_point_keeps_when_no_silence_in_window():
    new_t, snapped = r.snap_point(0.5, [(1.5, 2.2)], window=0.3)
    assert not snapped and new_t == 0.5


def test_filler_cut_not_collapsed_by_snap(tmp_path):
    # An acoustic filler (呃) sitting just inside silence: both ends would snap to the
    # same edge and erase the cut, leaving the sound. The guard keeps the resolved span.
    wav, out = str(tmp_path / "in.wav"), str(tmp_path / "out.mp3")
    make_test_wav(wav)  # tone 0-1.5s, silence 1.5-2.2s, tone 2.2-3.7s
    t = {"audio": wav, "duration": 4.0, "words": [
        {"id": 0, "text": "好", "start": 0.5, "end": 1.40, "speaker": "a"},
        {"id": 1, "text": "呃", "start": 1.55, "end": 1.80, "speaker": "a"},  # in the silence
        {"id": 2, "text": "對", "start": 2.3, "end": 2.9, "speaker": "a"}]}
    res = r.render(t, [{"type": "macro", "start": 1.55, "end": 1.80, "reason": "filler 呃"}], wav, out)
    removed = 4.0 - sum(e - s for s, e in res["segments"])
    assert removed > 0.15  # the ~0.25s filler span was actually removed, not snapped away


def test_render_snap_false_cut_keeps_acoustic_boundaries(tmp_path):
    # A filler cut's extent is acoustically exact (speaker's own track, floor-relative
    # threshold). render's snap works off the MIX's -35dB silencedetect and dragged
    # such boundaries back INTO the sound (measured 36-222ms of a removed 呃 audible
    # in the output). "snap": false must keep the span byte-exact.
    wav, out = str(tmp_path / "in.wav"), str(tmp_path / "out.mp3")
    make_test_wav(wav)  # tone 0-1.5s, silence 1.5-2.2s, tone 2.2-3.7s
    t = {"audio": wav, "duration": 4.0, "words": [
        {"id": 0, "text": "好", "start": 0.3, "end": 1.0, "speaker": "a"}]}
    cut = {"type": "macro", "start": 1.3, "end": 1.8, "snap": False, "reason": "filler 呃"}
    res = r.render(t, [cut], wav, out)
    assert (1.3, 1.8) not in res["segments"]           # sanity: the cut happened
    bounds = [b for seg in res["segments"] for b in seg]
    assert 1.3 in bounds and 1.8 in bounds             # exact span, no snap (1.5 edge nearby)
    # same cut WITHOUT the flag gets snapped (start pulled to the 1.5 silence edge)
    res2 = r.render(t, [{**cut, "snap": None}], wav, str(tmp_path / "out2.mp3"))
    assert 1.3 not in [b for seg in res2["segments"] for b in seg]


def test_render_short_cut_not_collapsed_by_snap(tmp_path):
    # A sub-0.1s stutter cut near a silence: if it snapped, both ends would grab the
    # same edge and erase (or shift) the cut. Word-id cuts skip snapping, so word 1 goes.
    wav, out = str(tmp_path / "in.wav"), str(tmp_path / "out.mp3")
    make_test_wav(wav)  # silence at 1.5-2.2s
    t = {"audio": wav, "duration": 4.0, "words": [
        {"id": 0, "text": "這個", "start": 0.5, "end": 1.30, "speaker": "a"},
        {"id": 1, "text": "這", "start": 1.30, "end": 1.38, "speaker": "a"},  # stutter frag
        {"id": 2, "text": "種", "start": 1.38, "end": 1.50, "speaker": "a"}]}
    r.render(t, [{"start_word": 1, "end_word": 1, "reason": "stutter"}], wav, out)
    kept = {w["id"] for w in json.load(open(out.replace(".mp3", "_kept_transcript.json"), encoding="utf-8"))["words"]}
    assert 1 not in kept and 0 in kept and 2 in kept


# ── Task 2.2: render (single-file) ───────────────────────────────────────────
def test_render_output_duration_matches_kept(tmp_path):
    wav, out = str(tmp_path / "in.wav"), str(tmp_path / "out.mp3")
    make_test_wav(wav)
    t = sample_transcript(); t["audio"] = wav
    cuts = [{"type": "micro", "start_word": 0, "end_word": 0, "reason": "repeat 我覺得"}]
    result = r.render(t, cuts, wav, out)
    assert os.path.exists(out)
    kept = json.load(open(out.replace(".mp3", "_kept_transcript.json"), encoding="utf-8"))
    assert kept["words"][0]["id"] == 1
    expected = sum(e - s for s, e in result["segments"])
    assert abs(r.probe_duration(out) - expected) < 0.3  # no drift / no extra audio


def test_render_kept_path_robust_to_non_mp3_out(tmp_path):
    # out is .wav: kept_transcript must derive from the stem, not clobber the audio
    # (the old .replace(".mp3",…) would have written JSON over out.wav).
    wav, out = str(tmp_path / "in.wav"), str(tmp_path / "out.wav")
    make_test_wav(wav)
    t = sample_transcript(); t["audio"] = wav
    r.render(t, [{"type": "micro", "start_word": 0, "end_word": 0, "reason": "x"}], wav, out)
    assert r.probe_duration(out) > 0  # audio intact, not overwritten by JSON
    assert os.path.exists(str(tmp_path / "out_kept_transcript.json"))


# ── speech leveling (per-track, before mix) ──────────────────────────────────
def test_filtergraph_levels_each_track_before_mix():
    fc = r._filtergraph(2, [(0.0, 1.0), (2.0, 3.0)], track_gains={0: -3.0, 1: 2.5})
    assert "dynaudnorm" not in fc  # its gated min-filter fades every phrase tail
    for ti, g in ((0, -3.0), (1, 2.5)):  # static gain + compand between concat and [tN]
        assert f"concat=n=2:v=0:a=1,volume={g}dB,{r._LEVEL}[t{ti}]" in fc
    # mastering is a separate measured STATIC pass — single-pass loudnorm in the graph
    # is dynamic (time-varying gain) and upsamples to 192kHz; it must never come back
    assert "amix" in fc and "loudnorm" not in fc


def test_filtergraph_levels_single_track():
    fc = r._filtergraph(1, [(0.0, 1.0)])
    assert "volume=0.0dB" in fc and fc.count("compand") == 1 and "loudnorm" not in fc


def test_master_gain_targets_integrated_loudness():
    assert r._master_gain(-20.0) == 4.0            # -20 LUFS -> -16
    assert r._master_gain(-10.0) == -6.0           # loud input turns the gain down
    assert r._master_gain(float("-inf")) == 0.0    # silence: leave alone
    # peaks are the limiter's job — a real episode has ~28dB crest (laughter bursts),
    # so a TP-capped static gain left the whole episode 13dB under target
    assert "alimiter" in r._MASTER and "level=false" in r._MASTER


def test_render_masters_to_target_and_keeps_source_samplerate(tmp_path):
    # End-to-end: output lands near -16 LUFS via ONE static gain, and the file keeps the
    # source sample rate (in-graph loudnorm silently upsampled wav output to 192kHz).
    import subprocess
    wav, out = str(tmp_path / "in.wav"), str(tmp_path / "out.wav")
    make_test_wav(wav)
    t = sample_transcript(); t["audio"] = wav
    res = r.render(t, [], wav, out)
    i, tp = r.measure_loudness(out)
    assert i > -18.5 and tp <= -1.0  # at/near target unless the TP cap bound first
    rate = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate",
                           "-of", "default=nk=1:nw=1", out],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert rate == "16000"
    assert "master_gain_db" in res


def test_speech_gain_aligns_track_to_target(tmp_path):
    # A quiet speaker (tone RMS ~ -35 dBFS) gets a positive gain toward -23; the gain
    # is computed from SPEECH frames only, so leading silence must not dilute it.
    import numpy as np, soundfile as sf
    sr = 16000
    tone = np.sin(2 * np.pi * 220 * np.arange(4 * sr) / sr).astype(np.float32)
    quiet = str(tmp_path / "q.wav")
    sf.write(quiet, np.concatenate([np.zeros(4 * sr, np.float32), 0.025 * tone]), sr)
    g = r.speech_gain(quiet)
    assert 8.0 <= g <= 12.0  # ~-35 dB speech -> ~+12 toward -23 (clamped at 12)
    silent = str(tmp_path / "s.wav")
    sf.write(silent, np.zeros(2 * sr, np.float32), sr)
    assert r.speech_gain(silent) == 0.0  # nothing voiced -> leave the track alone


def test_level_compand_is_compression_only():
    # The compand stage must be downward compression only: a 1:1 floor below the knee.
    # Any boost region needs a >1 slope below it to return to unity, and a sentence
    # tail decaying through that zone falls FASTER than the source — an audible
    # fade-out ("講到最後越來越小聲"). Regression: never reintroduce a boost curve.
    assert "compand" in r._LEVEL
    import re
    pts = re.search(r"points=([^:,]+)", r._LEVEL).group(1)
    pairs = [tuple(map(float, p.split("/"))) for p in pts.split("|")]
    for (i1, o1), (i2, o2) in zip(pairs, pairs[1:]):
        assert (o2 - o1) / (i2 - i1) <= 1.0  # no expansion segment anywhere


def test_render_does_not_steepen_fading_tail(tmp_path):
    # Same speaker trails off: 4s at 0.4 then 4s at 0.1 (-12 dB). Leveling must not
    # make the quiet tail fall further than the source did (the fade-out regression);
    # squeezing the loud head may narrow the gap, never widen it.
    import numpy as np, soundfile as sf
    sr = 16000
    n = int(4 * sr)
    tone = np.sin(2 * np.pi * 220 * np.arange(n) / sr).astype(np.float32)
    wav, out = str(tmp_path / "in.wav"), str(tmp_path / "out.mp3")
    sf.write(wav, np.concatenate([0.4 * tone, 0.1 * tone]), sr)
    t = {"audio": wav, "duration": 8.0,
         "words": [{"id": 0, "text": "好", "start": 0.0, "end": 0.5, "speaker": "a"}]}
    r.render(t, [], wav, out)
    x, sr2 = sf.read(out)
    x = x if x.ndim == 1 else x.mean(axis=1)
    rms = lambda a, b: float(np.sqrt(np.mean(x[int(a * sr2):int(b * sr2)] ** 2)))
    ratio = rms(1.0, 3.0) / rms(5.0, 7.0)  # head vs tail, away from the transition
    assert ratio <= 4.5  # input ratio 4x; must not widen (small tolerance for codec)


def test_render_levels_sudden_volume_change(tmp_path):
    # Same speaker: 6s quiet tone (0.03) then 6s loud tone (0.5) — a ~17x (24dB) RMS jump.
    # The STATIC chain (per-track gain + downward-only compand) must squeeze the loud
    # burst hard. Its design optimum here is ~15.8dB (the quiet half sits below the -32
    # knee, and downward-only means it is never boosted): the old <3x expectation was
    # met only by single-pass loudnorm's DYNAMIC gain — the exact mechanism hard rule 7
    # forbids (it manufactures phrase-tail fades). Static mastering keeps the ratio.
    import numpy as np, soundfile as sf
    sr = 16000
    n = int(6 * sr)
    tone = np.sin(2 * np.pi * 220 * np.arange(n) / sr).astype(np.float32)
    wav, out = str(tmp_path / "in.wav"), str(tmp_path / "out.mp3")
    sf.write(wav, np.concatenate([0.03 * tone, 0.5 * tone]), sr)
    t = {"audio": wav, "duration": 12.0,
         "words": [{"id": 0, "text": "好", "start": 0.0, "end": 0.5, "speaker": "a"}]}
    r.render(t, [], wav, out)
    x, sr2 = sf.read(out)
    x = x if x.ndim == 1 else x.mean(axis=1)
    rms = lambda a, b: float(np.sqrt(np.mean(x[int(a * sr2):int(b * sr2)] ** 2)))
    ratio = rms(7.0, 11.0) / rms(1.0, 5.0)  # mid-half windows, away from the transition
    assert ratio < 7.0  # ~16.7x in -> ~6x out: the compand stage did its (static) work


# ── Task 2.2: render (multitrack — per-track cut then mix) ────────────────────
def test_render_mute_silences_span_without_removing_time(tmp_path):
    import numpy as np, soundfile as sf
    sr = 16000
    tone = (0.3 * np.sin(2 * np.pi * 220 * np.arange(int(3 * sr)) / sr)).astype(np.float32)
    wav, out = str(tmp_path / "in.wav"), str(tmp_path / "out.mp3")
    sf.write(wav, tone, sr)
    transcript = {"audio": wav, "duration": 3.0,
                  "words": [{"id": 0, "text": "你好", "start": 0.0, "end": 0.5, "speaker": "in"}]}
    # speaker == _speaker(wav) == "in"; mute 1.0-2.0s, no cuts
    r.render(transcript, [], wav, out, mutes=[{"speaker": "in", "start": 1.0, "end": 2.0}])
    x, sr2 = sf.read(out)
    x = x if x.ndim == 1 else x.mean(axis=1)
    rms = lambda a, b: float(np.sqrt(np.mean(x[int(a * sr2):int(b * sr2)] ** 2)))
    assert abs(r.probe_duration(out) - 3.0) < 0.2  # time NOT removed
    assert rms(1.3, 1.7) < 0.02                     # muted span ~silent
    assert rms(0.1, 0.8) > 0.05                     # rest keeps the tone


def test_render_multitrack_mute_does_not_drop_track_from_mix(tmp_path):
    # Regression: a muted track's volume=0 output is a filter-output label; feeding it to
    # every segment's atrim by reusing one label drops most of that track in a multi-input
    # graph (the host vanishes from the mix). It must be asplit into one copy per segment.
    import numpy as np, soundfile as sf
    sr = 16000
    tone = (0.3 * np.sin(2 * np.pi * 220 * np.arange(int(3 * sr)) / sr)).astype(np.float32)
    a, b = str(tmp_path / "x--hosta.wav"), str(tmp_path / "x--guestb.wav")
    sf.write(a, tone, sr)                         # host: continuous tone
    sf.write(b, np.zeros(int(3 * sr), np.float32), sr)  # 2nd input -> multi-input graph
    out = str(tmp_path / "out.mp3")
    t = {"duration": 3.0, "tracks": [a, b],
         "words": [{"id": 0, "text": "甲", "start": 0.0, "end": 0.4, "speaker": "hosta"},
                   {"id": 1, "text": "乙", "start": 0.6, "end": 1.0, "speaker": "hosta"},
                   {"id": 2, "text": "丙", "start": 2.0, "end": 2.4, "speaker": "hosta"},
                   {"id": 3, "text": "丁", "start": 2.6, "end": 3.0, "speaker": "hosta"}]}
    cuts = [{"type": "micro", "start_word": 1, "end_word": 1, "reason": "x"}]  # split -> 2 segments
    r.render(t, cuts, None, out,
             mutes=[{"speaker": "hosta", "start": 1.2, "end": 1.6}])  # mute in a word gap
    x, sr2 = sf.read(out); x = x if x.ndim == 1 else x.mean(axis=1)
    tail = float(np.sqrt(np.mean(x[int(2.0 * sr2):] ** 2)))  # a LATER segment (post-mute)
    assert tail > 0.05  # host's tone survives in the mix after the muted span


def test_filtergraph_stem_mode_keeps_leveling_but_skips_loudnorm():
    # A stem is one track rendered alone: mastering it in isolation would shift the
    # speaker balance when the stems are re-mixed. The graph only levels; render()'s
    # master=False path (used for stems) skips the measured gain.
    fc = r._filtergraph(1, [(0.0, 1.0)])
    assert "loudnorm" not in fc and "anull[out]" in fc
    assert "compand" in fc and "volume=0.0dB" in fc


def test_render_stems_exports_mix_and_per_speaker_stems_as_wav_from_wav(tmp_path):
    a, b = str(tmp_path / "x--host.wav"), str(tmp_path / "x--guest.wav")
    make_two_tracks(a, b)
    t = sample_transcript(); t.pop("audio", None); t["tracks"] = [a, b]
    cuts = [{"type": "micro", "start_word": 0, "end_word": 0, "reason": "repeat"}]
    res = r.render_stems(t, cuts, None, str(tmp_path / "out"),
                         mutes=[{"speaker": "host", "start": 1.6, "end": 1.9}])
    assert set(res["outputs"]) == {"mix", "host", "guest"}  # 整合 + one per speaker
    expected = sum(e - s for s, e in res["segments"])
    for path in res["outputs"].values():
        assert path.endswith(".wav") and os.path.exists(path)  # wav sources -> wav out
        assert abs(r.probe_duration(path) - expected) < 0.3  # all cut at the same boundaries


def test_render_stems_output_format_follows_mp3_source(tmp_path):
    import subprocess
    wav, mp3 = str(tmp_path / "in.wav"), str(tmp_path / "x--host.mp3")
    make_test_wav(wav)
    subprocess.run(["ffmpeg", "-y", "-i", wav, mp3], check=True, capture_output=True)
    t = sample_transcript(); t["audio"] = mp3
    res = r.render_stems(t, [], mp3, str(tmp_path / "out"))
    assert res["outputs"]["mix"].endswith("final.mp3")      # mp3 source -> mp3 out
    assert res["outputs"]["host"].endswith("final_host.mp3")
    assert all(os.path.exists(p) for p in res["outputs"].values())


def test_verify_mutes_blocks_speakers_own_words_only():
    words = [{"id": 0, "text": "你好", "start": 0.0, "end": 0.4, "speaker": "host"},
             {"id": 1, "text": "呃", "start": 1.0, "end": 1.2, "speaker": "host"},
             {"id": 2, "text": "對", "start": 0.1, "end": 0.5, "speaker": "guest"}]
    # covers the host's own content word -> blocked
    assert r.verify_mutes([{"speaker": "host", "start": 0.1, "end": 0.3}], words)
    # sits in a gap in the host's speech -> fine (guest overlap is the whole point of mutes)
    assert r.verify_mutes([{"speaker": "host", "start": 0.5, "end": 0.9}], words) == []
    # covers only the host's filler -> fine (fillers are removable, mirroring _protected)
    assert r.verify_mutes([{"speaker": "host", "start": 0.95, "end": 1.25}], words) == []


def test_render_rejects_mute_over_speakers_own_word(tmp_path):
    import pytest
    wav, out = str(tmp_path / "x--host.wav"), str(tmp_path / "out.mp3")
    make_test_wav(wav)
    t = {"audio": wav, "duration": 4.0, "words": [
        {"id": 0, "text": "你好", "start": 0.2, "end": 0.8, "speaker": "host"}]}
    with pytest.raises(ValueError, match="mutes cover"):
        r.render(t, [], wav, out, mutes=[{"speaker": "host", "start": 0.3, "end": 0.6}])


def test_mute_pieces_partitions_and_folds_overlaps():
    assert r._mute_pieces([(1.0, 2.0)], 3.0) == [(0.0, 1.0, False), (1.0, 2.0, True),
                                                 (2.0, 3.0, False)]
    # overlapping spans fold; clamped to [0, duration]; mute reaching the end: no tail
    assert r._mute_pieces([(-1.0, 0.5), (0.4, 3.0)], 3.0) == [(0.0, 0.5, True),
                                                              (0.5, 3.0, True)]


def test_filtergraph_mute_fades_instead_of_hard_step():
    fc = r._filtergraph(2, [(0.0, 3.0)], track_mutes={0: [(1.0, 2.0)]})
    assert "enable=" not in fc                       # the old hard gain step is gone
    assert "atrim=1.0:2.0,asetpts=PTS-STARTPTS,volume=0" in fc   # muted piece
    assert "atrim=0.0:1.0,asetpts=PTS-STARTPTS,afade=t=in" in fc  # faded speech piece
    assert "concat=n=3:v=0:a=1,asplit=1[m0_0]" in fc  # rebuilt to full length, then split


def test_resolve_cut_times_reports_unknown_word_id():
    import pytest
    with pytest.raises(ValueError, match="unknown word id 99"):
        r.resolve_cut_times([{"start_word": 99, "end_word": 99}], sample_transcript()["words"])


def test_render_keeps_tail_after_last_word(tmp_path):
    # transcript duration ends at the last word (3.7s) but the audio is 4.0s: the tail
    # (room tone / decay past the last word) must survive, not be truncated.
    wav, out = str(tmp_path / "in.wav"), str(tmp_path / "out.mp3")
    make_test_wav(wav)  # 4.0s file
    t = sample_transcript(); t["audio"] = wav; t["duration"] = 3.7  # last word's end
    res = r.render(t, [], wav, out)
    assert abs(r.probe_duration(out) - 4.0) < 0.2
    assert res["segments"][-1][1] == r.probe_duration(wav)


def test_render_multitrack_cuts_each_track_and_mixes(tmp_path):
    a, b, out = str(tmp_path / "a.wav"), str(tmp_path / "b.wav"), str(tmp_path / "out.mp3")
    make_two_tracks(a, b)
    t = sample_transcript(); t.pop("audio", None); t["tracks"] = [a, b]
    cuts = [{"type": "micro", "start_word": 0, "end_word": 0, "reason": "repeat"}]
    result = r.render(t, cuts, None, out)
    assert os.path.exists(out)
    expected = sum(e - s for s, e in result["segments"])
    assert abs(r.probe_duration(out) - expected) < 0.3
