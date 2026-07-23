import os

import pytest

import helpers.transcribe as tr


def _flat(segments):
    return [x for seg in segments for x in seg]


def _raw():
    return {"language_code": "zho", "words": [
        {"text": "你好", "start": 0.0, "end": 0.4, "type": "word", "speaker_id": "speaker_0"},
        {"text": " ", "start": 0.4, "end": 0.4, "type": "spacing", "speaker_id": "speaker_0"},
        {"text": "(coughs)", "start": 0.5, "end": 0.9, "type": "audio_event", "speaker_id": "speaker_0"},
        {"text": "嗎", "start": 1.0, "end": 1.3, "type": "word", "speaker_id": "speaker_1"},
        {"text": "(laughter)", "start": 1.4, "end": 1.8, "type": "audio_event", "speaker_id": "speaker_1"},
    ]}


def test_posix_normalizes_backslash_paths():
    # transcript.json written on Windows must resolve on macOS/Linux -> forward slashes.
    assert tr._posix("source\\2026-06-08--ted35.wav") == "source/2026-06-08--ted35.wav"
    assert tr._posix("source/a.wav") == "source/a.wav"  # already-posix unchanged


def test_normalize_assigns_word_ids_and_skips_nonwords():
    t = tr.normalize_scribe(_raw())
    assert [w["id"] for w in t["words"]] == [0, 1]
    assert [w["text"] for w in t["words"]] == ["你好", "嗎"]
    assert t["words"][0]["speaker"] == "speaker_0"


def test_normalize_strips_hesitation_ellipsis_keeps_punct_and_decimals():
    raw = {"words": [
        {"text": "成長", "start": 0.0, "end": 0.5, "type": "word", "speaker_id": "s0"},
        {"text": "3.5", "start": 0.5, "end": 0.9, "type": "word", "speaker_id": "s0"},      # decimal kept
        {"text": "......", "start": 1.2, "end": 1.2, "type": "word", "speaker_id": "s0"},   # pure hesitation -> dropped
        {"text": "呃", "start": 1.5, "end": 1.7, "type": "word", "speaker_id": "s0"},        # filler word -> kept
        {"text": "......倍", "start": 3.0, "end": 3.4, "type": "word", "speaker_id": "s0"},  # ellipsis stripped off word
        {"text": "。", "start": 3.4, "end": 3.4, "type": "word", "speaker_id": "s0"},         # sentence punctuation -> kept
    ]}
    t = tr.normalize_scribe(raw)
    assert [w["text"] for w in t["words"]] == ["成長", "3.5", "呃", "倍", "。"]


def test_normalize_extracts_and_classifies_events():
    kinds = [(e["type"], e["speaker"]) for e in tr.normalize_scribe(_raw())["events"]]
    assert ("cough", "speaker_0") in kinds
    assert ("laughter", "speaker_1") in kinds


def test_normalize_duration_is_last_word_end():
    assert tr.normalize_scribe(_raw())["duration"] == 1.3


# ── new tests ──────────────────────────────────────────────────────────────

def test_speaker_from_filename():
    assert tr.speaker_from_filename("2026-06-01--t02-54-49am--guest468401--tony.wav") == "tony"
    assert tr.speaker_from_filename("2026-06-01--t02-54-49am--60b4e983be6c5f0878af8f50--ted35.wav") == "ted35"
    assert tr.speaker_from_filename("/some/path/2026-06-01--t02-54-49am--host--wan.flac") == "wan"
    assert tr.speaker_from_filename("/recordings/2026-06-01--t02-54-49am--guest--bohr.mp3") == "bohr"


def test_classify_chinese_events():
    # Scribe emits Chinese event labels for zh audio (confirmed at Gate 1)
    assert tr._classify_event("(咳嗽声)") == "cough"
    assert tr._classify_event("(清嗓声)") == "throat_clear"
    assert tr._classify_event("(笑声)") == "laughter"
    assert tr._classify_event("(吸气声)") == "breath"
    assert tr._classify_event("(呼吸声)") == "breath"
    # nose-clears/sniffs are goal-1 candidates — must map to throat_clear, not fall
    # through to other/breath (scribe_events only surfaces throat_clear/cough)
    assert tr._classify_event("(吸鼻声)") == "throat_clear"
    assert tr._classify_event("(擤鼻涕声)") == "throat_clear"
    assert tr._classify_event("(抽鼻子声)") == "throat_clear"
    assert tr._classify_event("(sniffs)") == "throat_clear"
    assert tr._classify_event("(三秒停顿)") == "pause"
    assert tr._classify_event("(七秒沉默)") == "pause"
    assert tr._classify_event("(背景噪音)") == "noise"
    assert tr._classify_event("(按键声)") == "noise"
    assert tr._classify_event("(something odd)") == "other"


def test_normalize_scribe_speaker_override():
    t = tr.normalize_scribe(_raw(), speaker="host")
    assert all(w["speaker"] == "host" for w in t["words"])
    assert all(e["speaker"] == "host" for e in t["events"])


def test_voiced_segments_complement_pad_merge():
    # one 1.0-2.0s silence in a 5s clip; pad 0.2 shrinks the gap on both sides
    segs = tr.voiced_segments([(1.0, 2.0)], 5.0, pad=0.2)
    assert _flat(segs) == pytest.approx([0.0, 1.2, 1.8, 5.0])


def test_voiced_segments_merges_when_padding_overlaps():
    # a 0.3s silence is narrower than 2x pad, so padding closes it -> one segment
    segs = tr.voiced_segments([(1.0, 1.3)], 3.0, pad=0.2)
    assert _flat(segs) == pytest.approx([0.0, 3.0])


def test_voiced_segments_all_silent_is_empty():
    # fully-silent track -> nothing to send to Scribe -> track is skipped
    assert tr.voiced_segments([(0.0, 5.0)], 5.0) == []


def test_remap_to_original_adds_back_removed_silence():
    # kept voiced [0,1.2] then [1.8,5.0]; 0.6s of silence was removed between them
    segs = [(0.0, 1.2), (1.8, 5.0)]
    assert tr.remap_to_original(0.5, segs) == pytest.approx(0.5)   # before any cut
    assert tr.remap_to_original(1.2, segs) == pytest.approx(1.2)   # seam
    assert tr.remap_to_original(1.5, segs) == pytest.approx(2.1)   # 1.8 + (1.5-1.2)


def test_remap_clamps_token_stretched_across_vad_seam():
    # Trimmed timeline: voiced segs [0,1.0]+[1.0,2.0] map to original [10,11]+[30,31].
    # A token spanning the seam (0.8-1.2 trimmed) would remap to 10.8-30.2 — 19.4s of
    # REMOVED silence inside one 嗯. Its end must be capped at its segment's end (11.0).
    # (starts at 0.8 = the pad-zone edge, so the start-shift rule below must NOT fire)
    segments = [(10.0, 11.0), (30.0, 31.0)]
    t = {"words": [{"id": 0, "text": "嗯", "start": 0.8, "end": 1.2, "speaker": "a"},
                   {"id": 1, "text": "好", "start": 1.3, "end": 1.5, "speaker": "a"}],
         "events": [{"type": "laughter", "start": 0.7, "end": 1.4, "speaker": "a"}]}
    out = tr._remap_transcript(t, segments)
    w0, w1 = out["words"]
    assert w0["start"] == pytest.approx(10.8) and w0["end"] <= 11.0  # capped at seg end
    assert w1["start"] == pytest.approx(30.3) and w1["end"] == pytest.approx(30.5)  # untouched
    assert out["events"][0]["end"] <= 11.0  # events capped the same way


def test_remap_shifts_seam_straddler_to_next_segment_start():
    # A token that STARTS inside a segment's trailing 0.2s speech-pad (VAD-certified
    # silence — no word can begin there) and runs past the seam belongs to the NEXT
    # segment: Scribe placed the onset slightly early. The old code parked it at the
    # previous segment's tail — 19s from its real position here.
    segments = [(10.0, 11.0), (30.0, 31.0)]
    t = {"words": [{"id": 0, "text": "歡迎", "start": 0.85, "end": 1.30, "speaker": "a"}],
         "events": [{"type": "laughter", "start": 0.9, "end": 1.5, "speaker": "a"}]}
    out = tr._remap_transcript(t, segments)
    w = out["words"][0]
    assert w["start"] == pytest.approx(30.0) and w["end"] == pytest.approx(30.3)
    ev = out["events"][0]
    assert ev["start"] == pytest.approx(30.0) and ev["end"] == pytest.approx(30.5)


def test_remap_keeps_straddler_that_starts_in_real_speech():
    # start 0.75 is BEFORE the pad zone (real speech by VAD) -> the sound genuinely sits
    # at segment 1's tail; end stays capped there (the measured 嗯→10s fix, unchanged)
    segments = [(10.0, 11.0), (30.0, 31.0)]
    t = {"words": [{"id": 0, "text": "好", "start": 0.75, "end": 1.2, "speaker": "a"}],
         "events": []}
    out = tr._remap_transcript(t, segments)
    w = out["words"][0]
    assert w["start"] == pytest.approx(10.75) and w["end"] <= 11.0


def test_merge_tracks_sorts_and_reids():
    track_a = {
        "audio": "", "duration": 1.5,
        "words": [
            {"id": 0, "text": "hello", "start": 0.0, "end": 0.4, "speaker": "a"},
            {"id": 1, "text": "world", "start": 1.0, "end": 1.5, "speaker": "a"},
        ],
        "events": [{"type": "cough", "start": 0.2, "end": 0.3, "speaker": "a"}],
    }
    track_b = {
        "audio": "", "duration": 0.9,
        "words": [
            {"id": 0, "text": "hi", "start": 0.5, "end": 0.9, "speaker": "b"},
        ],
        "events": [{"type": "laughter", "start": 0.6, "end": 0.8, "speaker": "b"}],
    }
    merged = tr.merge_tracks([track_a, track_b])
    assert [w["text"] for w in merged["words"]] == ["hello", "hi", "world"]
    assert [w["id"] for w in merged["words"]] == [0, 1, 2]
    assert [w["speaker"] for w in merged["words"]] == ["a", "b", "a"]
    assert merged["duration"] == 1.5
    assert [e["type"] for e in merged["events"]] == ["cough", "laughter"]
    # real audio length wins over last-word end (tail after the last word must survive
    # into render); word ends still win if a track's probe came up short
    assert tr.merge_tracks([track_a, track_b], duration=60.0)["duration"] == 60.0
    assert tr.merge_tracks([track_a, track_b], duration=1.0)["duration"] == 1.5


def test_silence_thresh_is_noise_floor_relative(tmp_path):
    import numpy as np, soundfile as sf
    sr = 16000
    t = np.arange(int(3.0 * sr)) / sr
    # quiet mic: near-digital-silence floor + a -43dB aside -> threshold drops to the
    # -60 clamp so the quiet sound survives VAD (the old absolute -40 erased it)
    x = np.zeros_like(t)
    x[(t >= 0) & (t < 1.0)] = (0.3 * np.sin(2 * np.pi * 200 * t))[(t >= 0) & (t < 1.0)]
    x[(t >= 1.0) & (t < 2.5)] = (0.007 * np.sin(2 * np.pi * 300 * t))[(t >= 1.0) & (t < 2.5)]
    quiet = str(tmp_path / "q.wav"); sf.write(quiet, x.astype(np.float32), sr)
    assert tr._silence_thresh_db(quiet) == -60.0
    # hot mic (floor ~ -29dB): clamp at the old -40 — never MORE aggressive than before
    hot = str(tmp_path / "h.wav")
    sf.write(hot, (0.05 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sr)
    assert tr._silence_thresh_db(hot) == float(tr._SILENCE_DB)
    # undecodable file -> the old constant (fail open to current behavior)
    bad = str(tmp_path / "b.m4a")
    with open(bad, "wb") as f:
        f.write(b"not audio")
    assert tr._silence_thresh_db(bad) == float(tr._SILENCE_DB)


def test_vad_keeps_quiet_speech_on_a_quiet_mic(tmp_path):
    # End-to-end VAD: the -43dB region (a soft laugh/aside) must reach Scribe. Under
    # the old absolute -40 it was trimmed -> no words, no laughter event, no protection
    # anywhere downstream (deadair would propose cutting it as dead air).
    import numpy as np, soundfile as sf
    sr = 16000
    t = np.arange(int(3.0 * sr)) / sr
    x = np.zeros_like(t)
    x[(t >= 0) & (t < 1.0)] = (0.3 * np.sin(2 * np.pi * 200 * t))[(t >= 0) & (t < 1.0)]
    x[(t >= 1.0) & (t < 2.5)] = (0.007 * np.sin(2 * np.pi * 300 * t))[(t >= 1.0) & (t < 2.5)]
    p = str(tmp_path / "q.wav"); sf.write(p, x.astype(np.float32), sr)
    segs = tr.voiced_segments(tr._detect_silence(p, 3.0, tr._silence_thresh_db(p)), 3.0)
    assert any(s <= 1.2 and 2.3 <= e for s, e in segs)  # quiet region fully voiced


def test_normalize_clamps_junk_word_durations():
    # Scribe emitted a single-char 股 spanning ~98s; such a word covers a minute of
    # timeline and the mid-word guard then blocks every cut inside it. The end must be
    # capped to a per-character budget; a normal word's end passes through untouched.
    from helpers.transcribe import normalize_scribe
    raw = {"words": [
        {"type": "word", "text": "股", "start": 349.0, "end": 447.0},
        {"type": "word", "text": "你好", "start": 448.0, "end": 448.5},
    ]}
    t = normalize_scribe(raw, speaker="a")
    assert t["words"][0]["end"] <= 349.0 + 1.0   # 98s junk clamped hard
    assert t["words"][1]["end"] == 448.5         # normal duration untouched


def test_extract_voiced_wav_survives_many_segments(tmp_path):
    # A full-length episode yields hundreds of VAD segments. Passed inline the filtergraph
    # blew past the OS argv cap (Windows CreateProcess: WinError 206) and transcription died
    # on the real episode while every 5-minute fixture passed. Must go via a script file.
    import numpy as np, soundfile as sf
    sr = 16000
    n_seg = 700
    x = (0.3 * np.sin(2 * np.pi * 200 * np.arange(int(80.0 * sr)) / sr)).astype(np.float32)
    p = str(tmp_path / "long.wav"); sf.write(p, x, sr)
    segs = [(i * 0.1, i * 0.1 + 0.05) for i in range(n_seg)]
    assert len("".join(f"[0:a]atrim={s}:{e},asetpts=PTS-STARTPTS[s{i}];"
                       for i, (s, e) in enumerate(segs))) > 32000  # past the argv ceiling
    wav = tr._extract_voiced_wav(p, segs)
    try:
        assert sf.info(wav).frames > 0
    finally:
        os.remove(wav)
