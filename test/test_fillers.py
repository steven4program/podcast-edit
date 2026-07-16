import numpy as np
import soundfile as sf

import helpers.fillers as f


def _track(path, spans, sr=16000, dur=3.0):
    t = np.arange(int(dur * sr)) / sr
    x = np.zeros_like(t)
    for a, b in spans:
        x[(t >= a) & (t < b)] = (0.3 * np.sin(2 * np.pi * 200 * t))[(t >= a) & (t < b)]
    sf.write(path, x.astype(np.float32), sr)


def test_acoustic_cut_lands_on_real_sound_not_token(tmp_path):
    # track: 你好 0.0-0.5, 呃 SOUND 1.0-1.3, 再見 2.0-2.5 (silence between)
    p = str(tmp_path / "2026-06-01--spkX.wav")
    _track(p, [(0.0, 0.5), (1.0, 1.3), (2.0, 2.5)])
    transcript = {"tracks": [p], "words": [
        {"id": 0, "text": "你好", "start": 0.0, "end": 0.5, "speaker": "spkX"},
        {"id": 1, "text": "呃", "start": 2.40, "end": 2.41, "speaker": "spkX"},  # mis-placed token
        {"id": 2, "text": "再見", "start": 2.0, "end": 2.5, "speaker": "spkX"},
    ]}
    cuts, flagged = f.propose_filler_cuts(transcript, [p])
    assert len(cuts) == 1 and not flagged
    # cut lands on the actual 呃 sound (~1.0-1.3), NOT the token position (2.4)
    assert 0.9 < cuts[0]["start"] < 1.1 and 1.2 < cuts[0]["end"] < 1.5


def test_quiet_track_filler_still_found(tmp_path):
    # A quiet mic: filler sound at amplitude 0.01 (RMS ~0.007, under the old absolute
    # 0.012 threshold -> was "already silent"). The noise-floor-relative threshold
    # must still find it.
    p = str(tmp_path / "2026-06-01--spkQ.wav")
    t = np.arange(int(3.0 * 16000)) / 16000
    x = np.zeros_like(t)
    for a, b in [(0.0, 0.5), (1.0, 1.3), (2.0, 2.5)]:
        x[(t >= a) & (t < b)] = (0.01 * np.sin(2 * np.pi * 200 * t))[(t >= a) & (t < b)]
    sf.write(p, x.astype(np.float32), 16000)
    transcript = {"tracks": [p], "words": [
        {"id": 0, "text": "你好", "start": 0.0, "end": 0.5, "speaker": "spkQ"},
        {"id": 1, "text": "呃", "start": 1.1, "end": 1.2, "speaker": "spkQ"},
        {"id": 2, "text": "再見", "start": 2.0, "end": 2.5, "speaker": "spkQ"},
    ]}
    cuts, flagged = f.propose_filler_cuts(transcript, [p])
    assert len(cuts) == 1 and not flagged
    assert 0.9 < cuts[0]["start"] < 1.1 and 1.2 < cuts[0]["end"] < 1.5


def test_flags_filler_overlapping_laughter(tmp_path):
    # 笑聲(0.8-1.3) and 呃(1.5-1.7) share the inter-word window: the first-to-last
    # voiced extent would cut BOTH. Laughter is never collateral -> flag, not cut.
    p = str(tmp_path / "2026-06-01--spkL.wav")
    _track(p, [(0.0, 0.5), (0.8, 1.3), (1.5, 1.7), (2.2, 2.6)])
    transcript = {"tracks": [p], "words": [
        {"id": 0, "text": "你好", "start": 0.0, "end": 0.5, "speaker": "spkL"},
        {"id": 1, "text": "呃", "start": 1.55, "end": 1.56, "speaker": "spkL"},
        {"id": 2, "text": "再見", "start": 2.2, "end": 2.6, "speaker": "spkL"},
    ], "events": [{"type": "laughter", "start": 0.8, "end": 1.3, "speaker": "spkL"}]}
    cuts, flagged = f.propose_filler_cuts(transcript, [p])
    assert not cuts
    assert len(flagged) == 1 and flagged[0]["reason"] == "overlaps laughter"


def test_flags_filler_embedded_in_continuous_speech(tmp_path):
    # no silence around the filler -> voiced span exceeds _MAX_EXTENT -> flagged, not cut
    p = str(tmp_path / "2026-06-01--spkY.wav")
    _track(p, [(0.0, 3.0)])  # continuous sound across the whole gap
    transcript = {"tracks": [p], "words": [
        {"id": 0, "text": "你好", "start": 0.0, "end": 0.5, "speaker": "spkY"},
        {"id": 1, "text": "呃", "start": 1.5, "end": 1.51, "speaker": "spkY"},
        {"id": 2, "text": "再見", "start": 2.5, "end": 3.0, "speaker": "spkY"},
    ]}
    cuts, flagged = f.propose_filler_cuts(transcript, [p])
    assert not cuts and len(flagged) == 1
