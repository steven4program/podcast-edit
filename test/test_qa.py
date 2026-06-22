import numpy as np

import helpers.qa as qa


def test_detect_seams_flags_energy_jump():
    sr = 16000
    samples = np.concatenate([0.5 * np.ones(sr), 0.001 * np.ones(sr)]).astype(np.float32)
    flags = qa.detect_seams(samples, sr, joins=[1.0])
    assert len(flags) == 1 and flags[0]["ratio"] > 4.0


def test_detect_seams_ignores_smooth_join():
    sr = 16000
    samples = (0.3 * np.ones(2 * sr)).astype(np.float32)
    assert qa.detect_seams(samples, sr, joins=[1.0]) == []


def test_joins_from_segments():
    # segments lengths 0.4, 1.2, 2.1 -> internal joins at 0.4 and 1.6
    segs = [(0.0, 0.4), (1.0, 2.2), (3.0, 5.1)]
    assert qa.joins_from_segments(segs) == [0.4, 1.6]


def test_long_silences_flags_trailing_silence():
    # silence that runs to EOF must still be flagged
    sr = 16000
    samples = np.concatenate([0.3 * np.ones(sr), np.zeros(3 * sr)]).astype(np.float32)
    sils = qa._long_silences(samples, sr)
    assert len(sils) == 1
    assert abs(sils[0][0] - 1.0) < 0.1 and abs(sils[0][1] - 4.0) < 0.1
