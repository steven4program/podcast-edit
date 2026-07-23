"""align.py pure logic: romanization + the energy-grounded trust decision (no torch needed).
The trust decision must NOT hinge on a confidence threshold — it keeps whichever time lands on
the speaker's own voiced energy, so it holds regardless of speaking rate / loudness."""
import numpy as np
import pytest

import helpers.align as A


def _rms_voiced(*spans, step=0.01, n=500):
    """rms array (thresh 0.5): 1.0 inside each (start,end) second span, 0 elsewhere."""
    rms = np.zeros(n)
    for s, e in spans:
        rms[int(s / step):int(e / step)] = 1.0
    return rms


def test_romanize_english_and_punctuation():
    assert A._romanize("Thinking") == "thinking"
    assert A._romanize("Machine，") == "machine"   # trailing punctuation stripped
    assert A._romanize("。") == "" and A._romanize("，") == ""


def test_romanize_hanzi_to_pinyin():
    pytest.importorskip("pypinyin")
    assert A._romanize("矽") == "xi" and A._romanize("谷") == "gu"


def test_reconcile_uses_alignment_when_scribe_time_is_on_silence():
    # The 矽谷 case: Scribe parks 矽 at 2.00 (silence); alignment puts it at 1.10 (on the voice).
    rms = _rms_voiced((1.0, 1.5))
    words = [{"id": 5, "text": "矽", "start": 2.00, "end": 2.05, "speaker": "a"}]
    out, report = A.reconcile(words, [(1.10, 1.30, 0.66)], rms, 0.01, 0.5)
    assert out[0]["t_src"] == "align" and out[0]["start"] == 1.10
    assert any(wid == 5 for wid, _ in report)          # flagged: scribe was on silence


def test_reconcile_falls_back_when_alignment_lands_on_silence():
    rms = _rms_voiced((1.0, 1.5))
    words = [{"id": 0, "text": "好", "start": 1.05, "end": 1.25, "speaker": "a"}]
    out, _ = A.reconcile(words, [(3.00, 3.20, 0.9)], rms, 0.01, 0.5)   # aligned on silence
    assert out[0]["t_src"] == "scribe" and out[0]["start"] == 1.05


def test_reconcile_falls_back_when_alignment_out_of_order():
    rms = _rms_voiced((1.0, 2.0))
    words = [{"id": 0, "text": "一", "start": 1.0, "end": 1.2, "speaker": "a"},
             {"id": 1, "text": "二", "start": 1.4, "end": 1.6, "speaker": "a"}]
    # second word's aligned start (0.5) precedes the first -> monotonic violation -> fallback
    out, _ = A.reconcile(words, [(1.05, 1.25, 0.9), (0.50, 0.70, 0.9)], rms, 0.01, 0.5)
    assert out[0]["t_src"] == "align" and out[1]["t_src"] == "scribe"


def test_reconcile_keeps_alignment_quietly_when_both_voiced():
    # Both on the voice, small drift -> take alignment (better boundaries), no review note.
    rms = _rms_voiced((1.0, 1.5))
    words = [{"id": 0, "text": "好", "start": 1.00, "end": 1.20, "speaker": "a"}]
    out, report = A.reconcile(words, [(1.05, 1.25, 0.8)], rms, 0.01, 0.5)
    assert out[0]["t_src"] == "align" and out[0]["start"] == 1.05
    assert report == []
