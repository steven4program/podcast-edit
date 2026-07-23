"""discourse.py: context decides, not the word. One assert per verdict path + geometry."""
from helpers.discourse import propose_discourse, _match, _content_stream


def _mk(spec):
    """spec: list of (text, speaker). Sequential 0.2s tokens; punctuation gets ~0 width."""
    words, t = [], 0.0
    for i, (text, spk) in enumerate(spec):
        dur = 0.0 if text.strip() in "，。、！？…—" else 0.2
        words.append({"id": i, "text": text, "start": round(t, 3),
                      "end": round(t + dur, 3), "speaker": spk})
        t += dur + (0.05 if dur else 0.0)
    return {"words": words, "events": []}


def _find(cands, text):
    return next(c for c in cands if c["text"] == text)


def test_monologue_tic_is_cut():
    # "…说。对，然后…" one speaker, sentence-initial, pivot after → a tic.
    t = _mk([("我", "a"), ("说", "a"), ("。", "a"), ("对", "a"), ("，", "a"),
             ("然", "a"), ("后", "a"), ("讲", "a")])
    c = _find(propose_discourse(t), "对")
    assert c["verdict"] == "cut"
    assert c["sentence_initial"] and c["followed_by_pivot"]
    assert not c["turn_response"] and not c["post_question"]


def test_turn_response_is_kept():
    # b answers a's turn — a real "对", never cut.
    t = _mk([("你", "a"), ("说", "a"), ("。", "a"), ("对", "b"), ("，", "b"),
             ("然", "b"), ("后", "b")])
    c = _find(propose_discourse(t), "对")
    assert c["verdict"] == "keep" and c["turn_response"]


def test_answering_question_is_kept():
    # same speaker but the prior clause is a question → "对" is the answer, keep it even
    # though it is sentence-initial with a pivot after.
    t = _mk([("对", "a"), ("吗", "a"), ("？", "a"), ("对", "a"), ("，", "a"),
             ("然", "a"), ("后", "a")])
    c = [x for x in propose_discourse(t) if x["text"] == "对"][-1]
    assert c["verdict"] == "keep" and c["post_question"]


def test_demonstrative_is_review():
    # "。那个 东西" — 那个 is a demonstrative, not a tic (no pivot after) → review, not cut.
    t = _mk([("说", "a"), ("。", "a"), ("那", "a"), ("个", "a"), ("东", "a"), ("西", "a")])
    c = _find(propose_discourse(t), "那个")
    assert c["verdict"] == "review" and not c["followed_by_pivot"]


def test_midclause_is_review():
    # "我 就是 想" — 就是 carries a grammatical role mid-clause → review, not cut.
    t = _mk([("我", "a"), ("就", "a"), ("是", "a"), ("想", "a")])
    c = _find(propose_discourse(t), "就是")
    assert c["verdict"] == "review" and not c["sentence_initial"]


def test_match_prefers_longest():
    # 对不对 must win over 对 so a tag question is not mis-cut as a bare 对.
    t = _mk([("对", "a"), ("不", "a"), ("对", "a")])
    stream = _content_stream(t["words"])
    assert _match(stream, 0) == ("对不对", 3)


def test_cut_span_folds_trailing_punct():
    # end_word is the token just before the next content word, so the trailing ，goes with
    # the tic (render then onset-aligns the end to the kept word).
    t = _mk([("说", "a"), ("。", "a"), ("对", "a"), ("，", "a"),
             ("然", "a"), ("后", "a")])
    c = _find(propose_discourse(t), "对")
    assert c["start_word"] == 2 and c["end_word"] == 3   # 对 + trailing ，, up to 然


def test_voiced_onset_recovers_late_token_start():
    # Scribe places 对/好 tokens at the tail of the sound. _voiced_onset must walk back over
    # the voiced run to its true start, not trust the late token_start.
    import numpy as np
    import helpers.discourse as d
    step, rms, thresh = 0.01, np.zeros(200), 0.5
    rms[50:90] = 1.0                                  # voiced run 0.50-0.90s
    onset = d._voiced_onset(rms, step, 0.80, 0.0, thresh)  # token starts late at 0.80
    assert 0.46 <= onset <= 0.51                      # recovers ~0.50, not 0.80
    assert d._voiced_onset(rms, step, 0.80, 0.60, thresh) >= 0.60  # never past the floor


def test_acoustic_cuts_cover_the_leaked_onset(tmp_path):
    # End-to-end: a 对 sound sits at [0.5,0.9] but Scribe's token is a late 50ms sliver
    # [0.80,0.85]. A word-id cut would leave [0.5,0.8] audible; acoustic_cuts must start the
    # cut at the true onset (~0.5) so the whole 对 goes, and mark it snap:false.
    import numpy as np, soundfile as sf
    import helpers.discourse as d
    sr = 16000
    t = np.arange(int(2.0 * sr)) / sr
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    sig = np.zeros_like(t)
    for s, e in ((0.0, 0.3), (0.5, 0.9), (1.2, 1.6)):   # 导 / 对 / 那, silence between
        m = (t >= s) & (t < e); sig[m] = tone[m]
    wav = str(tmp_path / "x--a.wav")
    sf.write(wav, sig.astype(np.float32), sr)
    transcript = {"words": [
        {"id": 0, "text": "导", "start": 0.0, "end": 0.3, "speaker": "a"},
        {"id": 1, "text": "。", "start": 0.3, "end": 0.3, "speaker": "a"},
        {"id": 2, "text": "对", "start": 0.80, "end": 0.85, "speaker": "a"},   # LATE token
        {"id": 3, "text": "，", "start": 0.85, "end": 0.85, "speaker": "a"},
        {"id": 4, "text": "那", "start": 1.2, "end": 1.6, "speaker": "a"}]}
    cand = [{"speaker": "a", "start_word": 2, "end_word": 3, "reason": "discourse 「对」"}]
    cuts, flagged = d.acoustic_cuts(transcript, [wav], cand)
    assert len(cuts) == 1 and not flagged
    cut = cuts[0]
    assert cut["snap"] is False and cut["type"] == "macro"
    assert cut["start"] < 0.75              # recovered the onset the token hid (bug was ~0.80)
    assert cut["start"] <= 0.55             # covers the real 对 sound start ~0.5
    assert abs(cut["end"] - 1.2) < 0.05     # ends at the next content word (那) onset
