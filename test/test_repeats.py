import helpers.repeats as rp


def W(*pairs):
    return [{"id": i, "text": t, "speaker": s} for i, (t, s) in enumerate(pairs)]


def test_phrase_repeat_deletes_earlier_copy_with_punctuation():
    # 我剛好，我剛好前 → delete first 我剛好 + comma (#0-3), keep the copy at #4
    words = W(("我", "a"), ("剛", "a"), ("好", "a"), ("，", "a"),
              ("我", "a"), ("剛", "a"), ("好", "a"), ("前", "a"))
    cuts = rp.find_repeats(words)
    assert [(c["start_word"], c["end_word"]) for c in cuts] == [(0, 3)]


def test_triple_single_token_emits_two_cuts():
    # 你你你用 → delete #0 and #1, keep the last 你 (#2)
    words = W(("你", "a"), ("你", "a"), ("你", "a"), ("用", "a"))
    cuts = rp.find_repeats(words)
    assert [c["start_word"] for c in cuts] == [0, 1]


def test_stutter_with_dash_is_caught_as_repeat():
    # 我-我們: the '-' is skipped, leaving adjacent 我 我 → caught (delete #0)
    words = W(("我", "a"), ("-", "a"), ("我", "a"), ("們", "a"))
    cuts = rp.find_repeats(words)
    assert cuts and cuts[0]["start_word"] == 0


def test_flags_emphasis_for_review():
    # 非常非常多 IS surfaced (helper is recall-only; the LLM must protect it)
    words = W(("非", "a"), ("常", "a"), ("非", "a"), ("常", "a"), ("多", "a"))
    assert any(c["start_word"] == 0 for c in cuts) if (cuts := rp.find_repeats(words)) else False


def test_ignores_cross_speaker_and_nonadjacent():
    assert rp.find_repeats(W(("對", "a"), ("對", "b"))) == []        # different speakers
    # 各自有各自 — the two 各自 are not adjacent (有 between)
    assert rp.find_repeats(W(("各", "a"), ("自", "a"), ("有", "a"),
                             ("各", "a"), ("自", "a"))) == []
