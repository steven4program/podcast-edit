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


def test_false_start_restart_repeat_is_precise():
    # 它非——它花: 它 repeats across the dash → remove 它非—— (#0-2), keep restart 它花 (#3)
    words = W(("它", "a"), ("非", "a"), ("——", "a"), ("它", "a"), ("花", "a"))
    fs = rp.find_false_starts(words)
    assert (fs[0]["start_word"], fs[0]["end_word"]) == (0, 2)


def test_false_start_bare_abandon_flags_one_token_span():
    # 的斯——那個: no restart-repeat → 1-token guess (斯——, #1-2), keep 那個 (#3)
    words = W(("的", "a"), ("斯", "a"), ("——", "a"), ("那", "a"), ("個", "a"))
    fs = rp.find_false_starts(words)
    assert (fs[0]["start_word"], fs[0]["end_word"]) == (1, 2)


def test_dash_at_clean_end_is_skipped():
    # dash then a different speaker → not a false start
    assert rp.find_false_starts(W(("好", "a"), ("——", "a"), ("對", "b"))) == []


# ── timestamp-collapse warning (fabricated token widths in stutter bursts) ────
def WT(*items):
    return [{"id": i, "text": t, "speaker": s, "start": st, "end": en}
            for i, (t, s, st, en) in enumerate(items)]


def test_repeat_collapsed_span_carries_warning():
    # Scribe fabricated widths in a 你-你 burst: removed span = keep.start - first.start
    # = 10ms for 1 char → the cut would remove a sliver of a continuous sound. ⚠ marker
    # tells the judging step to FLAG, not cut. A normal-width duplicate stays clean.
    words = WT(("你", "a", 1.000, 1.010), ("-", "a", 1.010, 1.010),
               ("你", "a", 1.010, 1.300), ("用", "a", 1.300, 1.500))
    cuts = rp.find_repeats(words)
    assert len(cuts) == 1 and "塌縮" in cuts[0]["reason"]
    normal = WT(("你", "a", 1.000, 1.200), ("你", "a", 1.200, 1.400),
                ("用", "a", 1.400, 1.600))
    assert "塌縮" not in rp.find_repeats(normal)[0]["reason"]


def test_false_start_collapsed_span_carries_warning():
    # 它非——它花 with the fragment collapsed to 30ms/2字 → ⚠; normal span → clean.
    words = WT(("它", "a", 1.000, 1.010), ("非", "a", 1.010, 1.020),
               ("——", "a", 1.020, 1.020), ("它", "a", 1.030, 1.300),
               ("花", "a", 1.300, 1.500))
    fs = rp.find_false_starts(words)
    assert len(fs) == 1 and "塌縮" in fs[0]["reason"]
    normal = WT(("它", "a", 1.000, 1.200), ("非", "a", 1.200, 1.450),
                ("——", "a", 1.450, 1.450), ("它", "a", 1.600, 1.800),
                ("花", "a", 1.800, 2.000))
    assert "塌縮" not in rp.find_false_starts(normal)[0]["reason"]


def test_false_start_collapse_detected_behind_real_pause():
    # Fragment tokens collapsed (这个—— = 70ms for 2 chars) but a real 650ms pause
    # follows before the restart: resolve keeps the pause (0.5s cap), so the removed
    # span is just the 70ms sliver. Measuring to the keep onset (720ms) would hide
    # the collapse — the warning must mirror resolve's semantics and still fire.
    words = WT(("这", "a", 1.000, 1.050), ("个", "a", 1.050, 1.070),
               ("——", "a", 1.070, 1.070), ("这", "a", 1.720, 1.900),
               ("让", "a", 1.900, 2.100))
    fs = rp.find_false_starts(words)
    assert len(fs) == 1 and "塌縮" in fs[0]["reason"]


def test_collapse_check_skipped_without_timestamps():
    # minimal fixtures (no start/end) must not crash and get no marker
    cuts = rp.find_repeats(W(("你", "a"), ("你", "a"), ("用", "a")))
    assert cuts and "塌縮" not in cuts[0]["reason"]
