from helpers.deadair import find_dead_air


def _w(i, s, e, spk="a"):
    return {"id": i, "text": "字", "start": s, "end": e, "speaker": spk}


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
