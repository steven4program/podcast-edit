from helpers.compile_cuts import compile_cuts


def _w(i, s, e, text="字", spk="host"):
    return {"id": i, "text": text, "start": s, "end": e, "speaker": spk}


def _t(words):
    return {"duration": max(w["end"] for w in words), "words": words}


def test_intro_word_emits_macro_cut_and_subsumes_inside():
    words = [_w(0, 0.0, 0.4, "嗨"), _w(1, 0.4, 0.8, "大家好"), _w(2, 5.0, 5.4, "主題")]
    cand = {"fillers": {"cuts": [{"type": "macro", "start": 0.2, "end": 0.35,
                                  "snap": False, "reason": "filler 呃"}]}}  # inside the intro
    spec, flagged, stats = compile_cuts(_t(words), cand, intro_word=2)
    macro = [c for c in spec["cuts"] if c["reason"].startswith("pre-show")]
    assert macro and macro[0]["start"] == 0.0 and macro[0]["end"] == 5.0  # to word #2's start
    assert stats["fillers"] == 0  # the in-intro filler was subsumed, not re-added


def test_drop_phrase_excludes_reduplication():
    words = [_w(0, 0.0, 0.3, "剛"), _w(1, 0.3, 0.6, "剛"), _w(2, 0.6, 1.0, "好")]
    cand = {"repeats": {"repeats": [
        {"type": "micro", "start_word": 0, "end_word": 0, "reason": "repeat 「剛」 — delete-earlier"}]}}
    # no baked-in reduplication list (editorial judgment stays the LLM's) → kept by default,
    kept = compile_cuts(_t(words), cand)[0]["cuts"]
    assert len(kept) == 1
    kept2 = compile_cuts(_t(words), cand, drop=["剛"])[0]["cuts"]  # LLM passes it as a drop
    assert kept2 == []


def test_exclude_id_vetoes_one_candidate():
    words = [_w(0, 0.0, 0.3, "我"), _w(1, 0.3, 0.6, "我"), _w(2, 0.6, 1.0, "說")]
    cand = {"repeats": {"repeats": [
        {"type": "micro", "start_word": 0, "end_word": 0, "reason": "repeat 「我」 — delete-earlier"}]}}
    assert compile_cuts(_t(words), cand)[0]["cuts"]                 # kept by default (real stutter)
    assert compile_cuts(_t(words), cand, exclude_ids=[0])[0]["cuts"] == []  # vetoed


def test_collapse_repeat_goes_to_flagged_not_cuts():
    words = [_w(0, 0.0, 0.3, "你"), _w(1, 0.3, 0.6, "你"), _w(2, 0.6, 1.0, "投")]
    cand = {"repeats": {"repeats": [{"type": "micro", "start_word": 0, "end_word": 0,
        "reason": "repeat 「你」 ⚠ 塌縮 30ms/1字 — token times fabricated; FLAG"}]}}
    spec, flagged, _ = compile_cuts(_t(words), cand)
    assert spec["cuts"] == []
    assert any("塌縮" in ln and "「你」" in ln for ln in flagged)


def test_cough_mutes_pass_and_coarticulated_flagged():
    words = [_w(0, 0.0, 1.0), _w(1, 3.0, 4.0)]
    cand = {"cough": {"mutes": [{"speaker": "host", "start": 1.5, "end": 2.0}],
                      "flagged": [{"start": 3.2, "end": 3.4, "type": "throat_clear"}]}}
    spec, flagged, _ = compile_cuts(_t(words), cand)
    assert spec["mutes"] == [{"speaker": "host", "start": 1.5, "end": 2.0}]
    assert any("co-articulated" in ln and "00:03" in ln for ln in flagged)


def test_midword_microcut_dropped_on_multitrack_crosstalk():
    # host stutters 我我 (cut the earlier 我); a GUEST word 0.2-0.5 straddles the cut's
    # boundary at 0.3 → the boundary is mid-word for the guest → render would reject it, so
    # compile drops the offending micro cut and notes it in flagged.
    words = [_w(0, 0.0, 0.3, "我"), _w(1, 0.3, 0.6, "我"),
             _w(2, 0.2, 0.5, "談", spk="guest"), _w(3, 0.6, 0.9, "好")]
    cand = {"repeats": {"repeats": [
        {"type": "micro", "start_word": 0, "end_word": 0, "reason": "repeat 「我」 — delete-earlier"}]}}
    spec, flagged, stats = compile_cuts(_t(words), cand)
    assert spec["cuts"] == []                       # dropped (mid-word on guest's track)
    assert stats["midword_dropped"] == 1
    assert any("mid-word" in ln for ln in flagged)

    # remove the overlapping guest word → the same cut is now clean and survives
    words_clean = [words[0], words[1], words[3]]
    spec2, _, stats2 = compile_cuts(_t(words_clean), cand)
    assert len(spec2["cuts"]) == 1 and stats2["midword_dropped"] == 0
