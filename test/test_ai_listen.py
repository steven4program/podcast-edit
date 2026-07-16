import helpers.ai_listen as ai


def test_parse_label_maps_keywords():
    assert ai.parse_label("This is a cough.") == "cough"
    assert ai.parse_label("sounds like throat clearing") == "throat_clear"
    assert ai.parse_label("laughter") == "laughter"
    assert ai.parse_label("normal speech") == "speech"
    assert ai.parse_label("unsure / music") == "other"


def test_retry_after_parses_server_delay():
    assert ai._retry_after("Please retry in 50.6s", 8.0) == 51.6
    assert ai._retry_after("retryDelay': '50s'", 8.0) == 51.0
    assert ai._retry_after("some other error", 8.0) == 8.0


def test_windows_cover_duration_with_overlap():
    w = ai._windows(25.0, 12.0, 11.0)
    assert w[0] == (0.0, 12.0)
    assert w[-1][1] == 25.0           # last window reaches the end
    assert all(b - a <= 12.0 + 1e-9 for a, b in w)


def test_parse_events_offsets_clamps_and_drops_laughter():
    txt = '[{"start":1.0,"end":1.5,"type":"throat_clear"},' \
          '{"start":2.0,"end":99.0,"type":"cough"},' \
          '{"start":3.0,"end":3.4,"type":"laughter"}]'
    evs = ai._parse_events(txt, offset=10.0, win_end=22.0)
    assert [e["type"] for e in evs] == ["throat_clear", "cough"]   # laughter dropped
    assert evs[0]["start"] == 11.0 and evs[0]["end"] == 11.5       # offset applied
    assert evs[1]["end"] == 22.0                                   # clamped to win_end


def test_parse_events_handles_junk():
    assert ai._parse_events("no json here", 0.0, 12.0) == []
    assert ai._parse_events("[broken", 0.0, 12.0) == []


def test_merge_collapses_near_duplicates_from_overlap():
    evs = [{"start": 11.2, "end": 11.6, "type": "throat_clear"},
           {"start": 11.5, "end": 11.9, "type": "throat_clear"},  # overlaps prev -> merge
           {"start": 30.0, "end": 30.3, "type": "cough"}]
    m = ai._merge(evs)
    assert len(m) == 2 and m[0]["end"] == 11.9


def test_verify_on_track_drops_silent_hallucinations(tmp_path):
    import numpy as np, soundfile as sf
    sr = 16000
    np.random.seed(0)
    x = (1e-4 * np.random.randn(5 * sr)).astype(np.float32)   # faint noise floor everywhere
    seg = slice(int(1.0 * sr), int(1.2 * sr))                 # a real burst at 1.0-1.2s
    x[seg] += (0.2 * np.sin(2 * np.pi * 200 * np.arange(seg.stop - seg.start) / sr)).astype(np.float32)
    wav = str(tmp_path / "host.wav"); sf.write(wav, x, sr)
    events = [{"start": 1.0, "end": 1.2, "type": "throat_clear"},   # real -> keep
              {"start": 3.0, "end": 3.2, "type": "throat_clear"}]   # at floor -> drop
    kept = ai.verify_on_track(events, wav, k=5.0)
    assert len(kept) == 1 and kept[0]["start"] == 1.0


def test_verify_on_track_keeps_real_event_at_time_zero(tmp_path):
    # Regression: (start - pad) went negative for an event at t≈0; a negative slice
    # index reads from the END of the track, sees near-silence, and drops a real event.
    import numpy as np, soundfile as sf
    sr = 16000
    np.random.seed(1)
    x = (1e-4 * np.random.randn(5 * sr)).astype(np.float32)
    burst = (0.2 * np.sin(2 * np.pi * 200 * np.arange(int(0.2 * sr)) / sr)).astype(np.float32)
    x[:len(burst)] += burst                                   # real energy at 0.0-0.2s
    x[-len(burst):] = 0.0                                     # tail is dead silent
    wav = str(tmp_path / "host.wav"); sf.write(wav, x, sr)
    kept = ai.verify_on_track([{"start": 0.0, "end": 0.2, "type": "cough"}], wav, k=5.0)
    assert len(kept) == 1


def test_scribe_events_filters_pads_and_merges():
    t = {"events": [
        {"type": "throat_clear", "start": 10.0, "end": 10.0, "speaker": "ted"},  # zero-width
        {"type": "cough", "start": 10.4, "end": 10.6, "speaker": "ted"},         # near prev -> merge
        {"type": "laughter", "start": 20.0, "end": 20.5, "speaker": "ted"},      # never a candidate
        {"type": "throat_clear", "start": 30.0, "end": 30.2, "speaker": "tony"}, # other speaker
    ]}
    evs = ai.scribe_events(t, "ted", pad=0.3)
    assert len(evs) == 1                       # laughter + other speaker excluded, pair merged
    assert evs[0]["start"] == 9.7 and evs[0]["end"] == 10.9  # padded both ways


def test_scribe_events_padding_stops_at_host_words():
    # The tag sits in a gap between the host's words; padding must clip at the word
    # edges so split_events still sees a gap event (mute), not a fake co-articulation.
    t = {"words": [{"id": 0, "text": "好", "start": 9.0, "end": 9.9, "speaker": "ted"},
                   {"id": 1, "text": "嗯", "start": 10.5, "end": 10.7, "speaker": "ted"},  # filler: no clip
                   {"id": 2, "text": "對", "start": 11.1, "end": 11.5, "speaker": "ted"}],
         "events": [{"type": "cough", "start": 10.0, "end": 10.9, "speaker": "ted"}]}
    evs = ai.scribe_events(t, "ted", pad=0.3)
    assert evs == [{"start": 9.9, "end": 11.1, "type": "cough"}]
    mutes, flagged = ai.split_events(evs, t["words"], host="ted")
    assert len(mutes) == 1 and not flagged


def test_scribe_events_padding_stops_at_host_laughter():
    # A clear right before his own laugh: the padded mute must clip at the laugh's
    # start — a mute reaching into a laugh beheads it (laughter is never collateral).
    t = {"words": [], "events": [
        {"type": "throat_clear", "start": 10.0, "end": 10.2, "speaker": "ted"},
        {"type": "laughter", "start": 10.4, "end": 11.0, "speaker": "ted"}]}
    evs = ai.scribe_events(t, "ted", pad=0.3)
    assert evs == [{"start": 9.7, "end": 10.4, "type": "throat_clear"}]


def test_scribe_events_skips_tag_overlapping_his_laughter():
    # The tag itself overlaps his laughter -> ambiguous audio, and any detector saying
    # laughter means keep: not a candidate at all.
    t = {"words": [], "events": [
        {"type": "throat_clear", "start": 10.0, "end": 10.5, "speaker": "ted"},
        {"type": "laughter", "start": 10.4, "end": 11.0, "speaker": "ted"}]}
    assert ai.scribe_events(t, "ted") == []


def test_scribe_events_merge_never_bridges_a_laugh():
    # Two clears sandwich a 0.3s laugh: each padded span clips at the laugh's edges,
    # but the 0.4s merge gap would bridge them ACROSS the laugh and the merged mute
    # would silence it. The bridged span must be dropped whole (lose two mutes, never
    # behead the laugh).
    t = {"words": [], "events": [
        {"type": "throat_clear", "start": 9.8, "end": 10.0, "speaker": "ted"},
        {"type": "laughter", "start": 10.0, "end": 10.3, "speaker": "ted"},
        {"type": "cough", "start": 10.3, "end": 10.5, "speaker": "ted"}]}
    assert ai.scribe_events(t, "ted", pad=0.3) == []


def test_split_events_gap_mutes_coarticulated_flags():
    words = [{"id": 0, "text": "你好", "start": 5.0, "end": 5.4, "speaker": "ted35"},
             {"id": 1, "text": "嗎", "start": 8.0, "end": 8.3, "speaker": "tony"}]
    events = [{"start": 6.0, "end": 6.3, "type": "throat_clear"},   # gap in ted35 -> mute
              {"start": 5.1, "end": 5.3, "type": "throat_clear"}]   # inside ted35 word -> flag
    mutes, flagged = ai.split_events(events, words, host="ted35")
    assert mutes == [{"speaker": "ted35", "start": 6.0, "end": 6.3}]
    assert len(flagged) == 1 and flagged[0]["start"] == 5.1


def test_split_events_flags_span_too_narrow_to_mute():
    # A zero-width Scribe tag whose padding was clipped by words on both sides yields a
    # ~20ms span: the real clear extends UNDER the words, so a mute silences nothing
    # while the review page claims it handled it. Too-narrow -> flag, not mute.
    words = [{"id": 0, "text": "好", "start": 9.5, "end": 10.28, "speaker": "ted"},
             {"id": 1, "text": "對", "start": 10.31, "end": 10.8, "speaker": "ted"}]
    events = [{"start": 10.29, "end": 10.31, "type": "throat_clear"}]  # 20ms gap event
    mutes, flagged = ai.split_events(events, words, host="ted")
    assert mutes == [] and len(flagged) == 1
