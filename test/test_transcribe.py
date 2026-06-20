import helpers.transcribe as tr


def _raw():
    return {"language_code": "zho", "words": [
        {"text": "你好", "start": 0.0, "end": 0.4, "type": "word", "speaker_id": "speaker_0"},
        {"text": " ", "start": 0.4, "end": 0.4, "type": "spacing", "speaker_id": "speaker_0"},
        {"text": "(coughs)", "start": 0.5, "end": 0.9, "type": "audio_event", "speaker_id": "speaker_0"},
        {"text": "嗎", "start": 1.0, "end": 1.3, "type": "word", "speaker_id": "speaker_1"},
        {"text": "(laughter)", "start": 1.4, "end": 1.8, "type": "audio_event", "speaker_id": "speaker_1"},
    ]}


def test_normalize_assigns_word_ids_and_skips_nonwords():
    t = tr.normalize_scribe(_raw())
    assert [w["id"] for w in t["words"]] == [0, 1]
    assert [w["text"] for w in t["words"]] == ["你好", "嗎"]
    assert t["words"][0]["speaker"] == "speaker_0"


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
    assert tr._classify_event("(三秒停顿)") == "pause"
    assert tr._classify_event("(七秒沉默)") == "pause"
    assert tr._classify_event("(背景噪音)") == "noise"
    assert tr._classify_event("(按键声)") == "noise"
    assert tr._classify_event("(something odd)") == "other"


def test_normalize_scribe_speaker_override():
    t = tr.normalize_scribe(_raw(), speaker="host")
    assert all(w["speaker"] == "host" for w in t["words"])
    assert all(e["speaker"] == "host" for e in t["events"])


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
