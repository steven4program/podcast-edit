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
