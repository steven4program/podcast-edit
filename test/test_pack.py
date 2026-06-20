import helpers.pack as pack
from test.fixtures import sample_transcript


def test_pack_splits_on_gap_and_speaker_and_concats_zh():
    lines = [l for l in pack.pack(sample_transcript()).splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[0] == "[00:00–00:01 #0-3 host] 我覺得我覺得這個東西"
    assert lines[1] == "[00:02–00:03 #4-7 guest] 對啊就是就是很棒"


def test_pack_no_inter_token_spaces():
    assert " 我覺得 " not in pack.pack(sample_transcript())


def test_mmss():
    assert pack._mmss(0) == "00:00"
    assert pack._mmss(75) == "01:15"


def test_pack_converts_simplified_to_traditional():
    t = {"words": [
        {"id": 0, "text": "欢迎", "start": 0.0, "end": 0.4, "speaker": "ted35"},
        {"id": 1, "text": "来到", "start": 0.4, "end": 0.8, "speaker": "ted35"},
        {"id": 2, "text": "节目", "start": 0.8, "end": 1.2, "speaker": "ted35"},
    ]}
    out = pack.pack(t)
    assert "歡迎來到節目" in out
    assert "欢迎" not in out and "节目" not in out
