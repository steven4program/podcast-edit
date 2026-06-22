import helpers.ai_listen as ai


def test_parse_label_maps_keywords():
    assert ai.parse_label("This is a cough.") == "cough"
    assert ai.parse_label("sounds like throat clearing") == "throat_clear"
    assert ai.parse_label("laughter") == "laughter"
    assert ai.parse_label("normal speech") == "speech"
    assert ai.parse_label("unsure / music") == "other"
