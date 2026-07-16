import helpers.review as rv
from test.fixtures import sample_transcript


def _spec():
    return {"cuts": [
        {"type": "micro", "start_word": 0, "end_word": 0, "reason": "repeat 「我覺得」 — delete-earlier"},
        {"type": "macro", "start": 2.55, "end": 2.95, "reason": "filler 就是"},
    ], "mutes": [{"speaker": "host", "start": 1.6, "end": 1.9}]}


def test_annotate_marks_removed_words_only():
    t = sample_transcript()
    from helpers.render import resolve_cut_times
    tagged = rv.annotate(t["words"], resolve_cut_times(_spec()["cuts"], t["words"]))
    removed = [w["id"] for w, c in tagged if c is not None]
    assert 0 in removed and 5 in removed          # word cut + macro-span cut
    assert 1 not in removed and 7 not in removed  # kept copy and untouched word


def test_rows_keep_overlapping_speakers_on_separate_whole_lines():
    # Two speakers talking over each other: merged word order interleaves a/b/a/b.
    # Each speaker's sentence must stay on ONE row, not shatter at every alternation.
    t = {"duration": 3.0, "words": [
        {"id": 0, "text": "一個", "start": 0.0, "end": 0.3, "speaker": "a"},
        {"id": 1, "text": "哦", "start": 0.1, "end": 0.2, "speaker": "b"},
        {"id": 2, "text": "小時", "start": 0.3, "end": 0.6, "speaker": "a"},
        {"id": 3, "text": "不好意思", "start": 0.4, "end": 0.9, "speaker": "b"},
        {"id": 4, "text": "錄完", "start": 0.6, "end": 0.9, "speaker": "a"},
    ]}
    rows, _, _ = rv._rows(t, {"cuts": [], "mutes": []})
    assert len(rows) == 2
    assert "一個小時錄完" in rows[0] and "哦不好意思" in rows[1]


def test_build_html_shows_cuts_mutes_and_seek_data():
    page = rv.build_html(sample_transcript(), _spec(), "orig.mp3", "edited.mp3")
    assert '<del class="repeat"' in page and "我覺得" in page
    assert '<del class="filler"' in page
    assert "🔇" in page and "1.6" in page           # mute badge at its time
    assert 'data-t="0.00"' in page                  # click-to-seek anchors
    assert "const SEGS = [[" in page                # kept segments for edited-time remap
    assert 'src="orig.mp3"' in page and 'src="edited.mp3"' in page


def test_build_html_uses_rendered_segments_for_seek_map():
    # The seek map must come from the segments the edited audio was ACTUALLY rendered
    # with (render snaps macro cuts; recomputing here drifted >1s by episode end).
    page = rv.build_html(sample_transcript(), _spec(), "o.mp3", "e.mp3",
                         segments=[[0.5, 2.0], [2.5, 4.0]])
    assert "const SEGS = [[0.5, 2.0], [2.5, 4.0]]" in page


def test_build_html_has_done_button_wired_to_export():
    page = rv.build_html(sample_transcript(), _spec(), "orig.mp3", "edited.mp3")
    assert 'id="done"' in page                       # the 完成 button exists
    assert 'fetch("export"' in page                  # …and POSTs to the serve endpoint


def test_build_html_multi_version_players_and_labeled_mutes():
    spec = _spec()
    spec["mutes"] = [  # same clear found by both detectors + one OpenAI-only
        {"speaker": "host", "start": 1.6, "end": 1.9, "label": "OpenAI"},
        {"speaker": "host", "start": 1.65, "end": 1.9, "label": "Scribe"},
        {"speaker": "host", "start": 3.4, "end": 3.6, "label": "OpenAI"},
    ]
    page = rv.build_html(sample_transcript(), spec, "orig.mp3",
                         [("OpenAI", "a.mp3"), ("Scribe", "b.mp3")])
    assert page.count('class="edit"') == 2          # one player per version
    assert "OpenAI+Scribe" in page                  # overlapping mutes -> one merged badge
    assert page.count("清喉嚨/咳嗽 靜音") == 2      # merged badge + the OpenAI-only one
    # version comparison affordances: per-badge labels, single-version highlight, toggle
    assert 'data-labels="OpenAI,Scribe"' in page and 'data-labels="OpenAI"' in page
    assert 'class="mute only"' in page              # the OpenAI-only mute stands out
    assert page.count('data-v=') == 3               # 全部 + one button per version


def test_original_mix_is_peak_limited(tmp_path):
    # Two hot tracks summed with amix normalize=0 exceed 0dBFS; the 'before' reference
    # must not clip into distortion (it would mislead A/B listening) — the transient
    # limiter caps the sum at its -2dB ceiling instead.
    import numpy as np, soundfile as sf
    sr = 16000
    tone = (0.9 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)
    a, b = str(tmp_path / "a.wav"), str(tmp_path / "b.wav")
    sf.write(a, tone, sr); sf.write(b, tone, sr)
    out = rv.build_original_mix([a, b], str(tmp_path / "orig.wav"))
    x, _ = sf.read(out)
    assert 0.5 < float(np.abs(x).max()) <= 0.85  # limited (~-2dB), not 1.0-clipped


def test_parse_range_forms_and_clamping():
    # <audio> seeking depends on 206 ranges; cover the three header forms + junk.
    assert rv.parse_range("bytes=0-99", 1000) == (0, 99)
    assert rv.parse_range("bytes=200-", 1000) == (200, 999)   # open-ended (browser seek)
    assert rv.parse_range("bytes=-100", 1000) == (900, 999)   # suffix form
    assert rv.parse_range("bytes=500-9999", 1000) == (500, 999)  # end clamped to file
    assert rv.parse_range("bytes=1000-", 1000) is None        # unsatisfiable
    assert rv.parse_range(None, 1000) is None
    assert rv.parse_range("bytes=-", 1000) is None
