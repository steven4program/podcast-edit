# Podcast Edit Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lean Claude Code skill that edits zh-TW audio podcasts — removing stutters/repeats, host coughs/throat-clears, and producing a smooth track — with the LLM making editorial judgments and ~5 thin Python helpers doing the mechanical work.

**Architecture:** One skill (`SKILL.md` + `helpers/` + `references/`). A single canonical `transcript.json` (stable word ids) is the only timing authority; the LLM reads a packed transcript and writes a cut list (`cuts.json`) citing word ids; `render.py` resolves ids → exact times, snaps boundaries to silence, cuts sample-accurately with micro-fades, and verifies no boundary falls mid-word. All artifacts live in `<audio_dir>/edit/`.

**Tech Stack:** Python 3.10+, ffmpeg/ffprobe, ElevenLabs Scribe API (transcription), optional Gemini API (cough/seam AI-listen), pytest.

## How this plan is organized: phases, each tested on real audio

The plan is divided into **6 phases**. Each phase builds its code with fast offline TDD,
then ends with a **MANUAL real-audio gate** — a deliverable you run on a real episode and
verify by ear before starting the next phase. The gates are the primary defense against
gradual deviation: a cut that drifts late in a long episode is something you will *hear* at
the Phase 2 gate (which deliberately tests a cut at ~95% into the episode).

Do not start a phase until the previous phase's gate passes.

- **Phase 1 — Transcribe + Pack** → gate: transcript accurate, no tail drift.
- **Phase 2 — Render** → gate: hand-written cuts at 5/50/95% land precisely and smoothly.
- **Phase 3 — QA** → gate: flags the seams you hear.
- **Phase 4 — Cut-proposal brain** → gate: LLM's auto cuts are correct on a real episode.
- **Phase 5 — Cough removal** → gate: coughs gone, laughs kept.
- **Phase 6 — Chapters + docs + E2E** → gate: clean full run, raw audio → final.

## Global Constraints

- Language of edited content: **zh-TW** (Traditional Chinese). Source is an **audio file** (mp3/wav/m4a); output is audio.
- `transcript.json` word-ids are the **only** timing source. Cuts cite ids; never re-derive indices from any other representation.
- Never cut mid-word: a cut boundary must never fall strictly inside a word's `[start, end]` interval. Snap to word edge, then to nearest silence within a window.
- 3ms micro-fade at every join. Final cut decodes to WAV (sample-accurate); never `-c copy` for the final render.
- Non-verbal removal is precision-first: auto-cut only confident cough/throat-clear; **laughter is never a removal candidate**; ambiguous or speech-overlapping → flag, never auto-cut.
- Repeat-collapse policy: **delete-earlier, keep-later**; protect intentional/emphatic repetition.
- Loudnorm target: **−16 LUFS stereo / −19 LUFS mono**. Output at original quality, never the 16kHz transcription copy.
- Source directory untouched; all outputs under `<audio_dir>/edit/`.
- All file paths in this plan are relative to `/Users/kaiwei/techporn/podcast/new-podcast-skill/`.
- Spec: `docs/superpowers/specs/2026-06-20-podcast-edit-skill-design.md`.
- **You need one real zh-TW episode** (ideally with host coughs and some laughter) available for the manual gates. Call its path `$EP` below.

## File Structure

```
new-podcast-skill/
  SKILL.md                      # philosophy, hard rules, phase workflow (the "brain")
  install.md                    # one-time setup
  README.md                     # short overview
  .env.example                  # ELEVENLABS_API_KEY, GEMINI_API_KEY (optional)
  .gitignore
  pyproject.toml
  helpers/
    __init__.py
    transcribe.py               # Scribe → transcript.json (normalize is the testable core)
    pack.py                     # transcript.json → packed.md (zh-TW aware)
    render.py                   # cuts.json + transcript.json + audio → final.mp3 + kept_transcript.json
    qa.py                       # seam/silence signal check → qa_report.md
    ai_listen.py                # optional Gemini clip classifier
  references/
    edit-heuristics.md          # zh-TW filler list, cut taxonomy, repeat/self-correction/cough policy
  test/
    __init__.py
    fixtures.py                 # synthetic clip + sample transcript builders
    test_pack.py  test_transcribe.py  test_render.py  test_qa.py  test_ai_listen.py
```

---
---

# PHASE 1 — Transcribe + Pack

**Phase deliverable:** run two commands on a real episode and get an accurate `packed.md`
whose timestamps still match the audio at the very end of the file.

---

### Task 1.1: Scaffold, fixtures, and git

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `README.md`, `helpers/__init__.py`, `test/__init__.py`, `test/fixtures.py`

**Interfaces:**
- Produces:
  - `test.fixtures.sample_transcript() -> dict` — small canonical transcript (ids 0..7, two speakers, one cough event).
  - `test.fixtures.make_test_wav(path: str) -> None` — writes a 4s WAV with tone/silence/tone so silence-snapping and rendering run without any API.

- [ ] **Step 1: Initialize git and write `.gitignore`**

Run:
```bash
cd /Users/kaiwei/techporn/podcast/new-podcast-skill && git init
```
Create `.gitignore`:
```
__pycache__/
*.pyc
.env
.venv/
edit/
*.egg-info/
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "podcast-edit"
version = "0.1.0"
description = "Lean zh-TW podcast audio editing skill"
requires-python = ">=3.10"
dependencies = ["requests>=2.31", "numpy>=1.24", "soundfile>=0.12"]

[project.optional-dependencies]
ai = ["google-genai>=1.0.0"]
dev = ["pytest>=8.0"]

[tool.setuptools]
packages = ["helpers"]
```

- [ ] **Step 3: Write `.env.example` and `README.md`**

`.env.example`:
```
ELEVENLABS_API_KEY=
GEMINI_API_KEY=
```
`README.md`:
```markdown
# podcast-edit

Lean Claude Code skill for editing zh-TW audio podcasts: removes stutters/repeats,
host coughs/throat-clears, and renders a smooth track. The LLM makes the editorial
calls; thin Python helpers do transcription, cutting, and QA. See `install.md`.
```

- [ ] **Step 4: Write `test/fixtures.py`**

```python
"""Shared test fixtures: a synthetic transcript and a synthetic WAV clip."""
import numpy as np
import soundfile as sf


def sample_transcript():
    """ids 0..7, two speakers, one cough event sitting in the 1.5-2.2s silence."""
    return {
        "audio": "fixture.wav",
        "duration": 4.0,
        "words": [
            {"id": 0, "text": "我覺得", "start": 0.0, "end": 0.4, "speaker": "host"},
            {"id": 1, "text": "我覺得", "start": 0.4, "end": 0.8, "speaker": "host"},
            {"id": 2, "text": "這個", "start": 0.8, "end": 1.1, "speaker": "host"},
            {"id": 3, "text": "東西", "start": 1.1, "end": 1.5, "speaker": "host"},
            {"id": 4, "text": "對啊", "start": 2.2, "end": 2.6, "speaker": "guest"},
            {"id": 5, "text": "就是", "start": 2.6, "end": 2.9, "speaker": "guest"},
            {"id": 6, "text": "就是", "start": 2.9, "end": 3.2, "speaker": "guest"},
            {"id": 7, "text": "很棒", "start": 3.2, "end": 3.7, "speaker": "guest"},
        ],
        "events": [
            {"type": "cough", "start": 1.6, "end": 1.9, "speaker": "host", "confidence": 0.8},
        ],
    }


def make_test_wav(path, sr=16000):
    """4s mono WAV: tone 0-1.5s, silence 1.5-2.2s, tone 2.2-3.7s, silence to 4s."""
    t = np.arange(int(4.0 * sr)) / sr
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    sig = np.zeros_like(t)
    sig[(t >= 0.0) & (t < 1.5)] = tone[(t >= 0.0) & (t < 1.5)]
    sig[(t >= 2.2) & (t < 3.7)] = tone[(t >= 2.2) & (t < 3.7)]
    sf.write(path, sig.astype(np.float32), sr)
```

- [ ] **Step 5: Install and verify**

Run:
```bash
cd /Users/kaiwei/techporn/podcast/new-podcast-skill && pip install -e ".[dev]" && python -c "import test.fixtures as f; print(len(f.sample_transcript()['words']))"
```
Expected: prints `8`.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "chore: scaffold podcast-edit skill (pyproject, fixtures, spec, plan)"
```

---

### Task 1.2: `pack.py` — transcript → packed.md (zh-TW aware)

**Files:** Create `helpers/pack.py`, `test/test_pack.py`

**Interfaces:**
- Produces:
  - `pack(transcript: dict, gap: float = 0.5) -> str` — phrases break when consecutive-word gap `> gap` OR speaker changes. Line: `[mm:ss–mm:ss #firstId-lastId speaker] <text>`, text concatenated (no separators), half-width `,.!?:;` mapped to full-width.
  - `_mmss(seconds: float) -> str`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_pack.py -v`
Expected: FAIL with `AttributeError: module 'helpers.pack' has no attribute 'pack'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Pack a canonical transcript into a token-efficient zh-TW reading view."""
_PUNCT = {",": "，", ".": "。", "!": "！", "?": "？", ":": "：", ";": "；"}


def _mmss(seconds):
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def _norm(text):
    return "".join(_PUNCT.get(c, c) for c in text)


def pack(transcript, gap=0.5):
    words = transcript["words"]
    if not words:
        return ""
    phrases, cur = [], [words[0]]
    for prev, w in zip(words, words[1:]):
        if w["start"] - prev["end"] > gap or w["speaker"] != prev["speaker"]:
            phrases.append(cur)
            cur = [w]
        else:
            cur.append(w)
    phrases.append(cur)
    lines = []
    for ph in phrases:
        first, last = ph[0], ph[-1]
        text = _norm("".join(w["text"] for w in ph))
        lines.append(f"[{_mmss(first['start'])}–{_mmss(last['end'])} "
                     f"#{first['id']}-{last['id']} {first['speaker']}] {text}")
    return "\n".join(lines)


if __name__ == "__main__":
    import json, sys
    print(pack(json.load(open(sys.argv[1]))))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/test_pack.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add helpers/pack.py test/test_pack.py && git commit -m "feat: pack transcript into zh-TW reading view"
```

---

### Task 1.3: `transcribe.py` — Scribe → canonical transcript.json

**Files:** Create `helpers/transcribe.py`, `test/test_transcribe.py`

**Interfaces:**
- Produces:
  - `normalize_scribe(raw: dict) -> dict` — pure. Maps Scribe response → `{audio, duration, words[], events[]}`. Integer ids only for `type=="word"`; `type=="audio_event"` → `events` with parsed type; event speaker = its `speaker_id` or nearest word.
  - `transcribe(audio_path, out_path, language="zho") -> dict` — extract mono 16k wav, call Scribe, write raw + normalized JSON, return normalized dict.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_transcribe.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'normalize_scribe'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""ElevenLabs Scribe -> canonical transcript.json. normalize_scribe is the testable core."""
import json, os, subprocess, tempfile
import requests

_EVENT_MAP = {"cough": "cough", "throat": "throat_clear", "laugh": "laughter", "breath": "breath"}


def _classify_event(text):
    low = text.lower()
    for key, label in _EVENT_MAP.items():
        if key in low:
            return label
    return "other"


def _nearest_speaker(words, t):
    return min(words, key=lambda w: abs(w["start"] - t))["speaker"] if words else None


def normalize_scribe(raw):
    words, events, next_id = [], [], 0
    for e in raw.get("words", []):
        if e.get("type", "word") == "word":
            words.append({"id": next_id, "text": e["text"], "start": e["start"],
                          "end": e["end"], "speaker": e.get("speaker_id")})
            next_id += 1
        elif e.get("type") == "audio_event":
            events.append({"type": _classify_event(e["text"]), "start": e["start"],
                           "end": e["end"], "speaker": e.get("speaker_id")})
    for ev in events:
        if ev["speaker"] is None:
            ev["speaker"] = _nearest_speaker(words, ev["start"])
    return {"audio": raw.get("audio", ""), "duration": words[-1]["end"] if words else 0.0,
            "words": words, "events": events}


def _extract_wav(audio_path):
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", wav],
                   check=True, capture_output=True)
    return wav


def transcribe(audio_path, out_path, language="zho"):
    key = os.environ["ELEVENLABS_API_KEY"]
    wav = _extract_wav(audio_path)
    try:
        with open(wav, "rb") as f:
            resp = requests.post("https://api.elevenlabs.io/v1/speech-to-text",
                                 headers={"xi-api-key": key},
                                 data={"model_id": "scribe_v1", "language_code": language,
                                       "diarize": "true", "tag_audio_events": "true",
                                       "timestamps_granularity": "word"},
                                 files={"file": f}, timeout=1800)
        resp.raise_for_status()
        raw = resp.json()
    finally:
        os.remove(wav)
    raw["audio"] = audio_path
    with open(out_path.replace(".json", "_raw.json"), "w") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    transcript = normalize_scribe(raw)
    transcript["audio"] = audio_path
    with open(out_path, "w") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    return transcript


if __name__ == "__main__":
    import sys
    transcribe(sys.argv[1], sys.argv[2])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/test_transcribe.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add helpers/transcribe.py test/test_transcribe.py && git commit -m "feat: Scribe transcription with canonical normalization"
```

---

### Task 1.4: 🟢 GATE 1 — Manual real-audio test (Transcribe + Pack)

**This is a manual step. It requires a real episode (`$EP`) and your ears/eyes. Do not proceed to Phase 2 until it passes.**

- [ ] **Step 1: Run the pipeline on a real episode**

```bash
mkdir -p "$(dirname "$EP")/edit"
ELEVENLABS_API_KEY=... python -m helpers.transcribe "$EP" "$(dirname "$EP")/edit/transcript.json"
python -m helpers.pack "$(dirname "$EP")/edit/transcript.json" > "$(dirname "$EP")/edit/packed.md"
```

- [ ] **Step 2: Verify Scribe schema assumptions (one-time)**

Open `edit/transcript_raw.json`. Confirm:
- `type=="word"` entries are whole zh-TW **words/short tokens**, not single characters with one timestamp each. (If they are per-character, that's still usable — note it — but confirm timestamps look right.)
- Audio-event entries exist and their `text` contains words matching `_EVENT_MAP` (cough/throat/laugh/breath). If not, update `_classify_event` and add a fixture for the real shape, then re-run Task 1.3 tests.

- [ ] **Step 3: Accuracy check**

Skim `edit/packed.md`. Text should be a faithful zh-TW transcript; speaker labels should roughly track who's talking.

- [ ] **Step 4: TAIL DRIFT CHECK (the important one)**

Take the **last** non-empty line of `packed.md`, read its start `mm:ss`. Open `$EP` in any player, seek to that time. The words you hear must match that line. Repeat for a line ~75% through.
- **PASS:** words match within ~1s at both points → no tail drift; the timing spine is trustworthy.
- **FAIL:** words are seconds off late in the file → Scribe timestamps drift on this audio. Stop and report; we'll switch the `transcribe.py` seam to local WhisperX before continuing.

---
---

# PHASE 2 — Render

**Phase deliverable:** cut a real episode from a hand-written `cuts.json` and hear that cuts
at 5%, 50%, and 95% of the file all land on the right words, smoothly, with no late drift.

---

### Task 2.1: `render.py` — data core (resolve, kept-segments, remap, verify)

**Files:** Create `helpers/render.py`, `test/test_render.py`

**Interfaces:**
- Produces:
  - `resolve_cut_times(cuts, words) -> list[dict]` — word-id cuts → `words[id].start/end`; event cuts keep `start`/`end`. (anti-misalignment)
  - `compute_kept_segments(cuts, duration) -> list[tuple]` — complement of merged cut intervals.
  - `remap_words(words, segments) -> list[dict]` — kept words shifted onto output timeline.
  - `verify_no_midword(boundaries, words) -> list[float]` — boundaries strictly inside a word (empty == OK).

- [ ] **Step 1: Write the failing test**

```python
import helpers.render as r
from test.fixtures import sample_transcript


def test_resolve_word_cut_uses_exact_word_times():
    cuts = [{"type": "micro", "start_word": 0, "end_word": 0, "reason": "repeat"}]
    out = r.resolve_cut_times(cuts, sample_transcript()["words"])
    assert out[0]["start"] == 0.0 and out[0]["end"] == 0.4


def test_resolve_event_cut_keeps_times():
    cuts = [{"type": "event", "start": 1.6, "end": 1.9, "reason": "cough"}]
    out = r.resolve_cut_times(cuts, sample_transcript()["words"])
    assert out[0]["start"] == 1.6 and out[0]["end"] == 1.9


def test_kept_segments_is_complement():
    cuts = [{"start": 0.0, "end": 0.4}, {"start": 1.6, "end": 1.9}]
    assert r.compute_kept_segments(cuts, 4.0) == [(0.4, 1.6), (1.9, 4.0)]


def test_kept_segments_merges_overlap():
    cuts = [{"start": 0.5, "end": 1.0}, {"start": 0.8, "end": 1.2}]
    assert r.compute_kept_segments(cuts, 2.0) == [(0.0, 0.5), (1.2, 2.0)]


def test_remap_words_shifts_onto_output_timeline():
    kept = r.remap_words(sample_transcript()["words"], [(0.4, 4.0)])
    assert kept[0]["id"] == 1
    assert abs(kept[0]["start"] - 0.0) < 1e-9
    assert abs(kept[-1]["end"] - (3.7 - 0.4)) < 1e-9


def test_verify_flags_midword_boundary():
    words = sample_transcript()["words"]
    assert r.verify_no_midword([0.2], words) == [0.2]
    assert r.verify_no_midword([0.4], words) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_render.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'resolve_cut_times'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Render core. Pure data logic here; audio/ffmpeg added in Task 2.2."""


def resolve_cut_times(cuts, words):
    by_id = {w["id"]: w for w in words}
    out = []
    for c in cuts:
        if c.get("start_word") is not None:
            start, end = by_id[c["start_word"]]["start"], by_id[c["end_word"]]["end"]
        else:
            start, end = c["start"], c["end"]
        out.append({**c, "start": start, "end": end})
    return out


def compute_kept_segments(cuts, duration):
    intervals = sorted((c["start"], c["end"]) for c in cuts)
    merged = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    segs, pos = [], 0.0
    for s, e in merged:
        if s > pos:
            segs.append((pos, s))
        pos = max(pos, e)
    if pos < duration:
        segs.append((pos, duration))
    return segs


def _removed_before(t, segments):
    kept = 0.0
    for s, e in segments:
        if e <= t:
            kept += e - s
        elif s < t < e:
            kept += t - s
    return kept


def remap_words(words, segments):
    kept = []
    for w in words:
        if not any(s <= w["start"] < e for s, e in segments):
            continue
        kept.append({**w, "start": _removed_before(w["start"], segments),
                     "end": _removed_before(w["end"], segments)})
    return kept


def verify_no_midword(boundaries, words):
    bad = []
    for t in boundaries:
        if any(w["start"] < t < w["end"] for w in words):
            bad.append(t)
    return bad
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/test_render.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add helpers/render.py test/test_render.py && git commit -m "feat: render data core (resolve/kept-segments/remap/verify)"
```

---

### Task 2.2: `render.py` — snap to silence + ffmpeg render + duration check

**Files:** Modify `helpers/render.py`, `test/test_render.py`

**Interfaces:**
- Produces:
  - `snap_point(t, silences, window=0.3) -> tuple[float, bool]` — snap to nearest silence-interval edge within window; else unchanged + False.
  - `detect_silences(wav_path, thresh_db=-35.0, min_len=0.3) -> list[tuple]`.
  - `probe_duration(path) -> float` — ffprobe duration in seconds.
  - `render(transcript, cuts, audio_path, out_path, snap_window=0.3) -> dict` — resolve → snap → verify → sample-accurate cut + 3ms fades → loudnorm → write mp3 + `<out>` kept transcript (with `segments`). Returns `{"segments", "snapped", "flagged"}`. Raises `ValueError` on mid-word boundary.

- [ ] **Step 1: Write the failing tests**

```python
import json, os
import helpers.render as r
from test.fixtures import sample_transcript, make_test_wav


def test_snap_point_moves_to_silence_within_window():
    new_t, snapped = r.snap_point(1.45, [(1.5, 2.2)], window=0.3)
    assert snapped and abs(new_t - 1.5) < 1e-9


def test_snap_point_keeps_when_no_silence_in_window():
    new_t, snapped = r.snap_point(0.5, [(1.5, 2.2)], window=0.3)
    assert not snapped and new_t == 0.5


def test_render_output_duration_matches_kept(tmp_path):
    wav, out = str(tmp_path / "in.wav"), str(tmp_path / "out.mp3")
    make_test_wav(wav)
    t = sample_transcript(); t["audio"] = wav
    cuts = [{"type": "micro", "start_word": 0, "end_word": 0, "reason": "repeat 我覺得"}]
    result = r.render(t, cuts, wav, out)
    assert os.path.exists(out)
    kept = json.load(open(out.replace(".mp3", "_kept_transcript.json")))
    assert kept["words"][0]["id"] == 1
    expected = sum(e - s for s, e in result["segments"])
    assert abs(r.probe_duration(out) - expected) < 0.3  # no drift / no extra audio
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test/test_render.py -k "snap or duration" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'snap_point'`.

- [ ] **Step 3: Append implementation to `helpers/render.py`**

```python
import json, os, subprocess


def detect_silences(wav_path, thresh_db=-35.0, min_len=0.3):
    proc = subprocess.run(
        ["ffmpeg", "-i", wav_path, "-af",
         f"silencedetect=noise={thresh_db}dB:d={min_len}", "-f", "null", "-"],
        capture_output=True, text=True)
    starts, sils = [], []
    for line in proc.stderr.splitlines():
        if "silence_start:" in line:
            starts.append(float(line.split("silence_start:")[1].strip()))
        elif "silence_end:" in line:
            end = float(line.split("silence_end:")[1].split("|")[0].strip())
            if starts:
                sils.append((starts.pop(), end))
    return sils


def snap_point(t, silences, window=0.3):
    best, best_d = None, window
    for s, e in silences:
        for edge in (s, e):
            if abs(edge - t) <= best_d:
                best, best_d = edge, abs(edge - t)
    return (best, True) if best is not None else (t, False)


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", path], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def render(transcript, cuts, audio_path, out_path, snap_window=0.3):
    words, duration = transcript["words"], transcript["duration"]
    resolved = resolve_cut_times(cuts, words)
    wav = out_path + ".src.wav"
    subprocess.run(["ffmpeg", "-y", "-i", audio_path, wav], check=True, capture_output=True)
    silences = detect_silences(wav)
    snapped_count = 0
    for c in resolved:
        ns, s1 = snap_point(c["start"], silences, snap_window)
        ne, s2 = snap_point(c["end"], silences, snap_window)
        c["start"], c["end"] = ns, ne
        snapped_count += int(s1) + int(s2)
    segments = compute_kept_segments(resolved, duration)
    boundaries = [b for seg in segments for b in seg if 0 < b < duration]
    flagged = verify_no_midword(boundaries, words)
    parts, labels = [], []
    for i, (s, e) in enumerate(segments):
        dur = e - s
        parts.append(f"[0:a]atrim={s}:{e},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:st=0:d=0.003,afade=t=out:st={max(dur-0.003,0)}:d=0.003[a{i}]")
        labels.append(f"[a{i}]")
    fc = ";".join(parts) + ";" + "".join(labels) + \
        f"concat=n={len(segments)}:v=0:a=1,loudnorm=I=-16:TP=-1.5:LRA=11[out]"
    subprocess.run(["ffmpeg", "-y", "-i", wav, "-filter_complex", fc, "-map", "[out]", out_path],
                   check=True, capture_output=True)
    os.remove(wav)
    kept = {"words": remap_words(words, segments), "segments": segments}
    with open(out_path.replace(".mp3", "_kept_transcript.json"), "w") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    if flagged:
        raise ValueError(f"mid-word boundaries after snap: {flagged}")
    return {"segments": segments, "snapped": snapped_count, "flagged": flagged}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("transcript"); p.add_argument("cuts")
    p.add_argument("audio"); p.add_argument("out")
    a = p.parse_args()
    res = render(json.load(open(a.transcript)), json.load(open(a.cuts))["cuts"], a.audio, a.out)
    print(json.dumps(res, ensure_ascii=False))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/test_render.py -v`
Expected: PASS (all render tests).

- [ ] **Step 5: Commit**

```bash
git add helpers/render.py test/test_render.py && git commit -m "feat: silence-snap, sample-accurate ffmpeg render, duration check, CLI"
```

---

### Task 2.3: 🟢 GATE 2 — Manual real-audio test (Render + the drift test)

**Manual. Requires `$EP`, its `edit/transcript.json` + `edit/packed.md` from Gate 1, and your ears.**

- [ ] **Step 1: Hand-write three spread-out cuts**

Open `edit/packed.md`. Pick three short, safe spans to delete — one **near the start (~5%)**, one **near the middle (~50%)**, one **near the end (~95%)** of the episode. Note each span's word ids from the `#startId-endId` prefixes. Write `edit/cuts.json`:
```json
{"cuts": [
  {"type": "micro", "start_word": <early_start>, "end_word": <early_end>, "reason": "drift test early"},
  {"type": "micro", "start_word": <mid_start>, "end_word": <mid_end>, "reason": "drift test mid"},
  {"type": "micro", "start_word": <late_start>, "end_word": <late_end>, "reason": "drift test late"}
]}
```

- [ ] **Step 2: Render**

```bash
D="$(dirname "$EP")/edit"
python -m helpers.render "$D/transcript.json" "$D/cuts.json" "$EP" "$D/preview.mp3"
```

- [ ] **Step 3: LISTEN at each cut (alignment + smoothness + drift)**

For each of the three cuts, find its approximate output time (use `edit/preview_kept_transcript.json` — the word right after each removed span shows its output `start`). Seek `preview.mp3` there and listen:
- The **intended words are gone** (not neighboring words) — proves correct alignment.
- The **join is smooth** — no click, no chopped syllable, no jarring jump.
- The **late (~95%) cut is as precise as the early one** — this is the gradual-deviation test. If the early cut is clean but the late cut removes the wrong words or sounds off, we have drift → stop and report.

- [ ] **Step 4: Sanity-check duration**

`preview.mp3` length ≈ original minus the three removed spans. (Big mismatch = a render bug.)

- **PASS:** all three cuts correct + smooth, late == early precision → render is trustworthy on real audio. Proceed to Phase 3.
- **FAIL:** capture which cut, what you heard, and the `kept_transcript` times; debug before continuing.

---
---

# PHASE 3 — QA

**Phase deliverable:** an automated seam/silence report on a real render that agrees with what you hear.

---

### Task 3.1: `qa.py` — seam & silence signal check

**Files:** Create `helpers/qa.py`, `test/test_qa.py`

**Interfaces:**
- Produces:
  - `detect_seams(samples, sr, joins, win=0.05, ratio=4.0) -> list[dict]` — RMS-jump flag per join.
  - `joins_from_segments(segments) -> list[float]` — output-timeline join points (cumulative kept lengths, excluding the final end).
  - `qa(audio_path, joins, out_md) -> dict` — load audio, detect seams + residual silences >2s, write markdown report, return findings.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import helpers.qa as qa


def test_detect_seams_flags_energy_jump():
    sr = 16000
    samples = np.concatenate([0.5 * np.ones(sr), 0.001 * np.ones(sr)]).astype(np.float32)
    flags = qa.detect_seams(samples, sr, joins=[1.0])
    assert len(flags) == 1 and flags[0]["ratio"] > 4.0


def test_detect_seams_ignores_smooth_join():
    sr = 16000
    samples = (0.3 * np.ones(2 * sr)).astype(np.float32)
    assert qa.detect_seams(samples, sr, joins=[1.0]) == []


def test_joins_from_segments():
    # segments lengths 0.4, 1.2, 2.1 -> internal joins at 0.4 and 1.6
    segs = [(0.0, 0.4), (1.0, 2.2), (3.0, 5.1)]
    assert qa.joins_from_segments(segs) == [0.4, 1.6]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_qa.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'detect_seams'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Signal-level QA: flag abrupt seams and residual long silences."""
import json
import numpy as np
import soundfile as sf


def _rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if len(x) else 0.0


def _mmss(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def detect_seams(samples, sr, joins, win=0.05, ratio=4.0):
    n, flags = int(win * sr), []
    for t in joins:
        i = int(t * sr)
        before, after = _rms(samples[max(0, i - n):i]), _rms(samples[i:i + n])
        lo, hi = sorted((before + 1e-9, after + 1e-9))
        if hi / lo > ratio:
            flags.append({"time": t, "ratio": hi / lo})
    return flags


def joins_from_segments(segments):
    joins, acc = [], 0.0
    for s, e in segments[:-1]:
        acc += e - s
        joins.append(round(acc, 6))
    return joins


def _long_silences(samples, sr, thresh=0.01, min_len=2.0):
    quiet, out, run = np.abs(samples) < thresh, [], 0
    for i, q in enumerate(quiet):
        if q:
            run += 1
        else:
            if run >= min_len * sr:
                out.append(((i - run) / sr, i / sr))
            run = 0
    return out


def qa(audio_path, joins, out_md):
    samples, sr = sf.read(audio_path)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    seams, sils = detect_seams(samples, sr, joins), _long_silences(samples, sr)
    lines = ["# QA report", "", f"- seams flagged: {len(seams)}"]
    lines += [f"  - [{_mmss(s['time'])}] abrupt seam (ratio {s['ratio']:.1f})" for s in seams]
    lines.append(f"- long silences: {len(sils)}")
    lines += [f"  - [{_mmss(s)}–{_mmss(e)}] residual silence" for s, e in sils]
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    return {"seams": seams, "long_silences": sils}


if __name__ == "__main__":
    import sys
    kept = json.load(open(sys.argv[2]))  # *_kept_transcript.json (has "segments")
    print(json.dumps(qa(sys.argv[1], joins_from_segments(kept["segments"]), sys.argv[3]),
                     ensure_ascii=False))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/test_qa.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add helpers/qa.py test/test_qa.py && git commit -m "feat: signal-level seam and silence QA"
```

---

### Task 3.2: 🟢 GATE 3 — Manual real-audio test (QA)

**Manual. Uses Phase 2's `preview.mp3` + `preview_kept_transcript.json`.**

- [ ] **Step 1: Run QA on the real render**

```bash
D="$(dirname "$EP")/edit"
python -m helpers.qa "$D/preview.mp3" "$D/preview_kept_transcript.json" "$D/qa_report.md"
```

- [ ] **Step 2: Cross-check report against your ears**

Open `qa_report.md`. For each flagged seam, seek `preview.mp3` there — is it actually rough? If you heard a bad seam in Gate 2 that QA did **not** flag, lower `ratio` (e.g. 4.0 → 3.0) in `detect_seams` and re-run. If QA flags many seams that sound fine, raise it. Tune until the report roughly matches your ears.

- **PASS:** report points at the seams you'd care about; no obviously-bad seam goes unflagged. Proceed to Phase 4.

---
---

# PHASE 4 — Cut-proposal brain

**Phase deliverable:** the LLM, reading the skill, proposes a full `cuts.json` for a real
episode that correctly catches repeats/fillers and preserves emphasis — your first full auto run.

---

### Task 4.1: `references/edit-heuristics.md`

**Files:** Create `references/edit-heuristics.md`

- [ ] **Step 1: Write the heuristics doc**

```markdown
# Edit heuristics (zh-TW)

The LLM consults this when proposing cuts. Knowledge, not machinery.

## Fillers (remove)
嗯、呃、啊、欸、那個（贅詞）、然後（贅詞）、就是（贅詞）、對對對（附和填充）

## Repeat-collapse (PRIORITY #1) — delete-earlier, keep-later
Adjacent duplicated word/phrase (exact or near-exact) → keep ONE, the LATER instance.
- `我覺得 我覺得 這個` → delete the first `我覺得`.
- `提到說 就是 就是 這個東西` → delete the first `就是`.
- Self-correction is the same family: `我想說…我要說的是` → keep the corrected later clause.
- PROTECT emphasis/rhetoric: `真的真的很棒`, deliberate doubling → keep. If unsure, keep + flag.

## Stutters / false starts
Partial-word repeats and abandoned starts → remove the broken attempt, keep the clean one.

## Macro deletions (segment-level)
Pre-show prep, off-topic chit-chat, tech debugging ("聽得到嗎"), repeated takes, privacy.
Prefer boundaries at sentence/breath edges.

## Non-verbal (cough/throat-clear) — precision-first  [built in Phase 5]
- Source: transcript.json events (Scribe), attributed to a speaker.
- Focus the configured cough-prone host; flag (don't auto-cut) others.
- Auto-cut only confident cough/throat_clear. LAUGHTER IS NEVER A CANDIDATE.
- Speech-overlapping cough → flag, never auto-cut.

## Dead air
Ignore ≤0.5s. Consider >1s. Long silences also surface in QA.

## Output format
Write edit/cuts.json: {"cuts":[{type, start_word, end_word, reason}, ...]}.
Every cut MUST cite word ids from packed.md (or start/end for events). Never invent times.
```

- [ ] **Step 2: Commit**

```bash
git add references/edit-heuristics.md && git commit -m "docs: zh-TW edit heuristics"
```

---

### Task 4.2: `SKILL.md` — workflow + hard rules (cut-proposal scope)

**Files:** Create `SKILL.md`

- [ ] **Step 1: Write `SKILL.md`**

````markdown
---
name: podcast-edit
description: Edit a zh-TW audio podcast — remove stutters/repeats, host coughs/throat-clears, render a smooth track. Use when the user wants to cut/edit/clean a podcast audio file.
---

# Podcast Edit

You edit zh-TW podcast audio. You make the editorial judgments by reading text;
thin helpers do transcription, cutting, and QA. Work in the folder containing the audio.

## Primary goals (priority order)
1. Remove stutters and repeated words/sentences (delete-earlier, keep-later).
2. Remove throat-clearing/coughing from the cough-prone host (precision-first; never laughs).
3. Make the edited track sound smooth (no clicks/jumps/abrupt seams).

## Hard rules (never violate)
1. transcript.json word-ids are the ONLY timing source. Cuts cite ids; never re-derive.
2. Never cut mid-word. render.py snaps to word edge then nearest silence.
3. 3ms micro-fade at every join; sample-accurate WAV cut; never `-c copy` for final.
4. Non-verbal removal is precision-first. Laughter is never removed. Ambiguous → flag.
5. Always render a preview and get user approval before the final render.
6. After render, verify passed (no mid-word boundary). On ValueError, fix the cut and re-run.
7. Loudnorm -16 LUFS stereo / -19 mono. Original quality out, never the 16k copy.
8. Source dir untouched. All artifacts under `<audio_dir>/edit/`.

## Workflow
0. Setup — confirm ffmpeg + ELEVENLABS_API_KEY. Ask the user the cough-prone host name.
1. Transcribe — `python -m helpers.transcribe <audio> edit/transcript.json`; on the first
   episode confirm `edit/transcript_raw.json` is per-word zh-TW with matching event tags.
   Then `python -m helpers.pack edit/transcript.json > edit/packed.md`.
2. Propose cuts — read `edit/packed.md` and `references/edit-heuristics.md`. Write
   `edit/cuts.json`, each cut citing start_word/end_word (or start/end for events) with
   type and reason. Present a grouped list: `[mm:ss] removed text — reason`.
3. Review gate — `python -m helpers.render edit/transcript.json edit/cuts.json <audio> edit/preview.mp3`.
   User reads the list and listens. Apply changes to cuts.json, re-render. Loop until approved.
4. Final render — same command → `edit/final.mp3` (+ `edit/final_kept_transcript.json`).
5. QA — `python -m helpers.qa edit/final.mp3 edit/final_kept_transcript.json edit/qa_report.md`.
   Review flagged seams. Issues → back to step 3.
6. Chapters — read `edit/final_kept_transcript.json`, write `edit/chapters.txt` (`mm:ss Title`).
7. Memory — append a one-line summary to `edit/project.md`.

## Helpers
- transcribe.py — Scribe → transcript.json (words+speaker+events).
- pack.py — transcript.json → packed.md.
- render.py — cuts.json+transcript.json+audio → final.mp3 + kept_transcript.json.
- qa.py — seam/silence check → qa_report.md.
- ai_listen.py — optional Gemini clip classifier (cough vs laugh).

## Anti-patterns
- Don't invent timestamps — always cite word ids from transcript.json.
- Don't auto-remove anything that might be a laugh.
- Don't skip the preview/approval gate.
- Don't write into the source folder or the skill folder.
````

- [ ] **Step 2: Commit**

```bash
git add SKILL.md && git commit -m "feat: SKILL.md workflow and hard rules"
```

---

### Task 4.3: 🟢 GATE 4 — Manual real-audio test (auto cut proposal)

**Manual. The agent runs the skill on `$EP` end-to-end through the review gate.**

- [ ] **Step 1: Full auto run**

Start the skill in the episode folder. Let it transcribe (reuse cached transcript), read
`packed.md` + heuristics, and write `edit/cuts.json` itself. It renders `edit/preview.mp3`
and presents the grouped cut list.

- [ ] **Step 2: Review the proposed cut list**

Check the list against the heuristics:
- Repeats collapsed keep-later? (`我覺得我覺得` → one kept)
- Fillers caught? Stutters caught?
- **Emphasis preserved?** (`真的真的` kept, not deleted)
- Any clearly wrong removals?

- [ ] **Step 3: Listen to the preview**

Spot-check several cuts across the whole file (including late ones). Then request a couple
of changes in chat ("restore cut at 12:30", "also cut the 然後 at 03:10") and confirm the
agent edits `cuts.json` and re-renders correctly.

- **PASS:** auto cuts are mostly right, emphasis preserved, changes round-trip cleanly. Proceed to Phase 5.

---
---

# PHASE 5 — Cough removal

**Phase deliverable:** on a real episode with the host's coughs, coughs are removed, laughs
are untouched, and ambiguous sounds are flagged not cut.

---

### Task 5.1: `ai_listen.py` — optional Gemini clip classifier

**Files:** Create `helpers/ai_listen.py`, `test/test_ai_listen.py`

**Interfaces:**
- Produces:
  - `parse_label(text) -> str` — reply → `cough/throat_clear/laughter/breath/speech/other`.
  - `classify(audio_path, start, end, model="gemini-2.5-flash") -> str` — extract clip, ask Gemini, return label.

- [ ] **Step 1: Write the failing test**

```python
import helpers.ai_listen as ai


def test_parse_label_maps_keywords():
    assert ai.parse_label("This is a cough.") == "cough"
    assert ai.parse_label("sounds like throat clearing") == "throat_clear"
    assert ai.parse_label("laughter") == "laughter"
    assert ai.parse_label("normal speech") == "speech"
    assert ai.parse_label("unsure / music") == "other"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_ai_listen.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'parse_label'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Optional Gemini clip classifier for cough recall and seam doubt."""
import os, subprocess, tempfile

_KEYWORDS = [("throat", "throat_clear"), ("cough", "cough"), ("laugh", "laughter"),
             ("breath", "breath"), ("speech", "speech"), ("talk", "speech")]
_PROMPT = ("Classify this short audio clip as exactly one of: cough, throat clearing, "
           "laughter, breath, speech. Answer with the single best label.")


def parse_label(text):
    low = text.lower()
    for key, label in _KEYWORDS:
        if key in low:
            return label
    return "other"


def _extract_clip(audio_path, start, end):
    fd, clip = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", audio_path,
                    "-ac", "1", "-ar", "16000", clip], check=True, capture_output=True)
    return clip


def classify(audio_path, start, end, model="gemini-2.5-flash"):
    from google import genai
    clip = _extract_clip(audio_path, start, end)
    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        uploaded = client.files.upload(file=clip)
        resp = client.models.generate_content(model=model, contents=[_PROMPT, uploaded])
        return parse_label(resp.text)
    finally:
        os.remove(clip)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/test_ai_listen.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add helpers/ai_listen.py test/test_ai_listen.py && git commit -m "feat: optional Gemini clip classifier"
```

---

### Task 5.2: Wire cough handling into the workflow

**Files:** Modify `SKILL.md` (expand workflow step 2 with the cough sub-procedure)

- [ ] **Step 1: Replace SKILL.md workflow step 2 with the cough-aware version**

Find the `2. Propose cuts —` line and replace that step with:
````markdown
2. Propose cuts — read `edit/packed.md` and `references/edit-heuristics.md`. Write
   `edit/cuts.json`:
   - Verbal cuts: repeats (keep-later), fillers, stutters, macro segments — cite word ids.
   - Cough/throat-clear: from `transcript.json` `events`. Keep only events whose speaker is
     the configured cough-prone host (flag others in chat, don't cut). For each candidate,
     if `GEMINI_API_KEY` is set, call `helpers.ai_listen.classify(audio, start, end)`; add
     a cut only when the label is `cough` or `throat_clear`. NEVER add `laughter`. Mark
     speech-overlapping candidates (event window overlaps a word) as flagged, not cut.
   Present a grouped list: `[mm:ss] removed text/sound — reason`. Show flagged (not-cut)
   coughs separately for the user to confirm by ear.
````

- [ ] **Step 2: Commit**

```bash
git add SKILL.md && git commit -m "feat: cough-aware cut proposal (precision-first)"
```

---

### Task 5.3: 🟢 GATE 5 — Manual real-audio test (cough removal)

**Manual. Requires `$EP` containing the host's coughs and ideally some laughter, `GEMINI_API_KEY` set.**

- [ ] **Step 1: Run the cough-aware proposal + render**

Let the skill propose cuts (now including cough events) and render `edit/preview.mp3`.

- [ ] **Step 2: Verify on your ears**

- The host's coughs/throat-clears you know about are **gone**.
- **No laughter was removed** (seek to known laughs — still there).
- Ambiguous/soft sounds were **flagged, not silently cut** — check the flagged list.
- No words got clipped where a cough sat next to speech.

- **PASS:** coughs removed, laughs intact, ambiguous ones flagged. Proceed to Phase 6.
- Tune: if soft coughs were missed, that's expected — they're in the flagged/manual long tail.

---
---

# PHASE 6 — Chapters + docs + end-to-end

**Phase deliverable:** a clean full run on a real episode, raw audio → `final.mp3` +
`chapters.txt`, plus install docs so the skill is usable from scratch.

---

### Task 6.1: `install.md` + chapters workflow confirmation

**Files:** Create `install.md` (chapters step already in SKILL.md workflow step 6)

- [ ] **Step 1: Write `install.md`**

```markdown
# Install

1. `pip install -e ".[dev,ai]"` (drop `ai` to skip Gemini).
2. Ensure `ffmpeg` and `ffprobe` are on PATH (`ffmpeg -version`).
3. `cp .env.example .env`; set `ELEVENLABS_API_KEY` (and optional `GEMINI_API_KEY`).
4. Register the skill: `ln -s "$PWD" ~/.claude/skills/podcast-edit`.
5. Verify: `pytest -q` (all tests pass).
```

- [ ] **Step 2: Run the full offline test suite**

Run: `pytest -q`
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add install.md && git commit -m "docs: install guide"
```

---

### Task 6.2: 🟢 GATE 6 — Manual real-audio test (full end-to-end)

**Manual. A clean run from scratch on `$EP`.**

- [ ] **Step 1: Full pipeline from raw audio**

Run the skill start to finish: transcribe → propose cuts (verbal + cough) → review/approve →
final render → QA → chapters. Produce `edit/final.mp3`, `edit/qa_report.md`, `edit/chapters.txt`.

- [ ] **Step 2: Final acceptance check**

- Whole track listened/spot-checked: smooth throughout, no late drift, repeats/fillers/coughs
  handled, laughs intact.
- `chapters.txt` timestamps land on real topic changes in `final.mp3`.
- `qa_report.md` has no unaddressed serious seams.
- `final.mp3` loudness is reasonable; source folder untouched; everything under `edit/`.

- **PASS:** the skill is done and trustworthy on real audio.

---

## Self-Review

**1. Spec coverage:**
- Primary goal 1 (stutters/repeats) → heuristics (4.1) + LLM proposal (4.2) + Gate 4. ✓
- Primary goal 2 (host cough/throat-clear) → events (1.3) + ai_listen (5.1) + cough workflow (5.2) + Gate 5. ✓
- Primary goal 3 (smooth) → snap+fade+sample-accurate (2.2) + verify (2.1) + seam QA (3.1) + Gates 2,3. ✓
- Data model (transcript/packed/cuts/kept_transcript) → Phases 1–2. ✓
- 5 helpers → Tasks 1.2,1.3,2.x,3.1,5.1. ✓
- Hard rules → Global Constraints + SKILL.md (4.2). ✓
- Workflow phases 0–7 → SKILL.md (4.2, 5.2, 6.1). ✓
- Review gate (preview mp3) → render (2.2) + workflow + Gate 4. ✓
- Chapters → SKILL.md step 6 + Gate 6. ✓
- Real-audio per-phase testing (user's explicit ask) → Gates 1–6, with tail-drift check in Gate 2. ✓
- Local-whisper fallback → documented seam (1.3 + Gate 1 FAIL path). ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step has complete code. Manual gates use explicit checklists, not vague instructions. ✓

**3. Type consistency:** `resolve_cut_times`/`compute_kept_segments`/`remap_words`/`verify_no_midword` (2.1) reused in `render` (2.2). `snap_point`/`detect_silences`/`probe_duration` (2.2) used in `render`. `render` writes `kept_transcript.json` with `segments`; `qa` CLI (3.1) reads that key via `joins_from_segments`. `parse_label`/`classify` (5.1) consistent. Cut dict shape (`start_word`/`end_word`/`start`/`end`/`type`/`reason`) consistent across 2.1, 2.2, 4.x, 5.2. ✓

**Note on hard rule 6 wording:** the spec said "boundary on a real word edge"; the plan implements the precise invariant "no boundary strictly inside a word" (`verify_no_midword`), which correctly allows silence-snapped boundaries that sit in gaps between words.
