# Podcast Edit Skill — Design Spec

**Date:** 2026-06-20
**Status:** Approved for planning
**Author:** kaiwei (with Claude)

A lean, owned-from-scratch Claude Code skill for editing Traditional Chinese (zh-TW)
audio podcasts. Replaces the heavy `podcast-edit-skill` (8k lines, opaque) by pushing
all editorial *judgment* into the LLM reading text, and confining scripts to the
mechanical, dangerous work (transcription, sample-accurate cutting, signal QA).

---

## Primary success criteria

Everything in this spec serves three goals, in priority order:

1. **Remove stutters and repeated words/sentences** — e.g. `我覺得 我覺得` → one `我覺得`;
   `提到說就是就是這個東西` → one `就是`. This is the core, most frequent edit.
2. **Remove throat-clearing and coughing from the cough-prone host** — precision-first,
   never delete laughter.
3. **The edited track sounds smooth** — no audible clicks, jumps, or unnatural seams.

A build that nails these three is a success. Everything else is supporting machinery.

---

## Design principles

- **LLM judges, scripts execute.** Cut decisions (what to remove and why) are LLM work
  over text. Scripts only transcribe, cut audio at exact timestamps, mix, and signal-check.
  This is how the whole thing stays ~5 thin helpers instead of 8k lines of encoded rules.
- **One source of truth, no re-derivation.** A single canonical word list with stable ids
  is the only timing authority. This structurally prevents the old skill's "text says one
  thing, cut lands elsewhere" bug.
- **Audio is cheap — render, don't simulate.** Concatenating audio segments is seconds, so
  "preview" and "final" are the same fast render path. No custom dynamic-skip player.
- **Precision over recall for destructive non-verbal cuts.** Wrongly deleting a laugh is far
  worse than missing a soft cough. When unsure, keep + flag.
- **Lean and ownable.** One skill, five helpers, one reference doc, one test. Every piece
  fits in your head.

### Non-goals (v1)

- No video. Audio in, audio out.
- No feedback-learning loop (no auto precision/recall, no auto-updated rules).
- No music beds, show notes, titles, or highlight clips. (Chapters only.)
- No interactive HTML review UI (text list + preview mp3 instead).
- No per-speaker loudness alignment (global loudnorm only) — revisit if needed.

---

## Architecture

One Claude Code skill, `podcast-edit`:

- `SKILL.md` carries philosophy, hard rules, and the phase workflow.
- Five thin Python helpers in `helpers/` wrap only mechanical/dangerous work.
- All session artifacts go in `<audio_dir>/edit/`. The source folder and the skill
  directory stay pristine.

### Data model (the spine)

**`transcript.json`** — canonical, from `transcribe.py`. The only timing authority.
Every word has a stable integer id:
```json
{
  "audio": "path/to/original.m4a",
  "duration": 7234.5,
  "words": [
    {"id": 0, "text": "今天", "start": 1.20, "end": 1.55, "speaker": "host"}
  ],
  "events": [
    {"type": "cough", "start": 12.3, "end": 12.8, "speaker": "host", "confidence": 0.7},
    {"type": "laughter", "start": 45.1, "end": 46.0, "speaker": "guest"}
  ]
}
```
Nothing downstream re-derives indices from this. Events carry an attributed speaker
(nearest speaking region by timestamp).

**`packed.md`** — the LLM's reading view, from `pack.py`. Phrases (break on ≥0.5s silence
or speaker change), each line prefixed with real times and the word-id range, then zh-TW
text (no inter-token spaces, full-width punctuation):
```
[12:03–12:09 #1204-1233 host] 我覺得我覺得這個東西其實沒有那麼複雜
```
Because the word-ids are inline, a cut the LLM picks cites those exact ids → the cut uses
those exact times. Text and cut cannot drift apart.

**`cuts.json`** — the edit decision list. Each cut cites word ids, or raw times for events:
```json
{
  "cuts": [
    {"id": "c1", "type": "macro", "start_word": 120, "end_word": 245, "reason": "pre-show chit-chat"},
    {"id": "c2", "type": "micro", "start_word": 1204, "end_word": 1207, "reason": "repeat 我覺得 (keep-later)"},
    {"id": "c3", "type": "event", "start": 12.3, "end": 12.8, "reason": "cough (host)", "confidence": 0.82}
  ]
}
```
Kept audio = the complement of the cuts.

**`kept_transcript.json`** — emitted by the render; kept words remapped to the *output*
timeline. Feeds chapters and verify.

### The five helpers (thin)

| Helper | Does only this |
|---|---|
| `transcribe.py` | ElevenLabs Scribe → `transcript.json` (words + diarization + events). The swappable seam; local-whisper fallback documented behind the same JSON output. |
| `pack.py` | `transcript.json` → `packed.md` (zh-TW aware: no inter-token spaces, full-width punctuation). |
| `render.py` | `cuts.json` + `transcript.json` + audio → `final.mp3` + `kept_transcript.json`. Owns boundary-snap, 3ms micro-fades, WAV-accurate cutting, loudnorm, and the built-in verify. |
| `qa.py` | Seam/silence signal check on the render → `qa_report.md` with clickable timestamps. |
| `ai_listen.py` | Gemini clip classifier (cough vs throat-clear vs laugh vs breath vs speech). Used for cough recall and QA seam doubt. Optional (needs `GEMINI_API_KEY`). |

No separate snap/chapters/verify scripts: snap and verify live inside `render.py`; chapters
and the cut decisions are LLM work over the JSON.

---

## Hard rules (non-negotiable; enforced in SKILL.md + render)

1. `transcript.json` word-ids are the only timing source. Cuts cite ids; never re-derive indices.
2. Never cut mid-word → snap to the word edge, then to the nearest silence within a window.
3. 3ms micro-fade at every join; decode to WAV for sample-accurate cuts (never `-c copy` for final).
4. Non-verbal removal is precision-first: auto-cut only confident cough/throat-clear; never
   laughter; flag ambiguous and speech-overlapping cases.
5. Always render a preview and get user approval before the final render.
6. Verify after render: every boundary lands on a real word edge; kept segments are contiguous;
   mismatch → abort + report.
7. Loudnorm −16 LUFS stereo / −19 mono; output at original quality, never the 16kHz transcription copy.
8. Source dir untouched; everything under `<audio>/edit/`.

---

## Cut taxonomy & policies

The LLM authors `cuts.json` in three families:

**Macro** (segment-level): pre-show prep, off-topic chit-chat, tech debugging, repeated takes,
privacy bits. Prefer boundaries at sentence/breath edges.

**Micro** (word/phrase-level):
- **Repeat-collapse** (priority #1): adjacent duplicated word/phrase, exact or near-exact →
  keep **one**. Policy: **delete-earlier, keep-later**. For `…提到說 [就是]ₐ [就是]ᵦ 這個東西`,
  delete `就是ₐ`, keep `就是ᵦ`. This is the *smoothness* rule, not an arbitrary pick:
  - The lead-in join `提到說 → 就是ᵦ` matches the original coarticulation (`提到說` was already
    followed by an identical `就是`), so no seam is audible.
  - The lead-out `就是ᵦ → 這個東西` is the speaker's own untouched continuation.
  - Generalizes to self-corrections (`我想說…我要說的是` → keep the corrected later version).
  - **Precision guard:** do not strip *intentional/emphatic* repetition (`真的真的很棒`,
    rhetorical doubling). When it reads as emphasis, keep it; flag if unsure.
- **Fillers**: 嗯 / 呃 / 啊 / 然後 / 那個 and similar (zh-TW list in `references/edit-heuristics.md`).
- **Stutters**: false starts, partial-word repeats.
- **Dead air**: silences beyond threshold (configurable; default suggest >1s, ignore ≤0.5s).

**Event** (non-verbal): cough / throat-clear (success criterion #2):
- Primary detector: Scribe audio events, attributed to a speaker by timestamp proximity.
- Focus on the configurable **cough-prone host**; still flag coughs attributed to others
  rather than ignoring them.
- Recall booster: `ai_listen.py` classifies each candidate clip (and signal-detected
  suspicious bursts) → cough / throat-clear / laugh / breath / speech.
- Auto-cut only confident cough/throat-clear. **Laughter is never a candidate.** Ambiguous
  or speech-overlapping → flag for the human, never auto-cut.

---

## Smoothness mechanics (success criterion #3)

Every cut boundary passes through, inside `render.py`:
1. Snap to word edge (from `transcript.json`).
2. Snap to nearest silence/energy-valley within a small window so joins happen in quiet.
   If no silence in the window, keep the word-edge cut and flag in QA.
3. Small kept-padding so words don't collide; 3ms micro-fade at the join to kill PCM clicks.
4. Sample-accurate: decode to WAV, cut, concat, re-encode (never lossy `-c copy` for final).

Then `qa.py` measures energy/spectral discontinuity at every join and flags any seam that
still sounds abrupt; `ai_listen.py` can double-check flagged seams.

---

## Workflow (phases in SKILL.md)

0. **Setup** — check `ffmpeg`, Python deps, `ELEVENLABS_API_KEY` (+ optional `GEMINI_API_KEY`).
   One-time `install.md`.
1. **Ingest & transcribe** — `transcribe.py` → `transcript.json` (cached; never re-transcribe);
   `pack.py` → `packed.md`. On the first-ever run, verify Scribe's zh-TW timestamps are
   per-word-usable on one real clip before trusting the pipeline.
2. **Propose cuts** — LLM reads `packed.md`, writes `cuts.json` across the taxonomy above.
   Presents a grouped markdown list: `[mm:ss] removed text — reason (confidence)`.
3. **Review gate** — `render.py` → `preview.mp3`. User reads the list + listens, requests
   changes in chat → LLM edits `cuts.json` → re-render (seconds). Loop until approved.
4. **Final render** — `render.py` → `final.mp3` + `kept_transcript.json`. Runs snap, fades,
   WAV-accurate cut, loudnorm, then verify.
5. **QA** — `qa.py` → `qa_report.md` (abrupt seams, residual long silences) with clickable
   timestamps; `ai_listen.py` double-checks flagged seams. Issues → back to step 3.
6. **Chapters** — LLM reads `kept_transcript.json` (output-timeline times) → `chapters.txt`
   (`mm:ss Title`). Optionally embed as ID3 chapters.
7. **Session memory** — one-line summary appended to `<audio>/edit/project.md`. The only
   persistence; no learning loop.

---

## Error handling

- Scribe fails, or zh-TW timestamp granularity is unusable → fall back to local-whisper
  behind the same `transcript.json` seam; warn the user.
- No silence in a cut's snap window → keep word-edge cut + micro-fade, flag in QA.
- Cough overlapping speech → flag, never auto-cut.
- Verify mismatch (boundary off a word edge, non-contiguous kept segments) → abort final, report.

---

## Testing (lean — just the money path)

One `test/test_render.py` over a small fixture (short `transcript.json` + short clip). Assert:
- A known cut removes exactly the intended word-id span.
- Every output boundary lands on a real word edge.
- Snap lands in silence when silence exists in the window; otherwise falls back to the word edge.

This is the correctness core (the two historical bugs + smoothness). Nothing else needs a test.

---

## Repo layout

```
new-podcast-skill/
  SKILL.md                      # philosophy, hard rules, phase workflow
  install.md
  .env.example                  # ELEVENLABS_API_KEY, GEMINI_API_KEY (optional)
  pyproject.toml
  helpers/
    transcribe.py  pack.py  render.py  qa.py  ai_listen.py
  references/
    edit-heuristics.md          # zh-TW filler list, cut taxonomy, repeat/self-correction policy
  test/
    test_render.py
```

`references/edit-heuristics.md` is where the old skill's 12 rule files collapse into one doc
the LLM consults — knowledge without machinery.

---

## Assumptions to confirm at spec review

- Language is **zh-TW** (Traditional Chinese).
- The **cough-prone host** is a known, configurable speaker; coughs from others are flagged,
  not auto-cut.
- ElevenLabs Scribe is the v1 transcription engine; local Whisper is a documented fallback
  behind the same JSON seam. (Scribe's zh-TW per-word timestamp granularity is verified on a
  real clip during the first run before committing.)
- Existing `GEMINI_API_KEY` (from the old skill) is available for `ai_listen.py`.
```