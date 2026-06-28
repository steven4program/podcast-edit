# AGENTS.md

Guidance for any AI agent working **on this repository**. Claude Code loads it via `CLAUDE.md`
(which imports this file), and other tools read `AGENTS.md` directly — one source of truth.

For the *runtime* podcast-editing workflow (what the agent does to edit an episode), see
`SKILL.md`. This file is about developing and maintaining the code.

## What this is
A lean Claude Code skill that edits zh-TW (Mandarin) **multitrack** podcast audio: removes
stutters / repeats / false-starts and the host's coughs / throat-clears, then renders a smooth
track. The LLM makes every editorial judgment by reading text; thin Python helpers do only the
mechanical work (transcribe, detect, cut, mix).

## Architecture
- **The LLM is the editor; helpers are dumb tools.** Helpers never decide *what* to cut — they
  transcribe, surface candidates, and render. The agent reads `work/packed.md` + the candidate
  lists and writes `cuts.json`.
- **Recall-first detection, one mechanism per category** (the agent then *judges* each candidate):
  - fillers (呃嗯啊欸) — `fillers.py`, acoustic (Scribe's filler timestamps are unreliable).
  - cough / throat-clear — `ai_listen.py`, a Gemini sweep over the host's *own* track.
  - repeats / `-` stutters / `——` false-starts — `repeats.py`, deterministic adjacent-dup scan.
  - The judging step drops false positives: emphasis (`非常非常多`), names (`萬萬`/`汪汪`),
    reduplicated words (`剛剛`), rhetoric (`懂A懂B懂B懂A`).
- **`render.py` makes output clean by construction** (this is why there is no QA step):
  cuts cite word-ids as the only timing source; it refuses a mid-word boundary on any track
  (`verify_no_midword`); word-id cuts skip silence-snapping and align the cut end to the next
  word's onset; every join gets a 3ms fade; output is loudnormed (-16 LUFS). Per-track cut → mix,
  so a single track can be muted (host cough) without touching the others.

## Layout
- `SKILL.md` — the runtime workflow + hard rules.
- `references/edit-heuristics.md` — zh-TW editing knowledge the agent consults when cutting.
- `helpers/` — one job each: `transcribe`, `pack`, `repeats`, `fillers`, `render`, `ai_listen`.
- `test/` — pytest, one file per helper; `fixtures.py` has a synthetic transcript + WAV.
- `docs/` — the implementation plan and design spec.
- `<audio_dir>/edit/` — per-episode artifacts, **tiered** (gitignored):
  top = durable state (`transcript.json`, `cuts.json`, `project.md`, `flagged.md`);
  `work/` = regeneratable intermediates (`*_raw.json`, `packed.md`, `preview.mp3`);
  `out/` = deliverables (`final.mp3`, `chapters.txt`).

## Setup & tests
```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ai,dev]"
cp .env.example .env   # ELEVENLABS_API_KEY (Scribe), GEMINI_API_KEY (cough)
```
Needs `ffmpeg` / `ffprobe` on PATH. **Run tests: `pytest test/ -q`.** Tests are offline — they
use synthetic fixtures and exercise the pure timeline / detection logic. Keep them that way (no
network or API calls in tests).

## Conventions
- **Simplicity first.** Minimum code that works; no speculative abstraction or config. Match the
  surrounding style. Prefer deleting over adding — the QA step and ~11 MB of scratch were removed
  because they earned their keep no longer.
- **Every non-trivial change leaves one runnable check** (an assert-level test per branch / parser
  / cut path). Trivial one-liners don't need a test.
- **Don't break render's invariants** (word-id-only timing, the mid-word guard, 3ms fades,
  onset-alignment, no-snap for micro cuts). They are the seam guarantee; `test_render.py` locks
  them — run it after any render change.
- Commit at a working checkpoint with a clear, scoped message; keep diffs surgical.
