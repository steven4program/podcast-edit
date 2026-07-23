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
  - fillers (呃嗯啊欸) — `fillers.py`, acoustic (Scribe's filler timestamps are unreliable);
    laughter-safe (an extent overlapping a laughter event is flagged, never cut).
  - cough / throat-clear — `ai_listen.py`. **Currently Scribe-only**: `scribe_events`
    reuses the Scribe event tags in transcript.json (`--provider scribe`, the CLI default;
    keyless). The higher-recall audio-LLM sweep machinery (`sweep_track`/`classify`) stays,
    but both backends (Gemini, OpenAI) are commented out in `ai_providers.py` by user
    request (2026-07-11) — re-enable there + restore the dep in pyproject `[ai]`. Detection
    logic is backend-agnostic (injected provider); both paths share verify/split.
  - repeats / `-` stutters / `——` false-starts — `repeats.py`, deterministic adjacent-dup scan.
  - dead air — `deadair.py`, word-union gap scan (nobody talking ≥1.2s → shorten to 0.8s);
    a gap overlapping a laughter event is never proposed, and boundaries are acoustically
    slid off any real sound (`refine_boundaries` — token times lie: fillers get misplaced
    near-zero-width tokens, and too-quiet sounds have no tokens at all).
  - The judging step drops false positives: emphasis (`非常非常多`), names (`萬萬`/`汪汪`),
    reduplicated words (`剛剛`), rhetoric (`懂A懂B懂B懂A`).
- **`render.py` makes output clean by construction** (this is why there is no QA step):
  cuts cite word-ids as the only timing source; it refuses a mid-word boundary on any track
  (`verify_no_midword`); word-id cuts skip silence-snapping and align the cut end to the next
  word's onset; every join gets a 3ms fade; each track is speech-leveled before the mix
  (a static per-track speech gain + downward-only compand compression — evens speaker gaps
  and sudden loud/quiet with NO time-varying gain: a silence-gated dynamic leveler
  (dynaudnorm) manufactures a fade-out at every phrase tail, and a boost curve steepens
  tail decay the same way) and the mix is mastered to -16 LUFS with a measured STATIC
  gain plus a transient peak limiter (never in-graph loudnorm: single-pass loudnorm is
  itself a dynamic normalizer and silently upsamples wav output to 192kHz). Per-track cut → mix,
  so a single track can be muted (host cough) without touching the others.

## Layout
- `SKILL.md` — the runtime workflow + hard rules.
- `references/edit-heuristics.md` — zh-TW editing knowledge the agent consults when cutting.
- `helpers/` — one job each: `transcribe` (runs `align` at the end — required), `align` (MMS
  forced-alignment re-timing — fixes Scribe's collapsed word times at the source), `pack`,
  `repeats`,
  `fillers`, `discourse` (context-dependent markers 对/好/然后…), `deadair`, `render`,
  `ai_listen`, `ai_providers` (pluggable cough-sweep backends), `review` (original-vs-edited
  listening page).
- `test/` — pytest, one file per helper; `fixtures.py` has a synthetic transcript + WAV.
- `docs/` — the implementation plan and design spec.
- `<audio_dir>/edit/` — per-episode artifacts, **tiered** (gitignored):
  top = durable state (`transcript.json`, `cuts.json`, `project.md`, `flagged.md`);
  `work/` = regeneratable intermediates (`*_raw.json`, `packed.md`, `preview.mp3`);
  `out/` = deliverables (`final.mp3`, `chapters.txt`).

## Setup & tests
```
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"        # core deps include torch/torchaudio/pypinyin (alignment)
python -m helpers.align --download   # fetch the ~1.2GB alignment model once (else the first
                                     # transcription downloads it mid-run)
cp .env.example .env   # ELEVENLABS_API_KEY (Scribe; cough detection is Scribe-only for now)
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
- **Cross-platform (macOS/Linux/Windows).** Keep helpers OS-agnostic: text file I/O passes
  `encoding="utf-8"` at every call site; CLI stdio is forced to UTF-8 in `helpers/__init__.py`
  (Windows console/pipe defaults to cp950 and raises on zh-TW); source paths are stored
  posix-style (`_posix`) so a transcript written on one OS renders on another; the ffmpeg
  filtergraph is passed via `-filter_complex_script` (Windows caps argv at ~32k chars).
- `README.md` (English) and `README.zh-TW.md` (繁中) are parallel — update **both** when either
  changes.
- Commit at a working checkpoint with a clear, scoped message; keep diffs surgical.
