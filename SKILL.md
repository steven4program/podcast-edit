---
name: podcast-edit
description: Edit a zh-TW audio podcast — remove stutters/repeats, host coughs/throat-clears, render a smooth track. Use when the user wants to cut/edit/clean a podcast audio file.
---

# Podcast Edit

You edit zh-TW podcast audio. You make the editorial judgments by reading text;
thin helpers do transcription, cutting, and QA. Work in the folder containing the audio.

Input is **multitrack**: one mono track per speaker (e.g. `…--ted35.wav` + `…--guestNNN--<name>.wav`),
all the same length/session. Speaker labels come from each filename's last `--` segment, so
they are 100% accurate — no diarization guessing.

## Primary goals (priority order)
1. Remove stutters and repeated words/sentences (delete-earlier, keep-later).
2. Remove throat-clearing/coughing from the cough-prone host (precision-first; never laughs). [Phase 5]
3. Make the edited track sound smooth (no clicks/jumps/abrupt seams).

## Hard rules (never violate)
1. transcript.json word-ids are the ONLY timing source. Cuts cite ids; never re-derive times.
2. Never cut mid-word — and in multitrack, never cut where ANY speaker is mid-word. Pick cut
   boundaries at word-union gaps (a moment everyone is between words). `render` verifies this
   and raises on a mid-word boundary; on that error, move the boundary to the nearest gap.
3. Per-track cut → mix; 3ms micro-fade at every join; sample-accurate. Never `-c copy` for final.
4. Non-verbal removal is precision-first. Laughter is never removed. Ambiguous → flag.
5. Always render a preview and get user approval before the final render.
6. After render, confirm verify passed (no mid-word boundary). On ValueError, fix the cut and re-run.
7. Loudnorm -16 LUFS (render applies this). Render from the original tracks, never a 16k copy.
8. Source tracks untouched. All artifacts under `<audio_dir>/edit/`.

## Workflow
0. Setup — confirm `ffmpeg` + `ELEVENLABS_API_KEY`. Ask the user the cough-prone host name.
1. Transcribe — `python -m helpers.transcribe <tracks_dir> edit/transcript.json` (per-track
   Scribe v2 with a VAD silence pre-pass + merge). On the first episode, confirm a
   `edit/transcript_<speaker>_raw.json` is per-word zh-TW with matching event tags.
   Then `python -m helpers.pack edit/transcript.json > edit/packed.md`.
2. Propose cuts — read `edit/packed.md` and `references/edit-heuristics.md`. Write
   `edit/cuts.json` as `{"cuts": [...], "mutes": [...]}`:
   - Verbal cuts: repeats (keep-later), stutters, macro segments — cite word ids.
   - Fillers: use `helpers.fillers.propose_filler_cuts` (Scribe filler timestamps are
     unreliable; it finds the sound acoustically and flags the unsafe ones).
   - Cough/throat-clear (precision-first): candidates = `transcript.json` `events` of type
     `cough`/`throat_clear` whose speaker is the configured cough-prone host. LAUGHTER IS
     NEVER A CANDIDATE. If `GEMINI_API_KEY` is set, confirm each with
     `helpers.ai_listen.classify(host_track, start, end)` — proceed only on `cough`/
     `throat_clear`; a `laughter`/`speech` label means keep+flag. Remove a confirmed cough by
     a **mute** on the host's track for the event span (no time removed, other speakers
     untouched) — handles speech-overlapping coughs cleanly. Ambiguous → flag, don't cut.
   Present a grouped list: `[mm:ss] removed text/sound — reason`, plus flagged-for-review.
3. Review gate — `python -m helpers.render edit/transcript.json edit/cuts.json edit/preview.mp3`.
   (Tracks are read from transcript.json; for a single-file source pass `--audio <file>`.)
   User reads the list and listens. Apply changes to cuts.json, re-render. Loop until approved.
4. Final render — same command → `edit/final.mp3` (+ `edit/final_kept_transcript.json`).
5. QA — `python -m helpers.qa edit/final.mp3 edit/final_kept_transcript.json edit/qa_report.md`.
   Review flagged seams. Issues → back to step 3.
6. Chapters — read `edit/final_kept_transcript.json`, write `edit/chapters.txt` (`mm:ss Title`).
7. Memory — append a one-line summary to `edit/project.md`.

## Helpers
- transcribe.py — multitrack dir (or single file) → transcript.json (words+speaker+events).
- pack.py — transcript.json → packed.md (zh-TW reading view).
- render.py — transcript.json + cuts.json → preview/final.mp3 + kept_transcript.json (per-track cut → mix).
- qa.py — seam/silence check → qa_report.md.
- ai_listen.py — optional Gemini clip classifier (cough vs laugh). [Phase 5]

## Anti-patterns
- Don't invent timestamps — always cite word ids from transcript.json.
- Don't auto-remove anything that might be a laugh.
- Don't skip the preview/approval gate.
- Don't write into the source folder or the skill folder.
