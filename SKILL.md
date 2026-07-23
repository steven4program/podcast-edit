---
name: podcast-edit
description: Edit a zh-TW audio podcast — remove stutters/repeats, host coughs/throat-clears, render a smooth track. Use when the user wants to cut/edit/clean a podcast audio file.
---

# Podcast Edit

You edit zh-TW podcast audio. You make the editorial judgments by reading text;
thin helpers do transcription, cutting, and rendering. Work in the folder containing the audio.

Input is **multitrack**: one mono track per speaker (e.g. `…--ted35.wav` + `…--guestNNN--<name>.wav`),
all the same length/session. Speaker labels come from each filename's last `--` segment, so
they are 100% accurate — no diarization guessing.

## Primary goals
1. **Remove throat-clearing/nose-clearing/coughing from the cough-prone host — the headline
   value.** This is where a human editor spends most of their time, so removing it is the
   whole point of the skill. **Recall-first**: hunt these aggressively (the host clears almost
   continuously). Mute the ones that land in a gap in his own speech; flag the ones
   co-articulated with his own words (cut/mute can't remove those without losing the words —
   spectral denoise territory, out of scope). NEVER delete laughter. [Phase 5]
2. Remove stutters and repeated words/sentences (delete-earlier, keep-later).
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
7. Loudness is render's job: per-track speech leveling (a static per-track speech gain +
   downward-only compand compression — no time-varying gain, which fades phrase tails) evens
   out speaker gaps and sudden loud/quiet, then mastering anchors the mix at -16 LUFS with a
   measured static gain + a millisecond transient limiter (never dynamic loudness
   normalization). Render from the original tracks, never a 16k copy.
8. Source tracks untouched. All artifacts under `<audio_dir>/edit/`.

## Workflow
Artifacts live under `<audio_dir>/edit/`, in three tiers (helpers create the subdirs on write):
**top** = state & reports you keep or act on (`transcript.json`, `cuts.json`, `project.md`,
`flagged.md`); **`edit/work/`** = regeneratable intermediates (`*_raw.json`, `packed.md`,
`preview.mp3`); **`edit/out/`** = deliverables (`final.mp3`, `chapters.txt`). Helpers `makedirs`
their own output dir, but `flagged.md` is written by you in step 2 — top always exists by then.

0. Setup — confirm `ffmpeg` + `ELEVENLABS_API_KEY`. Ask the user the cough-prone host name.
1. Transcribe — `python -m helpers.transcribe <tracks_dir> edit/transcript.json` (per-track
   Scribe v2 with a VAD silence pre-pass + merge). On the first episode, confirm a
   `edit/work/transcript_<speaker>_raw.json` is per-word zh-TW with matching event tags.
   1a. (Optional, recommended) Re-time — `python -m helpers.align edit/transcript.json --write`
   (needs `pip install -e ".[align]"`; ~30 min/episode on CPU, downloads a ~1.2GB model once).
   Scribe's word TIMES collapse on fast / run-together speech (a filler then swallows the next
   word — e.g. 矽谷 — and cut boundaries land mid-sound); forced alignment re-times the KNOWN
   text against the audio to fix them at the source. It only changes word start/end (text /
   events untouched), keeps Scribe times where alignment is off-voice / out-of-order, and
   writes a review list. **Run it HERE, before pack and every recall pass** — everything
   downstream (esp. `ai_listen`'s cough mutes, which must sit in the speaker's real gaps) then
   builds on the corrected times. Skipping it just leaves Scribe's times (the old behaviour).
   Then `python -m helpers.pack edit/transcript.json > edit/work/packed.md`.
2. Propose cuts — read `edit/work/packed.md` and `references/edit-heuristics.md`. Write
   `edit/cuts.json` as `{"cuts": [...], "mutes": [...]}`:
   - Repeats/stutters/false-starts (recall-first): run `python -m helpers.repeats edit/transcript.json`.
     It returns `repeats` (adjacent duplicates incl. `-` dash-stutters, delete-earlier) and
     `false_starts` (`——` abandoned attempts). Don't eyeball packed.md for these — this is the
     recall pass. REVIEW each and drop false positives: reduplicated words/names (`剛剛`, `萬萬`,
     `汪汪`), emphasis/rhetoric (`非常非常多`, `懂A懂B懂B懂A`). For a `false_start` flagged ⚠,
     extend the span if the abandoned fragment is longer than the 1-token guess (`我覺得——`, not
     `得——`). A candidate marked ⚠ 塌縮 goes to flagged.md, never cuts.json — its token times
     are fabricated (stutter burst) and the cut would leave the sound audible while the
     transcript shows it removed. Keep the confirmed ones. Add macro segment cuts (off-topic,
     retakes) from packed.md.
     Pre-show chatter is cut BY DEFAULT: macro-cut word 0 up to the host's show-opening
     greeting (類似「嗨，大家好，歡迎來到今天節目」, wording varies — see edit-heuristics).
   - Fillers: run `python -m helpers.fillers edit/transcript.json` (Scribe filler
     timestamps are unreliable; it finds the sound acoustically on the speaker's own
     track and flags the unsafe ones — embedded in speech, cross-talk, or overlapping
     laughter). Handles ONLY 呃嗯啊欸 (always garbage).
   - Context-dependent markers (对/好/然后/就是/那个/其实/我觉得…, recall-first): run
     `python -m helpers.discourse edit/transcript.json`. These are the SAME token used two
     ways (a real answer vs a discourse tic) — NEVER blanket-delete. It buckets each by
     objective context features: confirm the `cut` bucket (same-speaker tics before a pivot
     connective), read the `review` bucket and default to KEEP, and never cut `keep` (answers
     another speaker or a question). Take the confirmed cuts' geometry from `--acoustic`
     (`python -m helpers.discourse edit/transcript.json --acoustic`) — snap:false macros on
     each marker's true voiced onset; a word-id span leaks the onset (Scribe tokens sit at the
     sound's tail). See edit-heuristics "Context-dependent markers".
   - Dead air: run `python -m helpers.deadair edit/transcript.json`. It proposes shortening
     every nobody-talking gap ≥1.2s down to 0.8s (word-union across speakers; gaps covered by
     a laughter event are never proposed; boundaries are acoustically slid off any real
     sound the tokens don't show). REVIEW each: keep a pause that is doing dramatic
     work; also remember laughter/applause isn't words — trust the event protection.
   - Cough/throat-clear (THE headline goal, recall-first): run
     `python -m helpers.ai_listen <host_track> edit/transcript.json <host>`.
     **Current mechanism: Scribe-only** (`--provider scribe`, now the CLI default) — it
     reuses the Scribe audio-event tags already in transcript.json (no extra API call, no
     key). TELL the user its recall is much lower than the audio-LLM sweep (a test episode:
     Scribe tagged 13 clears where the sweep found 38, barely overlapping) — soft clears
     WILL be missed.
     **Audio-LLM sweep (higher recall, currently DISABLED):** Gemini and OpenAI backends
     are both commented out in `ai_providers.py` (user request 2026-07-11). To re-enable:
     uncomment a provider there, restore its dep in pyproject `[ai]`, set the key in .env,
     then sweep with `--provider gemini` (`--throttle 0` on paid tier; free tier: keep
     `--throttle 7` and/or `--max-requests N` to cap cost).
     Both paths share the same verify/split pipeline below. The sweep drops silent-window hallucinations
     (`verify_on_track` — an event must have real energy on the host's own track, so another
     speaker talking on their track never counts), and splits hits into:
       - `mutes` — the clear sits in a **gap** in his own speech → add to `cuts.json` `mutes`;
         render mutes his track for that span (no time removed, other speakers untouched).
       - `flagged` — **co-articulated** with his own words → write to `edit/flagged.md`
         (`mm:ss + type`) for manual repair. Cut/mute can't remove these without taking the
         words too (spectral denoise is out of scope — see the roadmap in the design spec).
     LAUGHTER IS NEVER A CANDIDATE.
   Present a grouped list: `[mm:ss] removed text/sound — reason`, plus the `flagged.md` items.
3. Review gate — `python -m helpers.render edit/transcript.json edit/cuts.json edit/work/preview.mp3`.
   (Tracks are read from transcript.json; for a single-file source pass `--audio <file>`.)
   User reads the list and listens. Apply changes to cuts.json, re-render. Loop until approved.
4. Final render — `python -m helpers.render edit/transcript.json edit/cuts.json edit/out/final.mp3`
   (+ `edit/out/final_kept_transcript.json`). Seams are clean by construction (render's mid-word
   guard + 3ms fade per join), so there is no signal-level QA step — an RMS-ratio seam check
   only re-flags normal pause→speech.
5. Chapters — read `edit/out/final_kept_transcript.json`, write `edit/out/chapters.txt` (`mm:ss Title`).
6. Memory — append a one-line summary to `edit/project.md`.

## Helpers
- transcribe.py — multitrack dir (or single file) → transcript.json (words+speaker+events).
- pack.py — transcript.json → packed.md (zh-TW reading view).
- repeats.py — adjacent-duplicate + `——` false-start finder (recall aid); LLM reviews candidates.
- deadair.py — long nobody-talking gaps → shorten-to-0.8s macro-cut proposals (laughter-safe,
  boundaries acoustically verified per track).
- render.py — transcript.json + cuts.json → preview/final.mp3 + kept_transcript.json (per-track cut → mix).
  `--stems` treats OUT as a directory and exports in the source tracks' format (wav in →
  wav out, mp3 in → mp3 out): `final.<ext>` (integrated mix) + one `final_<speaker>.<ext>`
  stem per source track, all cut at the same boundaries (stems keep per-track leveling but
  skip mastering, preserving speaker balance for re-mixing).
- ai_listen.py — throat/nose-clear discovery on the host track: `scribe_events` (keyless,
  reuses Scribe's event tags — low recall; the current default, since the audio-LLM
  backends are commented out in ai_providers.py) or `sweep_track` (audio-LLM, disabled) →
  `verify_on_track` (drop hallucinations) → `split_events` (mute gaps / flag co-articulated);
  `classify` confirms a single clip (needs an enabled provider). [Phase 5]
- review.py — transcript.json + cuts.json → `edit/work/review.html`: original vs edited
  players + the transcript annotated with every cut/mute and its reason (offer it to the
  user at the review gate). With `--serve [PORT]` it also serves the page and enables the
  完成 button, which exports the mix + per-speaker stems to `edit/out/` in the source
  tracks' format (opened as a plain file the button explains it needs --serve).

## Anti-patterns
- Don't invent timestamps — always cite word ids from transcript.json.
- Don't auto-remove anything that might be a laugh.
- Don't skip the preview/approval gate.
- Don't write into the source folder or the skill folder.
