# Edit heuristics (zh-TW)

The LLM consults this when proposing cuts. Knowledge, not machinery.

## Fillers (remove)
嗯、呃、啊、欸、那個（贅詞）、然後（贅詞）、就是（贅詞）、對對對（附和填充）。
Keep a filler when it carries meaning ("那個" as a real demonstrative, "就是" introducing a definition).

**Filler timestamps are unreliable.** Scribe gives 呃/嗯 near-zero-width tokens, often parked
in the wrong silence (verified: a 呃 whose sound is at 47.2s got a token at 49.0s). So do NOT
cut on the token span — use `helpers.fillers.propose_filler_cuts`, which locates the actual
filler SOUND acoustically in the gap between the surrounding content words, on that speaker's
own track. Fillers embedded in continuous speech/stutter or in cross-talk are **flagged, not
cut** (review manually; cross-talk ones are handled in Phase 5 by muting that speaker's track).

## Repeat-collapse (PRIORITY #1) — delete-earlier, keep-later
Adjacent duplicated word/phrase (exact or near-exact) → keep ONE, the LATER instance.
- `我覺得 我覺得 這個` → delete the first `我覺得`.
- `提到說 就是 就是 這個東西` → delete the first `就是`.
- Self-correction is the same family: `我想說…我要說的是` → keep the corrected later clause.
- PROTECT emphasis/rhetoric: `真的真的很棒`, deliberate doubling → keep. If unsure, keep + flag.

## Stutters / false starts
Partial-word repeats and abandoned starts → remove the broken attempt, keep the clean one.
Scribe v2 marks these with `-` (e.g. `它最-最主要`, `K-Kinesis`) — strong stutter signal.

## Macro deletions (segment-level)
Pre-show prep, off-topic chit-chat, tech debugging ("聽得到嗎"), repeated takes, privacy.
Prefer boundaries at sentence/breath edges.

## Non-verbal (cough/throat-clear) — THE headline goal, recall-first
This is where a human editor spends most of their time; removing it is the whole point of the
skill. Hunt aggressively — the cough-prone host clears his throat/nose almost continuously.

- **Recall, not just Scribe events.** Scribe's event tagger has poor recall on soft nasal/throat
  sounds (caught 3 of ~12 in a test clip) and RMS energy can't tell a throat-clear from a
  voiced syllable — so don't rely on `transcript.json` events alone. The recall mechanism is a
  **Gemini sweep over the host's own track**: slice it into ~12s windows and ask Gemini to
  timestamp every throat/nose-clear. Scribe events are a starting set, not the candidate list.
- **Candidate = throat-clear/cough on the cough-prone host's track.** Flag other speakers' ones.
- **LAUGHTER IS NEVER A CANDIDATE** (Scribe OR Gemini saying "laughter" → keep). This is the one
  place precision beats recall: a wrongly-deleted laugh is unforgivable; a missed clear is not.
- Confirm a candidate with `helpers.ai_listen.classify`; proceed only on cough/throat_clear.
- **Removal is bounded by overlap with his OWN speech:**
  - In a *gap* in his own words → **mute** his track for that span (render `mutes=`): vanishes
    with no time removed, other speakers untouched (their overlap is fine — only his track mutes).
  - *Co-articulated* (firing while he speaks his own words) → **flag, do NOT cut/mute** — muting
    would take his words too. Cleaning these needs spectral denoise (out of scope v1).

## Dead air
Ignore ≤0.5s. Consider >1s. Long silences also surface in QA.

## Multitrack reality (this project)
Input is one mono track per speaker; transcript.json is the merge of all tracks on one
timeline, with `tracks: [paths]` and 100%-accurate speaker labels (from filenames).
- **Speakers overlap.** A cut's start/end must fall where NO speaker is mid-word — a
  *word-union gap* (a moment everyone is between words). Cutting at one speaker's word edge
  can land mid-word for another speaker who is talking over them, and `render` will reject it
  (`verify_no_midword`). Pick cut boundaries at clean gaps; if render raises a mid-word
  ValueError, move the boundary out to the nearest gap and re-render.
- Silent passages of a track are dropped before transcription, so absent speakers simply
  contribute no words for that span (not silence, not hallucinated text).

## Output format
Write `edit/cuts.json`: `{"cuts":[{type, start_word, end_word, reason}, ...]}`.
- `type`: `micro` (word/phrase), `macro` (segment), or `event` (cough; uses start/end).
- Word cuts cite `start_word`/`end_word` ids from `packed.md`. Event cuts use `start`/`end`.
- Every cut MUST cite ids from the transcript (or event start/end). Never invent times.
