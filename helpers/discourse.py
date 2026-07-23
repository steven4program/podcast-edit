"""Context-dependent discourse-marker recall — a triage aid for the cut-proposal brain.

Some fillers are NOT always garbage. 呃/嗯/啊/欸 never carry meaning (helpers.fillers cuts
them acoustically), but 对/好/然后/就是/那个… are the SAME token used two ways: a real
answer/affirmation ("你是說首頁嗎?" — "对") vs a discourse tic that just closes one thought
and pivots to the next ("…往哪邊導。对,然后就是…"). You cannot decide from the WORD — only
from context. So this pass does not decide; it surfaces every candidate WITH the objective
context features that separate the two, and buckets each into a suggested verdict the LLM
confirms:

  - keep    — answers another speaker (turn boundary) or a question → real meaning.
  - cut     — same speaker, sentence-initial, followed by a pivot connective (那/然后/就是…),
              plausible duration → a tic. Emitted as a word-id micro cut (render onset-aligns
              the end to the next content word, so the tic's real sound + trailing pause go
              even when Scribe under-measured the token itself).
  - review  — ambiguous (same speaker but mid-clause / demonstrative 那个 / adjective 好…) →
              the LLM reads the context text and decides.
  - flag    — the removed span is implausibly short per char (token times fabricated); cutting
              would leave the sound audible → manual repair, never auto-cut.

Recall aid like helpers.repeats: false positives are fine (they land in review/keep for the
LLM to drop); the thing to avoid is missing a candidate. When in doubt the verdict leans keep —
over-keeping a tic is a blemish, deleting a real answer is an error.
"""
import json

from .render import _protected, verify_no_midword
from .repeats import _collapse_warning, _is_content, _norm

# Curated, conservative. Simplified — Scribe transcribes zh-TW audio to simplified glyphs,
# one char per token (就是 = 就+是), punctuation as its own token. Broad is fine: the feature
# triage sorts grammatical uses into review/keep, so only clear tics reach `cut`.
_MARKERS = ("对不对", "对吧", "对啊", "对", "然后", "就是", "那个", "那", "好",
            "其实", "反正", "所以说", "你知道吗", "你知道", "我觉得", "怎么讲",
            "这样子", "对对对", "是这样")
# A pivot connective right after the marker is the tic tell ("对,那…" / "对,然后…").
_PIVOT = ("那", "然后", "还有", "所以", "就是", "那个", "但是", "可是", "不过", "因为",
          "第一", "第二", "第三", "接下来", "另外", "再来", "首先", "我们", "它", "这")
# The marker answers a question → keep.
_QUESTION_TAIL = ("吗", "呢", "对不对", "是不是", "好不好", "对吧", "?", "？")
_SENT_END = ("。", "！", "？", "…", "—", "?", "!", "")
_SKIP = set("，。、！？…—–-　 .,!?；：\"'「」（）()")

_MAX_DUR = 1.2      # a marker whose voiced span is longer than this is embedded in speech
_TURN_WINDOW = 1.5  # another speaker talking within this many s before → backchannel, keep


def _content_stream(words):
    """[(i, id, norm, speaker)] for content tokens (drops punctuation; keeps 呃嗯 out via
    repeats._is_content). Index i is the position in `words`, for boundary/context lookups."""
    return [(i, w["id"], _norm(w["text"]), w["speaker"])
            for i, w in enumerate(words) if _is_content(w["text"])]


def _match(stream, s):
    """Longest marker matching the content run starting at stream position s, same speaker.
    Greedily concatenates same-speaker tokens until they equal (hit) or diverge from a marker.
    -> (marker_text, n_tokens) or None. _MARKERS is ordered longest-first per family."""
    spk = stream[s][3]
    for m in _MARKERS:
        acc, span = "", 0
        while s + span < len(stream) and len(acc) < len(m) and stream[s + span][3] == spk:
            acc += stream[s + span][2]
            span += 1
            if acc == m:
                return m, span
            if not m.startswith(acc):
                break
    return None


def _turn_response(words, i0, spk):
    """True if this marker answers ANOTHER speaker: the nearest preceding content token is a
    different speaker, or a different speaker was talking within _TURN_WINDOW before it."""
    start = words[i0].get("start")
    for w in reversed(words[:i0]):
        if w["text"].strip() in _SKIP:
            continue
        if w["speaker"] != spk:
            return True
        # same-speaker content reached; still check for cross-talk just before
        break
    if start is not None:
        for w in reversed(words[:i0]):
            if w.get("end") is None or w["end"] < start - _TURN_WINDOW:
                break
            if w["speaker"] != spk and _is_content(w["text"]):
                return True
    return False


def _post_question(words, i0, spk):
    """True if a question is being answered: scan back to the previous sentence end and look
    for ？ or a question tail (吗/呢/对不对…) from anyone in that window."""
    for w in reversed(words[:i0]):
        t = w["text"].strip()
        if t in ("。", "！", ".", "!"):
            break
        if t in _QUESTION_TAIL or any(q in t for q in ("对不对", "是不是", "好不好")):
            return True
    return False


def _sentence_initial(words, i0, turn):
    if turn or i0 == 0:
        return True
    prev = words[i0 - 1]["text"].strip()
    if prev in _SENT_END:
        return True
    # allow one comma/punct between the sentence end and the marker ("。 对" vs "。，对")
    if prev in _SKIP and i0 >= 2 and words[i0 - 2]["text"].strip() in _SENT_END:
        return True
    return False


def _next_content(words, after_i):
    for j in range(after_i, len(words)):
        if _is_content(words[j]["text"]):
            return j
    return None


def _followed_by_pivot(words, after_i, spk):
    """Concatenate the next few same-speaker content tokens; does it open with a pivot?"""
    acc, j, n = "", after_i, 0
    while j < len(words) and n < 4:
        w = words[j]
        if w["text"].strip() in _SKIP:
            j += 1
            continue
        if w["speaker"] != spk:
            break
        acc += _norm(w["text"])
        n += 1
        j += 1
    return any(acc.startswith(p) for p in _PIVOT)


def _context(words, i0, i1, pad=3):
    return "".join(w["text"] for w in words[max(0, i0 - pad):i1 + 1 + pad])


def propose_discourse(transcript):
    """-> list of candidate dicts (verdict + features + word-id span). Does NOT write cuts;
    the LLM confirms each `cut`, spot-checks `review`, and repairs `flag` manually."""
    words = transcript["words"]
    by_id = {w["id"]: w for w in words}
    stream = _content_stream(words)
    out, s = [], 0
    while s < len(stream):
        m = _match(stream, s)
        if not m:
            s += 1
            continue
        text, span = m
        i0 = stream[s][0]
        i1 = stream[s + span - 1][0]
        spk = words[i0]["speaker"]
        first_id, last_id = words[i0]["id"], words[i1]["id"]

        turn = _turn_response(words, i0, spk)
        question = _post_question(words, i0, spk)
        nxt_i = _next_content(words, i1 + 1)
        sent_init = _sentence_initial(words, i0, turn)
        pivot = _followed_by_pivot(words, i1 + 1, spk) if nxt_i is not None else False
        dur = sum(max(0.0, words[k]["end"] - words[k]["start"]) for k in range(i0, i1 + 1))

        # Fold trailing punctuation into the removed span (end just before the next content
        # word); render onset-aligns the end there, taking the tic's sound + comma + pause.
        end_word = (by_id_prev(words, nxt_i) if nxt_i is not None else last_id)
        keep_id = words[nxt_i]["id"] if nxt_i is not None else last_id + 1
        warn = _collapse_warning(by_id, first_id, end_word, keep_id, [text])

        if turn or question:
            verdict = "keep"
            why = "answers another speaker" if turn else "answers a question"
        elif dur > _MAX_DUR:
            verdict = "review"; why = f"long span {dur:.2f}s — maybe embedded in speech"
        elif sent_init and pivot and not warn:
            verdict = "cut"; why = "sentence-initial tic + pivot connective follows"
        elif sent_init and pivot and warn:
            verdict = "flag"; why = "tic but token times fabricated"
        else:
            verdict = "review"
            why = ("mid-clause — grammatical role?" if not sent_init
                   else "no pivot after — real content?")

        out.append({"verdict": verdict, "text": text, "speaker": spk,
                    "start_word": first_id, "end_word": end_word,
                    "dur_ms": round(dur * 1000), "turn_response": turn,
                    "post_question": question, "sentence_initial": sent_init,
                    "followed_by_pivot": pivot, "context": _context(words, i0, i1),
                    "reason": f"discourse 「{text}」 — {why}{warn}"})
        s += span
    return out


def by_id_prev(words, nxt_i):
    """The word id immediately before the next content word (the last of the removed span)."""
    return words[nxt_i - 1]["id"]


# ── acoustic localization (the cut geometry) ─────────────────────────────────
# Scribe places 对/好/… tokens at the TAIL of the sound (measured: a 对 whose voiced onset
# is at 181.65s gets a token starting 181.75s — 100ms late, dur 20ms). Cutting on the token
# span leaves the marker's ONSET audible (same failure fillers.py exists to avoid). So a
# confirmed cut is located acoustically on the speaker's own track: back off the token start
# to the true voiced onset, cut [onset, next content onset] as a snap:false macro.
_MAX_ACOUSTIC = 1.6   # onset->next-onset longer than this => embedded in speech => flag


def _voiced_onset(rms, step, token_start, floor, thresh, min_gap=0.03, pad=0.02):
    """True onset of the voiced run containing token_start: walk back over voiced frames, stop
    at a silence gap >= min_gap, never past `floor` (the previous content word's end). Recovers
    the leaked onset of a late-placed token. -> seconds (padded a touch earlier), >= floor."""
    n = len(rms)
    if n == 0:
        return max(floor, token_start)
    i = min(int(token_start / step), n - 1)
    lo = max(0, int(floor / step))
    onset, gap, j = i, 0, i
    while j >= lo:
        if rms[j] > thresh:
            onset, gap = j, 0
        else:
            gap += 1
            if gap * step >= min_gap:
                break
        j -= 1
    return max(floor, onset * step - pad)


def acoustic_cuts(transcript, track_paths, candidates=None):
    """Convert confirmed `cut` candidates into acoustic macro cuts (snap:false), locating each
    marker's true voiced onset on the speaker's own track. -> (cuts, flagged). A span that runs
    together with neighbouring speech (onset lands mid-word, or the extent is too long) is
    flagged for manual repair, never force-cut. Mirrors fillers.propose_filler_cuts."""
    from .fillers import _envelope, _voiced_thresh
    from .transcribe import speaker_from_filename
    words = transcript["words"]
    order = {w["id"]: i for i, w in enumerate(words)}
    envs = {speaker_from_filename(p): _envelope(p) for p in track_paths}
    thrs = {spk: _voiced_thresh(e[0]) for spk, e in envs.items()}
    if candidates is None:
        candidates = [c for c in propose_discourse(transcript) if c["verdict"] == "cut"]
    cuts, flagged = [], []
    for c in candidates:
        spk = c["speaker"]
        i0, i1 = order[c["start_word"]], order[c["end_word"]]
        prev = next((w for w in reversed(words[:i0]) if _protected(w) and w["speaker"] == spk), None)
        nxt = next((w for w in words[i1 + 1:] if _protected(w)), None)
        if spk not in envs or prev is None or nxt is None:
            flagged.append({**c, "reason": c["reason"] + " — no track/boundary; manual"}); continue
        rms, step = envs[spk]
        onset = _voiced_onset(rms, step, words[i0]["start"], prev["end"], thrs[spk])
        end = nxt["start"]
        if end - onset > _MAX_ACOUSTIC or verify_no_midword([onset, end], words):
            flagged.append({**c, "reason": c["reason"] + " — boundary mid-word / embedded; manual"})
            continue
        cuts.append({"type": "macro", "start": onset, "end": end, "snap": False,
                     "reason": c["reason"]})
    return cuts, flagged


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Context-dependent discourse-marker recall.")
    p.add_argument("transcript", help="transcript.json (its `tracks` locate the audio)")
    p.add_argument("--acoustic", action="store_true",
                   help="emit ready-to-merge cuts for the `cut` bucket via acoustic_cuts: each "
                        "marker's TRUE voiced onset on the speaker's own track, cut as a "
                        "snap:false macro. Use THIS, not the word-id span — Scribe places 对/好 "
                        "tokens at the tail of the sound, so a word-id cut leaks the onset. "
                        "Review and drop any the LLM judges to keep before merging.")
    a = p.parse_args()
    t = json.load(open(a.transcript, encoding="utf-8"))
    if a.acoustic:
        cuts, flagged = acoustic_cuts(t, t.get("tracks") or [t["audio"]])
        print(json.dumps({"cuts": cuts, "flagged": flagged}, ensure_ascii=False, indent=2))
    else:
        cands = propose_discourse(t)
        buckets = {}
        for c in cands:
            buckets.setdefault(c["verdict"], []).append(c)
        summary = {k: len(v) for k, v in buckets.items()}
        print(json.dumps({"summary": summary, "discourse": cands}, ensure_ascii=False, indent=2))
