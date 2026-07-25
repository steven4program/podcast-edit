"""Merge the detection-pass candidate lists into a render-ready cuts.json + flagged.md.

This closes the one gap in the step-2 workflow that used to invite improvisation: an episode
yields ~800 candidates across five JSON files (repeats, fillers, discourse, deadair, cough),
and hand-typing them into cuts.json is impractical — so the agent reached for a throwaway
script. This helper is the sanctioned mechanical merge instead.

It does NOT make editorial judgments (that stays the LLM's job, per the whole architecture).
It only:
  - concatenates the categories that need no per-candidate judgment (fillers, deadair,
    discourse `cut`-bucket acoustic geometry, cough mutes, precise false_starts);
  - applies the LLM's per-episode DECISIONS, passed as inputs:
      * `intro_word` — the id of the first kept word (a macro cut [0, its start] removes the
        pre-show / botched-retake chatter; subsumes any candidate inside it);
      * `drop` — repeat phrase units the LLM judged false positives (剛剛/彎彎/一步一步/非常…);
      * `exclude_ids` — specific candidate start_words the LLM vetoed one by one;
  - routes the candidates that must NOT be auto-cut to flagged.md (⚠ 塌縮 stutter-burst repeats
    whose token times are fabricated, and co-articulated coughs — DAW/spectral repair);
  - drops any word-id micro cut whose boundary lands mid-word on the merged multitrack timeline
    (cross-talk makes a clean word-edge for one speaker land inside another's word — render
    would reject the whole render; we drop just the offender so the output is safe by
    construction). This is the same check render runs, done here so cuts.json never fails render.

`compile_cuts` is the pure, testable core (transcript + candidates dict -> spec). The CLI reads
the five work/*.json the detection passes wrote and writes cuts.json + flagged.md.
"""
import json, os

from .render import resolve_cut_times, compute_kept_segments, verify_no_midword


def _mmss(seconds):
    s = int(seconds); return f"{s // 60:02d}:{s % 60:02d}"


def _phrase(reason):
    import re
    m = re.search(r"「(.+?)」", reason or ""); return m.group(1) if m else ""


def _drop_midword_microcuts(spec, words):
    """Drop word-id micro cuts whose resolved boundary lands mid-word on the merged timeline.
    Returns (clean_cuts, dropped) — dropped carries a reason for flagged.md. Word-id cuts skip
    render's snap, so their resolved boundary IS the final one; macros with snap:false likewise.
    Verifying here mirrors render's guard so the emitted cuts.json renders without a ValueError."""
    resolved = resolve_cut_times(spec["cuts"], words)
    segs = compute_kept_segments(resolved, max(w["end"] for w in words))
    boundaries = set(verify_no_midword([b for s in segs for b in s], words))
    if not boundaries:
        return spec["cuts"], []
    keep, dropped = [], []
    for c, r in zip(spec["cuts"], resolved):
        if r["start"] in boundaries or r["end"] in boundaries:
            dropped.append(c)
        else:
            keep.append(c)
    return keep, dropped


def compile_cuts(transcript, candidates, intro_word=None, drop=(), exclude_ids=()):
    """-> (spec, flagged_lines, stats). `candidates` maps category -> that pass's parsed JSON:
      repeats:   {"repeats": [...], "false_starts": [...]}
      fillers:   {"cuts": [...], ...}
      discourse: {"cuts": [...], ...}   (the --acoustic output; `cut`-bucket geometry)
      deadair:   {"dead_air": [...]}
      cough:     {"mutes": [...], "flagged": [...]}
    Any category may be absent."""
    words = transcript["words"]
    by_id = {w["id"]: w for w in words}
    drop, exclude = set(drop), set(exclude_ids)
    cuts, mutes, flagged, stats = [], [], [], {}

    intro_end = by_id[intro_word]["start"] if intro_word is not None else 0.0
    if intro_word is not None:
        cuts.append({"type": "macro", "start": 0.0, "end": round(intro_end, 3),
                     "reason": "pre-show / retake — cut to the opening greeting"})

    def _end_t(c):
        return by_id[c["end_word"]]["end"] if c.get("end_word") is not None else c["end"]

    def _start_t(c):
        return by_id[c["start_word"]]["start"] if c.get("start_word") is not None else c["start"]

    def _in_intro(c):
        return _end_t(c) <= intro_end + 0.01

    # repeats + false_starts — judgment categories: 塌縮 -> flagged; drop LLM's reduplications
    # (by phrase) and vetoes (by start_word); everything else is a confirmed cut.
    rp = candidates.get("repeats", {})
    for kind in ("repeats", "false_starts"):
        stats[kind] = 0
        for c in rp.get(kind, []):
            if "塌縮" in c.get("reason", ""):
                flagged.append(f"{_mmss(_start_t(c))}  {kind} 塌縮 「{_phrase(c['reason'])}」 "
                               f"— token times fabricated; DAW 手修")
            elif _phrase(c["reason"]) in drop or c.get("start_word") in exclude or _in_intro(c):
                continue
            else:
                cuts.append(c); stats[kind] += 1

    # acoustic / deterministic categories — take all outside the intro cut.
    for kind, key in (("fillers", "cuts"), ("discourse", "cuts"), ("deadair", "dead_air")):
        stats[kind] = 0
        for c in candidates.get(kind, {}).get(key, []):
            if c["end"] <= intro_end + 0.01:
                continue
            cuts.append(c); stats[kind] += 1

    # cough — mutes in his own gaps; co-articulated ones flagged for DAW/spectral repair.
    co = candidates.get("cough", {})
    for m in co.get("mutes", []):
        if m["end"] > intro_end + 0.01:
            mutes.append(m)
    for ev in co.get("flagged", []):
        flagged.append(f"{_mmss(ev['start'])}  {ev.get('type', 'throat_clear')} "
                       f"(co-articulated 與自己說話重疊) — DAW spectral 手修")

    spec = {"cuts": cuts, "mutes": mutes}
    kept, midword = _drop_midword_microcuts(spec, words)
    for c in midword:
        flagged.append(f"{_mmss(_start_t(c))}  {c.get('type', 'cut')} boundary mid-word on the "
                       f"multitrack timeline (cross-talk) 「{_phrase(c.get('reason', ''))}」 — dropped")
    spec["cuts"] = kept
    stats["midword_dropped"] = len(midword)
    return spec, sorted(flagged), stats


_WORK_FILES = {"repeats": "repeats.json", "fillers": "fillers.json",
               "discourse": "discourse_acoustic.json", "deadair": "deadair.json",
               "cough": "cough.json"}


def _load_candidates(work_dir):
    out = {}
    for cat, fname in _WORK_FILES.items():
        path = os.path.join(work_dir, fname)
        if os.path.isfile(path):
            out[cat] = json.load(open(path, encoding="utf-8"))
    return out


def write_flagged(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Flagged — 需人工 DAW 修復（無法安全自動切除）\n\n"
                "清喉聲與自己說話 co-articulated（切/靜音會連字一起吃）、"
                "口吃 burst 時間戳塌縮、以及多軌 cross-talk 落在字中間者。\n\n")
        f.write("\n".join(lines) + ("\n" if lines else ""))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Merge detection-pass candidates (work/*.json) into cuts.json + flagged.md. "
                    "Editorial decisions are inputs, not baked in — see --intro-word/--drop/--exclude.")
    p.add_argument("transcript", help="edit/transcript.json (its dir holds work/ and gets cuts.json)")
    p.add_argument("--intro-word", type=int, default=None,
                   help="id of the first KEPT word; emits a macro cut [0, its start] (pre-show/retake)")
    p.add_argument("--drop", nargs="*", default=[], metavar="PHRASE",
                   help="repeat phrase units the LLM judged false positives (剛剛 慢慢 彎彎 一步 非常…)")
    p.add_argument("--exclude", nargs="*", type=int, default=[], metavar="START_WORD",
                   help="specific candidate start_word ids the LLM vetoed")
    a = p.parse_args()
    base = os.path.dirname(os.path.abspath(a.transcript))
    t = json.load(open(a.transcript, encoding="utf-8"))
    cand = _load_candidates(os.path.join(base, "work"))
    spec, flagged, stats = compile_cuts(t, cand, a.intro_word, a.drop, a.exclude)
    json.dump(spec, open(os.path.join(base, "cuts.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    write_flagged(os.path.join(base, "flagged.md"), flagged)
    print(json.dumps({"cuts": len(spec["cuts"]), "mutes": len(spec["mutes"]),
                      "flagged": len(flagged), "by_category": stats}, ensure_ascii=False, indent=2))
