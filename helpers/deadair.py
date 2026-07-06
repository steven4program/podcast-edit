"""Dead-air detection: word-union gaps where NOBODY talks -> macro-cut proposals.

A recall aid like repeats.py — deterministic scan, the LLM reviews each proposal
(drop a pause that is doing dramatic work). A gap that overlaps a laughter event is
never proposed: laughter is content (hard rule). Each qualifying gap is shortened to
`keep` seconds — half left after the last word, half before the next — so the cut sits
in the middle of the silence and speech rhythm stays natural. Proposals are macro cuts
(start/end seconds); render snap + the mid-word guard keep the seams safe.
"""
import json


def find_dead_air(words, events=None, min_gap=1.2, keep=0.8):
    """-> macro cut dicts for every interior gap >= min_gap in the WORD-UNION timeline
    (a moment nobody is talking, across all speakers)."""
    laughs = [(e["start"], e["end"]) for e in (events or []) if e.get("type") == "laughter"]
    out, reach = [], None
    for w in sorted(words, key=lambda w: w["start"]):
        if reach is not None:
            gap = w["start"] - reach
            if gap >= min_gap:
                s, e = reach + keep / 2, w["start"] - keep / 2
                if any(ls < e and s < le for ls, le in laughs):
                    pass  # laughter lives in this pause -> keep it whole
                else:
                    out.append({"type": "macro", "start": round(s, 3), "end": round(e, 3),
                                "reason": f"dead air {gap:.1f}s → keep {keep:.1f}s"})
        reach = w["end"] if reach is None else max(reach, w["end"])
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Propose cuts that shorten long dead air.")
    p.add_argument("transcript")
    p.add_argument("--min-gap", type=float, default=1.2, help="propose only gaps >= this (s)")
    p.add_argument("--keep", type=float, default=0.8, help="pause length left in place (s)")
    a = p.parse_args()
    t = json.load(open(a.transcript, encoding="utf-8"))
    cuts = find_dead_air(t["words"], t.get("events"), a.min_gap, a.keep)
    print(json.dumps({"dead_air": cuts}, ensure_ascii=False, indent=2))
