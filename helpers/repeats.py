"""Deterministic adjacent-repeat finder — a recall aid for the cut-proposal brain.

Repeats/stutters were the one edit category with NO recall mechanism: fillers have
acoustic detection, coughs have a Gemini sweep, but adjacent duplicates relied entirely
on the LLM eyeballing packed.md (and it missed some). This scans the transcript for
adjacent duplicated tokens/phrases per speaker and emits candidate delete-earlier cuts.
It does NOT judge: the LLM still reviews each one (protect emphasis like 非常非常多 /
真的真的, drop 附和 like 對對對). False positives are fine; misses are the thing to avoid.
"""
import json, re

_STRIP = "，。、！？…—–-　 .,!?；：「」（）()"
_FILLERS = ("呃", "嗯", "啊", "欸")
_DASH = ("——", "—", "–", "ーー")  # abandon / trailing-off markers ('-' is handled as a repeat)


def _norm(text):
    return text.strip(_STRIP)


def _is_content(text):
    s = _norm(text)
    return bool(re.search(r"\w", s)) and s not in _FILLERS


def find_repeats(words, max_phrase=6):
    """Adjacent same-speaker duplicates → cut the EARLIER copy, keep the later.
    Punctuation/fillers between copies are skipped for matching but folded into the
    deleted span (end_word runs up to just before the kept copy). Triples emit one cut
    per extra copy. Returns cut dicts citing word ids (render resolves + onset-aligns)."""
    content = [(w["id"], _norm(w["text"]), w["speaker"]) for w in words if _is_content(w["text"])]
    n = len(content)
    out, i = [], 0
    while i < n:
        hit = 0
        for k in range(min(max_phrase, (n - i) // 2), 0, -1):
            a, b = content[i:i + k], content[i + k:i + 2 * k]
            if all(a[j][1] == b[j][1] for j in range(k)) and len({t[2] for t in a + b}) == 1:
                hit = k
                break
        if hit:
            a = content[i:i + hit]
            keep_id = content[i + hit][0]            # first token of the kept (later) copy
            phrase = "".join(t[1] for t in a)
            out.append({"type": "micro", "start_word": a[0][0], "end_word": keep_id - 1,
                        "reason": f"repeat 「{phrase}」 — delete-earlier, keep #{keep_id} (review: emphasis?)"})
            i += hit                                 # the kept copy may itself repeat again
        else:
            i += 1
    return out


def find_false_starts(words, lookback=4):
    """Abandoned attempts marked by a '——' em-dash → propose removing the abandoned
    fragment + dash, keeping what comes after. When the speaker RESTARTS by repeating a
    token across the dash (它非——它花: 它 repeats), the fragment's start is known, so the
    span is precise. Otherwise the fragment length is ambiguous, so emit a 1-token best
    guess flagged for span review (correct for 斯——/在——/不——, needs extending for 我覺得——).
    Recall aid like find_repeats — the LLM confirms the cut and the span."""
    out = []
    for di, d in enumerate(words):
        if d["text"].strip() not in _DASH:
            continue
        spk = d["speaker"]
        post = next((w for w in words[di + 1:] if _is_content(w["text"])), None)
        if post is None or post["speaker"] != spk:      # dash at a clean end / speaker change
            continue
        pre = [w for w in words[:di] if _is_content(w["text"]) and w["speaker"] == spk]
        if not pre:
            continue
        pn = _norm(post["text"])
        hit = next((w for w in reversed(pre[-lookback:]) if _norm(w["text"]) == pn), None)
        if hit:                                          # restart-repeat → precise span
            frag = "".join(_norm(x["text"]) for x in pre if x["id"] >= hit["id"])
            out.append({"type": "micro", "start_word": hit["id"], "end_word": d["id"],
                        "reason": f"false start 「{frag}——」 restart 「{pn}」 #{post['id']} — keep later"})
        else:                                            # bare abandon → flag span for review
            prev = pre[-1]
            out.append({"type": "micro", "start_word": prev["id"], "end_word": d["id"],
                        "reason": f"false start 「{_norm(prev['text'])}——」? ⚠ extend span if longer — keep 「{pn}」 #{post['id']}"})
    return out


if __name__ == "__main__":
    import sys
    t = json.load(open(sys.argv[1]))
    print(json.dumps({"repeats": find_repeats(t["words"]),
                      "false_starts": find_false_starts(t["words"])}, ensure_ascii=False, indent=2))
