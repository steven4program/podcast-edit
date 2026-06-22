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


if __name__ == "__main__":
    import sys
    t = json.load(open(sys.argv[1]))
    print(json.dumps(find_repeats(t["words"]), ensure_ascii=False, indent=2))
