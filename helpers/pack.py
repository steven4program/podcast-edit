"""Pack a canonical transcript into a token-efficient zh-TW reading view.

Scribe transcribes Mandarin to Simplified; we convert the reading view to
Traditional (Taiwan) here. transcript.json stays the faithful Scribe record.
"""
from opencc import OpenCC

_PUNCT = {",": "，", ".": "。", "!": "！", "?": "？", ":": "：", ";": "；"}
_CC = OpenCC("s2twp")  # Simplified -> Traditional (Taiwan, with idioms)


def _mmss(seconds):
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def _norm(text):
    return _CC.convert("".join(_PUNCT.get(c, c) for c in text))


def pack(transcript, gap=0.5):
    words = transcript["words"]
    if not words:
        return ""
    phrases, cur = [], [words[0]]
    for prev, w in zip(words, words[1:]):
        if w["start"] - prev["end"] > gap or w["speaker"] != prev["speaker"]:
            phrases.append(cur)
            cur = [w]
        else:
            cur.append(w)
    phrases.append(cur)
    lines = []
    for ph in phrases:
        first, last = ph[0], ph[-1]
        text = _norm("".join(w["text"] for w in ph))
        lines.append(f"[{_mmss(first['start'])}–{_mmss(last['end'])} "
                     f"#{first['id']}-{last['id']} {first['speaker']}] {text}")
    return "\n".join(lines)


if __name__ == "__main__":
    import json, sys
    print(pack(json.load(open(sys.argv[1], encoding="utf-8"))))
