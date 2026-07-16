"""Build a self-contained review page: original vs edited audio side by side, plus the
full transcript annotated with what was removed where and why.

build_html is the testable core (pure: transcript + cuts spec -> HTML string). The CLI
also mixes the raw tracks into one listenable original (plain amix, no processing) and
writes everything under edit/work/. Open review.html directly in a browser — clicking a
line seeks BOTH players (edited time is remapped through the kept segments).
"""
import argparse, html, json, os, subprocess

from opencc import OpenCC

from .render import resolve_cut_times, compute_kept_segments, _MASTER

_CC = OpenCC("s2twp")  # display in zh-TW, same as pack.py; transcript.json stays as-is

_CATS = [("filler", "filler", "贅詞"), ("repeat", "repeat", "重複"),
         ("false start", "false-start", "重講/口吃")]


def _cat(cut):
    r = cut.get("reason", "")
    for key, cls, label in _CATS:
        if r.startswith(key):
            return cls, label
    if cut.get("type") == "event":
        return "event", "咳嗽/雜音"
    return "macro", "段落刪除"


def _mmss(seconds):
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def _tw(text):
    return html.escape(_CC.convert(text))


def annotate(words, resolved_cuts):
    """[(word, covering-cut-or-None)]: removed iff a resolved cut span contains the
    word's start — the same rule render's remap_words applies."""
    return [(w, next((c for c in resolved_cuts if c["start"] <= w["start"] < c["end"]), None))
            for w in words]


def _rows(transcript, spec, gap=1.0):
    words = transcript["words"]
    resolved = resolve_cut_times(spec["cuts"], words)
    tagged = annotate(words, resolved)
    # Group per SPEAKER, not by merged-timeline order: speakers overlap constantly
    # (backchannels), and merged order interleaves their words character by character.
    # Each speaker's run breaks only at his own >gap pause; rows then sort by start,
    # so overlap reads as two consecutive whole sentences.
    by_spk = {}
    for item in tagged:
        by_spk.setdefault(item[0]["speaker"], []).append(item)
    phrases = []
    for items in by_spk.values():
        cur = [items[0]]
        for (pw, _), item in zip(items, items[1:]):
            if item[0]["start"] - pw["end"] > gap:
                phrases.append(cur)
                cur = [item]
            else:
                cur.append(item)
        phrases.append(cur)
    phrases.sort(key=lambda ph: ph[0][0]["start"])

    rows = []
    for ph in phrases:
        spans, i = [], 0
        while i < len(ph):
            w, c = ph[i]
            if c is None:
                spans.append(_tw(w["text"]))
                i += 1
            else:
                j = i
                while j < len(ph) and ph[j][1] is c:
                    j += 1
                cls, label = _cat(c)
                title = f"{label}：{_tw(c.get('reason', ''))}"
                spans.append(f'<del class="{cls}" title="{title}">'
                             f'{_tw("".join(t[0]["text"] for t in ph[i:j]))}</del>')
                i = j
        f = ph[0][0]
        rows.append((f["start"], f'<div class="ph" data-t="{f["start"]:.2f}">'
                                 f'<span class="ts">[{_mmss(f["start"])} {_tw(f["speaker"])}]</span> '
                                 f'{"".join(spans)}</div>'))
    all_labels = sorted({m["label"] for m in spec.get("mutes", []) if m.get("label")})
    for m in _merge_labeled_mutes(spec):
        dur = m["end"] - m["start"]
        who = f'（{"+".join(m["labels"])}）' if m["labels"] else ""
        # a mute found by only SOME versions is the interesting difference -> "only"
        only = ' only' if m["labels"] and len(m["labels"]) < len(all_labels) else ""
        rows.append((m["start"], f'<div class="ph muterow" data-t="{m["start"]:.2f}" '
                                 f'data-labels="{html.escape(",".join(m["labels"]))}">'
                                 f'<span class="ts">[{_mmss(m["start"])} {_tw(m["speaker"])}]</span> '
                                 f'<span class="mute{only}" title="{m["start"]:.1f}–{m["end"]:.1f}s 只靜音他這一軌，不刪時間">'
                                 f'🔇 清喉嚨/咳嗽 靜音 {dur:.1f}s{_tw(who)}</span></div>'))
    rows.sort(key=lambda r: r[0])
    return [h for _, h in rows], resolved, all_labels


def _merge_labeled_mutes(spec, tol=0.5):
    """spec["mutes"] entries may carry a "label" (which edited version mutes it).
    Overlapping/near spans across versions collapse into one badge listing every label,
    so the page shows agreement ("OpenAI+Scribe") vs one-detector-only mutes."""
    merged = []
    for m in sorted(spec.get("mutes", []), key=lambda m: m["start"]):
        last = merged[-1] if merged else None
        if last and m["speaker"] == last["speaker"] and m["start"] <= last["end"] + tol:
            last["end"] = max(last["end"], m["end"])
            last["labels"] |= {m["label"]} if m.get("label") else set()
        else:
            merged.append({**m, "labels": {m["label"]} if m.get("label") else set()})
    for m in merged:
        m["labels"] = sorted(m["labels"])
    return merged


_PAGE = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>剪輯審聽 — {title}</title>
<style>
  body {{ font-family: system-ui, "Microsoft JhengHei", sans-serif; margin: 0; color: #222; }}
  header {{ position: sticky; top: 0; background: #fff; border-bottom: 1px solid #ddd;
            padding: 10px 16px; z-index: 9; }}
  .player {{ display: flex; align-items: center; gap: 10px; margin: 4px 0; }}
  .player b {{ width: 5em; }}
  audio {{ width: min(560px, 70vw); height: 32px; }}
  .legend {{ font-size: 13px; margin-top: 6px; color: #555; }}
  .legend span {{ margin-right: 12px; }}
  main {{ max-width: 860px; margin: 0 auto; padding: 12px 16px 60vh; }}
  .ph {{ padding: 3px 6px; border-radius: 4px; line-height: 1.9; cursor: pointer; }}
  .ph:hover {{ background: #f2f6ff; }}
  .ph.now {{ background: #e8f0fe; }}
  .ts {{ color: #999; font-size: 12px; margin-right: 4px; }}
  del {{ padding: 0 2px; border-radius: 3px; }}
  del.repeat {{ background: #fde8e8; color: #c0392b; }}
  del.false-start {{ background: #f3e8fd; color: #8e44ad; }}
  del.filler {{ background: #fdf1e3; color: #ca6f1e; }}
  del.macro, del.event {{ background: #eee; color: #666; }}
  .mute {{ background: #e3f0fd; color: #1a6fc4; border-radius: 3px; padding: 0 4px; }}
  .mute.only {{ background: #fff3cd; color: #9a6b00; }}  /* found by only one version */
  .hint {{ color: #888; font-size: 12px; }}
  .vtoggle {{ margin-top: 4px; font-size: 13px; }}
  .vtoggle button {{ margin-right: 6px; padding: 2px 10px; border: 1px solid #bbb;
                     border-radius: 12px; background: #fff; cursor: pointer; }}
  .vtoggle button.on {{ background: #1a6fc4; color: #fff; border-color: #1a6fc4; }}
  .muterow.voff {{ display: none; }}
  .done {{ margin-top: 6px; }}
  .done button {{ padding: 3px 14px; border: 1px solid #2e7d32; border-radius: 12px;
                  background: #2e7d32; color: #fff; cursor: pointer; }}
</style></head>
<body>
<header>
  <div class="player"><b>原始</b><audio id="orig" controls preload="metadata" src="{orig_src}"></audio></div>
{players}
  <div class="legend">
    <span><del class="repeat">重複</del></span>
    <span><del class="false-start">重講/口吃</del></span>
    <span><del class="filler">贅詞</del></span>
    <span><del class="macro">段落刪除</del></span>
    <span><span class="mute">🔇 清喉嚨靜音</span></span>
    <span><span class="mute only">🔇 僅單一版本偵測到</span></span>
    <span class="hint">刪除線 = 已從成品移除（滑鼠移上去看原因）。點任一句：所有播放器一起跳到該處。</span>
  </div>
{toggle}
  <div class="done"><button id="done">完成 — 匯出（整合＋分軌，格式同來源）</button>
    <span id="donest" class="hint"></span></div>
</header>
<main>
{rows}
</main>
<script>
const SEGS = {segs};
const orig = document.getElementById("orig");
const edits = [...document.querySelectorAll("audio.edit")];
function editTime(t) {{  // original-timeline t -> edited-timeline (kept seconds before t)
  let c = 0;
  for (const [s, e] of SEGS) {{
    if (t < s) return c;
    if (t < e) return c + (t - s);
    c += e - s;
  }}
  return c;
}}
const rows = [...document.querySelectorAll(".ph")];
rows.forEach(p => p.addEventListener("click", () => {{
  const t = +p.dataset.t;
  orig.currentTime = t;
  edits.forEach(a => a.currentTime = editTime(t));  // all versions share the same cuts
}}));
orig.addEventListener("timeupdate", () => {{  // follow along while the original plays
  const t = orig.currentTime;
  let cur = null;
  for (const p of rows) {{ if (+p.dataset.t <= t) cur = p; else break; }}
  rows.forEach(p => p.classList.toggle("now", p === cur));
}});
document.querySelectorAll(".vtoggle button").forEach(b => b.addEventListener("click", () => {{
  document.querySelectorAll(".vtoggle button").forEach(x => x.classList.toggle("on", x === b));
  const v = b.dataset.v;  // "" = show every mute badge; otherwise only that version's
  document.querySelectorAll(".muterow").forEach(r =>
    r.classList.toggle("voff", v && !r.dataset.labels.split(",").includes(v)));
}}));
document.getElementById("done").addEventListener("click", async () => {{
  const st = document.getElementById("donest");
  st.textContent = "匯出中…（渲染需要一點時間）";
  try {{
    const r = await fetch("export", {{method: "POST"}});
    if (!r.ok) throw new Error(await r.text());
    st.textContent = "已匯出到 edit/out/：" + (await r.json()).files.join("、");
  }} catch (e) {{
    st.textContent = "匯出失敗 — 這顆按鈕需要用 python -m helpers.review … --serve 開啟本頁";
  }}
}});
</script>
</body></html>
"""


def build_html(transcript, spec, orig_src, edits, title="episode"):
    """edits: [(label, src)] — one player per edited version (all must share the same
    cuts; only mutes may differ, which don't move the timeline)."""
    if isinstance(edits, str):          # back-compat: a single edited source
        edits = [("剪輯後", edits)]
    rows, resolved, labels = _rows(transcript, spec)
    segs = compute_kept_segments(resolved, transcript["duration"])
    players = "\n".join(
        f'  <div class="player"><b>{_tw(label)}</b>'
        f'<audio class="edit" controls preload="metadata" src="{html.escape(src)}"></audio></div>'
        for label, src in edits)
    toggle = ""
    if len(labels) > 1:  # per-version mute filter, only when there IS a comparison
        btns = '<button data-v="" class="on">全部</button>' + "".join(
            f'<button data-v="{html.escape(l)}">只看 {html.escape(l)} 的靜音</button>' for l in labels)
        toggle = f'  <div class="vtoggle">靜音標記：{btns}</div>'
    return _PAGE.format(title=html.escape(title), orig_src=html.escape(orig_src),
                        players=players, toggle=toggle, rows="\n".join(rows),
                        segs=json.dumps([[round(s, 3), round(e, 3)] for s, e in segs]))


def parse_range(rng, size):
    """'bytes=a-b' / 'bytes=a-' / 'bytes=-n' -> (start, end) clamped to the file,
    or None when absent/malformed/unsatisfiable."""
    import re
    m = re.match(r"bytes=(\d*)-(\d*)$", rng or "")
    if not m or (not m.group(1) and not m.group(2)):
        return None
    if m.group(1):
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
    else:
        start, end = max(0, size - int(m.group(2))), size - 1
    if start >= size or start > end:
        return None
    return start, min(end, size - 1)


class _Limited:
    """File wrapper that stops after n bytes — copyfile() must not stream past the
    declared Content-Length of a 206 response."""
    def __init__(self, f, n):
        self.f, self.n = f, n

    def read(self, size=-1):
        if self.n <= 0:
            return b""
        chunk = self.f.read(self.n if size is None or size < 0 else min(size, self.n))
        self.n -= len(chunk)
        return chunk

    def close(self):
        self.f.close()


def serve(out_html, transcript_path, cuts_path, stems_dir, port=8765):
    """Serve the review page so the 完成 button works: POST …/export renders the
    approved cuts via render_stems — stems_dir/final.<ext> (integrated mix) + one
    final_<speaker>.<ext> per source track, in the source tracks' format. transcript.json
    and cuts.json are re-read from disk ON EVERY export, so the button always renders the
    cut list as it is NOW — a long-lived server must not export the stale spec it was
    started with. Server root is the audio dir (two levels above work/) so the page's
    relative audio srcs resolve."""
    import functools, http.server
    from .render import render_stems
    out_dir = os.path.dirname(os.path.abspath(out_html))          # edit/work
    root = os.path.dirname(os.path.dirname(out_dir))              # audio dir

    class Handler(http.server.SimpleHTTPRequestHandler):
        def send_head(self):
            # <audio> seeking sends Range requests; SimpleHTTPRequestHandler has no
            # 206 support, so the players restart from 0:00 on every click-to-seek.
            # Serve files ourselves with Accept-Ranges + partial content.
            path = self.translate_path(self.path)
            if not os.path.isfile(path):
                return super().send_head()
            size = os.path.getsize(path)
            rng = parse_range(self.headers.get("Range"), size)
            start, end = rng if rng else (0, size - 1)
            f = open(path, "rb")
            f.seek(start)
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Accept-Ranges", "bytes")
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            return _Limited(f, end - start + 1)

        def do_POST(self):
            # The server is loopback-only, but any web page open in the browser can
            # still POST to localhost (CSRF) — accept only same-origin requests.
            import re as _re
            origin = self.headers.get("Origin")
            if origin and not _re.match(r"https?://(127\.0\.0\.1|localhost)(:\d+)?$", origin):
                self.send_error(403, "cross-origin export rejected")
                return
            if not self.path.rstrip("/").endswith("/export"):
                self.send_error(404, "unknown endpoint")
                return
            try:
                transcript = json.load(open(transcript_path, encoding="utf-8"))
                spec = json.load(open(cuts_path, encoding="utf-8"))
                res = render_stems(transcript, spec["cuts"], transcript.get("audio"),
                                   stems_dir, mutes=spec.get("mutes"))
                body = json.dumps({"files": sorted(os.path.basename(f)
                                                   for f in res["outputs"].values())},
                                  ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self.send_error(500, type(e).__name__, str(e))

    handler = functools.partial(Handler, directory=root)
    url_path = os.path.relpath(os.path.abspath(out_html), root).replace(os.sep, "/")
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as srv:
        print(f"review server: http://127.0.0.1:{port}/{url_path}  (Ctrl+C to stop)")
        srv.serve_forever()


def build_original_mix(tracks, out_path):
    """Plain amix of the source tracks — the 'before' reference. normalize=0 keeps true
    per-track levels, so simultaneous speakers can sum past 0dBFS; the transient limiter
    (render's _MASTER, transparent below its -2dB ceiling) stops the reference player
    clipping into distortion, which would mislead A/B listening at the review gate."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    inputs = [x for t in tracks for x in ("-i", t)]
    subprocess.run(["ffmpeg", "-y", *inputs, "-filter_complex",
                    f"amix=inputs={len(tracks)}:normalize=0,{_MASTER}[m]", "-map", "[m]", out_path],
                   check=True, capture_output=True)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build the original-vs-edited review page.")
    p.add_argument("transcript"); p.add_argument("cuts"); p.add_argument("out_html")
    p.add_argument("--edited", action="append", default=None, metavar="LABEL=PATH",
                   help="edited audio, repeatable for multi-version comparison "
                        "(default: 剪輯後=<edit>/out/final.mp3)")
    p.add_argument("--mutes", action="append", default=None, metavar="LABEL=CUTS_JSON",
                   help="label mute badges by version: take mutes from each file and tag "
                        "them (overlaps collapse into one badge listing both labels)")
    p.add_argument("--serve", nargs="?", const=8765, type=int, default=None, metavar="PORT",
                   help="serve the page and enable the 完成 button (exports WAV mix + "
                        "per-speaker stems to <edit>/out)")
    a = p.parse_args()
    t = json.load(open(a.transcript, encoding="utf-8"))
    spec = json.load(open(a.cuts, encoding="utf-8"))
    if a.mutes:
        spec["mutes"] = [{**m, "label": lab}
                         for item in a.mutes
                         for lab, path in [item.split("=", 1)]
                         for m in json.load(open(path, encoding="utf-8")).get("mutes", [])]
    out_dir = os.path.dirname(a.out_html) or "."
    os.makedirs(out_dir, exist_ok=True)
    sources = t.get("tracks") or [t["audio"]]
    if len(sources) == 1:
        orig = sources[0]
    else:
        orig = build_original_mix(sources, os.path.join(out_dir, "original_mix.mp3"))
    rel = lambda p_: os.path.relpath(p_, out_dir).replace("\\", "/")
    default_final = os.path.join(os.path.dirname(a.transcript) or ".", "out", "final.mp3")
    edits = [(lab, rel(path)) for item in (a.edited or [f"剪輯後={default_final}"])
             for lab, path in [item.split("=", 1)]]
    page = build_html(t, spec, rel(orig), edits,
                      title=os.path.basename(os.path.abspath(out_dir + "/..")))
    with open(a.out_html, "w", encoding="utf-8") as f:
        f.write(page)
    print(a.out_html)
    if a.serve is not None:
        serve(a.out_html, a.transcript, a.cuts,
              os.path.join(os.path.dirname(a.transcript) or ".", "out"), a.serve)
