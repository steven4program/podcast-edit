English | [繁體中文](README.zh-TW.md)

# podcast-edit

Lean Claude Code skill for editing zh-TW (Mandarin) multitrack audio podcasts: removes
stutters / repeats / false-starts and the host's coughs / throat-clears, then renders a smooth,
loudness-leveled track (per-speaker speech leveling + -16 LUFS). The LLM makes the editorial calls by reading text; thin Python helpers transcribe,
detect, and render.

## Setup
```
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[ai,dev]"
cp .env.example .env   # ELEVENLABS_API_KEY (transcription; cough detection reuses its
                       # event tags — the Gemini/OpenAI sweep backends are currently disabled)
```
Requires `ffmpeg` / `ffprobe` on PATH. Cross-platform (macOS/Linux/Windows) — helpers force
UTF-8 I/O and store source paths posix-style so a transcript works on either OS.

## Use
It's a Claude Code skill. Run it from the folder holding your audio tracks (one mono file per
speaker) and follow the workflow in [`SKILL.md`](SKILL.md): transcribe → propose cuts → preview
→ approve → final render → chapters. Artifacts land under `<audio_dir>/edit/` (`out/final.mp3`
is the result).

## Develop
Run the tests with `pytest test/ -q`. Agent / contributor guidance is in
[`AGENTS.md`](AGENTS.md); the editing heuristics the model follows are in
[`references/edit-heuristics.md`](references/edit-heuristics.md).
