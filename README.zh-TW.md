[English](README.md) | 繁體中文

# podcast-edit

精簡的 Claude Code skill，用來剪輯 zh-TW（華語）**多軌**音訊 podcast：移除結巴／重複／false-start
以及主持人的咳嗽／清喉嚨，再渲染出順暢、音量平穩的音軌（逐講者語音等化 + -16 LUFS）。編輯判斷由 LLM 讀文字決定；輕量 Python helper
只做轉錄、偵測與渲染等機械工作。

## 安裝
```
python -m venv .venv && source .venv/bin/activate   # Windows：.venv\Scripts\activate
pip install -e ".[ai,dev]"
cp .env.example .env   # ELEVENLABS_API_KEY（轉錄）+ 咳嗽偵測 provider 金鑰
                       # （AI_PROVIDER=gemini 加 GEMINI_API_KEY；OpenAI 後端目前停用）
```
需要 PATH 上有 `ffmpeg` / `ffprobe`。跨平台（macOS/Linux/Windows）—— helpers 強制 UTF-8
輸入輸出、來源路徑以正斜線儲存，所以 transcript 在任一 OS 都能用。

## 使用
這是一個 Claude Code skill。在放音訊軌的資料夾（每位講者一個單聲道檔）裡執行，照
[`SKILL.md`](SKILL.md) 的流程走：轉錄 → 提案剪輯 → 預覽 → 核准 → final render → 章節。
產物會放在 `<audio_dir>/edit/`（`out/final.mp3` 是成品）。

## 開發
用 `pytest test/ -q` 跑測試。Agent／貢獻者指南見 [`AGENTS.md`](AGENTS.md)；模型遵循的剪輯
啟發法則見 [`references/edit-heuristics.md`](references/edit-heuristics.md)。
