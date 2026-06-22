"""Optional Gemini clip classifier for cough recall and seam doubt.

Precision-first: only used to confirm a candidate is a real cough/throat_clear before cutting.
Laughter must NEVER be cut, so a 'laughter' label means keep.
"""
import os, subprocess, tempfile

_KEYWORDS = [("throat", "throat_clear"), ("cough", "cough"), ("laugh", "laughter"),
             ("breath", "breath"), ("speech", "speech"), ("talk", "speech")]
_PROMPT = ("Classify this short audio clip as exactly one of: cough, throat clearing, "
           "laughter, breath, speech. Answer with the single best label.")


def parse_label(text):
    low = text.lower()
    for key, label in _KEYWORDS:
        if key in low:
            return label
    return "other"


def _extract_clip(audio_path, start, end):
    fd, clip = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", audio_path,
                    "-ac", "1", "-ar", "16000", clip], check=True, capture_output=True)
    return clip


def classify(audio_path, start, end, model="gemini-2.5-flash"):
    from google import genai
    clip = _extract_clip(audio_path, start, end)
    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        uploaded = client.files.upload(file=clip)
        resp = client.models.generate_content(model=model, contents=[_PROMPT, uploaded])
        return parse_label(resp.text)
    finally:
        os.remove(clip)
