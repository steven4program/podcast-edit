"""Pluggable audio-LLM providers for the cough sweep / clip classifier.

The detection logic in ai_listen.py needs exactly one thing from a model: give it a short
audio clip + a text prompt, get back the model's text answer. That single method is the whole
contract, so swapping or adding a backend is one class here + one registry entry — ai_listen.py
stays provider-agnostic. `json_mode` asks the backend to return JSON (used by the sweep); the
caller still parses defensively, so a backend that can't hard-enforce JSON is fine.

SDKs are imported lazily inside each provider, so you only need the deps for the one you use.
"""
import os

# import base64  # only needed by the disabled OpenAIProvider below


class GeminiProvider:
    """Google Gemini (native multimodal audio). Needs GEMINI_API_KEY + `google-genai`."""
    name = "gemini"
    default_model = "gemini-2.5-flash"

    def __init__(self, model=None):
        from google import genai
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model or self.default_model

    def generate(self, prompt, clip_path, json_mode=False):
        uploaded = self.client.files.upload(file=clip_path)
        kwargs = {"config": {"response_mime_type": "application/json"}} if json_mode else {}
        resp = self.client.models.generate_content(
            model=self.model, contents=[prompt, uploaded], **kwargs)
        return resp.text


# OpenAI backend DISABLED by user request (2026-07-03) — uncomment the class and its
# registry entry below to re-enable (needs OPENAI_API_KEY + `pip install openai`).
# class OpenAIProvider:
#     """OpenAI audio-capable chat model. Needs OPENAI_API_KEY + `openai`. Clips are the
#     16k mono WAVs from _extract_clip, sent inline as base64 (format='wav')."""
#     name = "openai"
#     default_model = "gpt-audio"  # stable alias for the current audio-capable chat model
#
#     def __init__(self, model=None):
#         from openai import OpenAI
#         self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
#         self.model = model or self.default_model
#
#     def generate(self, prompt, clip_path, json_mode=False):
#         # Note: audio chat models (gpt-audio*) reject response_format=json_object, so we don't
#         # hard-enforce JSON — the sweep prompt already asks for a JSON array and the caller
#         # parses defensively (_parse_events regexes the array out). json_mode is thus advisory.
#         with open(clip_path, "rb") as f:
#             b64 = base64.b64encode(f.read()).decode()
#         resp = self.client.chat.completions.create(
#             model=self.model,
#             messages=[{"role": "user", "content": [
#                 {"type": "text", "text": prompt},
#                 {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
#             ]}])
#         return resp.choices[0].message.content


# Registry: name -> provider class. Add a backend by defining a class with the same
# (model=None) __init__ and .generate(prompt, clip_path, json_mode) and listing it here.
_PROVIDERS = {p.name: p for p in (GeminiProvider,)}  # OpenAIProvider disabled (see above)


def get_provider(name=None, model=None):
    """Build a provider by name (default: $AI_PROVIDER, else 'gemini'). `model` overrides
    that provider's default_model. Raises ValueError on an unknown name."""
    name = (name or os.environ.get("AI_PROVIDER") or "gemini").lower()
    try:
        cls = _PROVIDERS[name]
    except KeyError:
        raise ValueError(f"unknown AI provider {name!r}; choose from {sorted(_PROVIDERS)}")
    return cls(model=model)
