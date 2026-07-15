"""Provider abstraction: registry/selection logic + that ai_listen drives an injected
provider (no network — a fake provider stands in for Gemini/OpenAI)."""
import pytest

from helpers import ai_providers as ap
from helpers import ai_listen as ai
from test.fixtures import make_test_wav


class FakeProvider:
    """Records calls; returns a canned sweep hit (one throat-clear) regardless of prompt."""
    def __init__(self, model=None):
        self.model = model
        self.calls = []

    def generate(self, prompt, clip_path, json_mode=False):
        self.calls.append({"prompt": prompt, "json_mode": json_mode})
        return '[{"start": 1.6, "end": 1.9, "type": "throat_clear"}]'


# ── registry / factory ───────────────────────────────────────────────────────
def test_get_provider_unknown_name_raises():
    with pytest.raises(ValueError):
        ap.get_provider("not-a-provider")


def test_get_provider_selects_by_name(monkeypatch):
    monkeypatch.setitem(ap._PROVIDERS, "fake", FakeProvider)
    p = ap.get_provider("fake", model="m1")
    assert isinstance(p, FakeProvider) and p.model == "m1"


def test_get_provider_defaults_to_env(monkeypatch):
    monkeypatch.setitem(ap._PROVIDERS, "fake", FakeProvider)
    monkeypatch.setenv("AI_PROVIDER", "fake")
    assert isinstance(ap.get_provider(), FakeProvider)


def test_get_provider_explicit_name_overrides_env(monkeypatch):
    monkeypatch.setitem(ap._PROVIDERS, "fake", FakeProvider)
    monkeypatch.setitem(ap._PROVIDERS, "fake2", FakeProvider)
    monkeypatch.setenv("AI_PROVIDER", "fake")
    p = ap.get_provider("fake2")
    assert isinstance(p, FakeProvider)


def test_no_audio_llm_providers_enabled():
    # Both backends disabled by user request (Gemini 2026-07-11, OpenAI 2026-07-03);
    # detection is Scribe-only — see ai_providers.py to re-enable.
    assert "gemini" not in ap._PROVIDERS
    assert "openai" not in ap._PROVIDERS


def test_get_provider_error_points_to_scribe_fallback(monkeypatch):
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="scribe"):
        ap.get_provider()  # default (gemini) is disabled -> actionable error


# ── ai_listen uses the injected provider (backend-agnostic) ───────────────────
def test_sweep_track_uses_injected_provider(tmp_path):
    wav = str(tmp_path / "host.wav")
    make_test_wav(wav)  # 4s: tone, silence 1.5-2.2s, tone
    fp = FakeProvider()
    events = ai.sweep_track(wav, fp, window=12.0, hop=11.0, throttle=0)
    assert fp.calls, "provider was never called"
    assert fp.calls[0]["json_mode"] is True          # sweep asks for JSON
    assert len(events) == 1 and events[0]["type"] == "throat_clear"


def test_classify_uses_injected_provider(tmp_path):
    wav = str(tmp_path / "host.wav")
    make_test_wav(wav)

    class P:
        def generate(self, prompt, clip_path, json_mode=False):
            return "this is throat clearing"

    assert ai.classify(wav, 1.0, 2.0, P()) == "throat_clear"
