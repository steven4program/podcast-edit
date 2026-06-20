"""Shared test fixtures: a synthetic transcript and a synthetic WAV clip."""
import numpy as np
import soundfile as sf


def sample_transcript():
    """ids 0..7, two speakers, one cough event sitting in the 1.5-2.2s silence."""
    return {
        "audio": "fixture.wav",
        "duration": 4.0,
        "words": [
            {"id": 0, "text": "我覺得", "start": 0.0, "end": 0.4, "speaker": "host"},
            {"id": 1, "text": "我覺得", "start": 0.4, "end": 0.8, "speaker": "host"},
            {"id": 2, "text": "這個", "start": 0.8, "end": 1.1, "speaker": "host"},
            {"id": 3, "text": "東西", "start": 1.1, "end": 1.5, "speaker": "host"},
            {"id": 4, "text": "對啊", "start": 2.2, "end": 2.6, "speaker": "guest"},
            {"id": 5, "text": "就是", "start": 2.6, "end": 2.9, "speaker": "guest"},
            {"id": 6, "text": "就是", "start": 2.9, "end": 3.2, "speaker": "guest"},
            {"id": 7, "text": "很棒", "start": 3.2, "end": 3.7, "speaker": "guest"},
        ],
        "events": [
            {"type": "cough", "start": 1.6, "end": 1.9, "speaker": "host", "confidence": 0.8},
        ],
    }


def make_test_wav(path, sr=16000):
    """4s mono WAV: tone 0-1.5s, silence 1.5-2.2s, tone 2.2-3.7s, silence to 4s."""
    t = np.arange(int(4.0 * sr)) / sr
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    sig = np.zeros_like(t)
    sig[(t >= 0.0) & (t < 1.5)] = tone[(t >= 0.0) & (t < 1.5)]
    sig[(t >= 2.2) & (t < 3.7)] = tone[(t >= 2.2) & (t < 3.7)]
    sf.write(path, sig.astype(np.float32), sr)
