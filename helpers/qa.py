"""Signal-level QA: flag abrupt seams and residual long silences."""
import json
import numpy as np
import soundfile as sf


def _rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if len(x) else 0.0


def _mmss(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def detect_seams(samples, sr, joins, win=0.05, ratio=4.0):
    n, flags = int(win * sr), []
    for t in joins:
        i = int(t * sr)
        before, after = _rms(samples[max(0, i - n):i]), _rms(samples[i:i + n])
        lo, hi = sorted((before + 1e-9, after + 1e-9))
        if hi / lo > ratio:
            flags.append({"time": t, "ratio": hi / lo})
    return flags


def joins_from_segments(segments):
    joins, acc = [], 0.0
    for s, e in segments[:-1]:
        acc += e - s
        joins.append(round(acc, 6))
    return joins


def _long_silences(samples, sr, thresh=0.01, min_len=2.0):
    quiet = np.abs(samples) < thresh
    out, run = [], 0
    for i, q in enumerate(quiet):
        if q:
            run += 1
        else:
            if run >= min_len * sr:
                out.append(((i - run) / sr, i / sr))
            run = 0
    if run >= min_len * sr:
        out.append(((len(quiet) - run) / sr, len(quiet) / sr))
    return out


def qa(audio_path, joins, out_md):
    samples, sr = sf.read(audio_path)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    seams, sils = detect_seams(samples, sr, joins), _long_silences(samples, sr)
    lines = ["# QA report", "", f"- seams flagged: {len(seams)}"]
    lines += [f"  - [{_mmss(s['time'])}] abrupt seam (ratio {s['ratio']:.1f})" for s in seams]
    lines.append(f"- long silences: {len(sils)}")
    lines += [f"  - [{_mmss(s)}–{_mmss(e)}] residual silence" for s, e in sils]
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    return {"seams": seams, "long_silences": sils}


if __name__ == "__main__":
    import sys
    kept = json.load(open(sys.argv[2]))  # *_kept_transcript.json (has "segments")
    print(json.dumps(qa(sys.argv[1], joins_from_segments(kept["segments"]), sys.argv[3]),
                     ensure_ascii=False))
