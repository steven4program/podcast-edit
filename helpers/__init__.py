"""Podcast-edit helpers.

Cross-platform I/O: force UTF-8 on stdout/stderr so a helper's zh-TW output (pack's reading
view, the JSON candidate dumps) survives being printed or redirected on Windows, where the
console/pipe default is a legacy code page (cp950) that raises on Chinese text. On macOS/Linux
UTF-8 is already the default, so this is a no-op there. Guarded: under pytest (and any wrapper
that replaces the streams) stdout has no .reconfigure — skip silently. File reads/writes pass
encoding="utf-8" explicitly at each call site; this covers only the process's stdio streams.
"""
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
