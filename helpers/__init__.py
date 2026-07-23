"""Podcast-edit helpers.

Cross-platform I/O: force UTF-8 on stdout/stderr so a helper's zh-TW output (pack's reading
view, the JSON candidate dumps) survives being printed or redirected on Windows, where the
console/pipe default is a legacy code page (cp950) that raises on Chinese text. On macOS/Linux
UTF-8 is already the default, so this is a no-op there. Guarded: under pytest (and any wrapper
that replaces the streams) stdout has no .reconfigure — skip silently. File reads/writes pass
encoding="utf-8" explicitly at each call site; this covers only the process's stdio streams.
"""
import os
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def env_path():
    """The project's own .env — anchored to the package, not the cwd (helpers run from the
    audio directory). This is the one setup tells you to create."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def _candidate_env_paths():
    """Where to look for .env, best first.

    The package-anchored path is the project's own file and always wins. A non-editable
    install (`pip install .` without -e) puts the package in site-packages, where that path
    points at a .env nobody created — so fall back to walking up from the cwd. Fallback
    only: from an audio directory nested in some other project, that walk can reach an
    unrelated .env, which must never shadow this project's own.
    """
    seen, out = set(), []
    for p in (env_path(), *[os.path.join(d, ".env") for d in _cwd_chain()]):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _cwd_chain():
    d = os.path.abspath(os.getcwd())
    while True:
        yield d
        parent = os.path.dirname(d)
        if parent == d:
            return
        d = parent


def _load_env(path=None):
    """Load .env so a filled-in key actually reaches the helpers.

    Setup tells you to write ELEVENLABS_API_KEY into .env, but nothing read it — the key
    only worked if you also exported it by hand, and a fresh shell died on a bare
    KeyError. Never overrides an already-exported variable: a real environment wins over
    a checked-out file.

    Returns the path actually loaded, or None.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:      # dependency not installed — exported vars still work
        return None
    for p in ([path] if path else _candidate_env_paths()):
        if not os.path.isfile(p):
            continue
        # utf-8-sig, and via a stream: Notepad (the obvious editor on Windows) saves a
        # UTF-8 BOM, which python-dotenv reads as part of the FIRST setting's name — the
        # key then silently never loads. Handed a decoded stream, the BOM is already gone.
        with open(p, encoding="utf-8-sig") as fh:
            if load_dotenv(stream=fh, override=False):
                return p
    return None


_DOTENV_PATH = _load_env()
_DOTENV_LOADED = _DOTENV_PATH is not None


def _names_in_env_file(path):
    """Setting names assigned in the .env file, ignoring comments and blanks."""
    try:
        with open(path, encoding="utf-8-sig") as fh:   # tolerate a Notepad-written BOM
            lines = fh.readlines()
    except OSError:
        return set()
    out = set()
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            out.add(line.split("=", 1)[0].strip())
    return out


def require_env(name, hint=""):
    """Fetch a required setting, or explain how to supply it.

    os.environ[...] raised a bare KeyError that reads like a bug in the code rather than a
    missing setting. Worse is telling someone to fill in a .env they already filled in:
    without python-dotenv installed the file is never read, so that case gets named
    outright instead of sending them back to a file that is already correct.
    """
    val = os.environ.get(name)
    if val:
        return val
    path = env_path()
    if not _DOTENV_LOADED:
        # Dead end to avoid: a correctly filled .env that is silently ignored, answered
        # with "fill in your .env". Name the uninstalled package instead.
        found = next((p for p in _candidate_env_paths()
                      if name in _names_in_env_file(p)), None)
        if found:
            raise SystemExit(
                f"{name} is set in {found}, but that file could not be read: the\n"
                f"python-dotenv package is not installed, so .env is being ignored.\n"
                f"Install the dependencies:\n"
                f"    pip install -e \".[ai,dev]\"\n"
                f"or export {name} in your shell instead.")
    raise SystemExit(
        f"{name} is not set.\n"
        f"Add it to the .env file at the repo root (see .env.example):\n"
        f"    {path}\n"
        f"    {name}=<your key>\n"
        f"or export it in your shell." + (f"\n{hint}" if hint else ""))
