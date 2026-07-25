#!/usr/bin/env python3
"""PreToolUse guard (Write|Edit|Bash|PowerShell): keep the repo free of improvised scratch.

Ships with the repo so the rule enforces for every user, not just whoever set it up. It denies
the three things an agent tends to improvise into the project:
  1. any write into `source/` (the untouchable original audio dir) via Write/Edit,
  2. a new script file (.py/.sh/.js/.ts/.mjs/.cjs) anywhere in the repo EXCEPT `helpers/`,
     `test/`, and `.claude/hooks/` (where code legitimately lives), and
  3. a scratch/debug file (.err/.log/.tmp/…) ANYWHERE in the repo — these are the throwaway
     redirect targets (`… 2>rp.err`) that dirty the tree and make runs non-reproducible.

Crucially it inspects Bash/PowerShell commands too, not just Write/Edit: the scratch `.err`
files this rule exists to stop are created by a shell *redirect*, which never goes through the
Write tool. Redirect targets (`>`, `>>`, `2>`, `&>`, `tee`) are resolved against the repo root
and run through the same path rules — but the `source/` blanket block is Write/Edit-only, because
the documented pipeline legitimately redirects artifacts into `source/edit/`
(`python -m helpers.pack … > edit/work/packed.md`). Targets containing shell expansion
(`$VAR`, globs, `~`) are skipped — they can't be resolved statically, so we default to allow.

Everything else — docs, json, toml, edits to real helpers/tests, `/dev/null`, temp/scratchpad
writes — passes through untouched. A recurring mechanical step should become a tested helper
instead (see helpers/compile_cuts.py); one-off scratch goes to the OS temp dir.

Reads the PreToolUse JSON on stdin; on a violation prints a deny decision and exits 0.
Pure decision logic lives in `decide()` so test/test_guard_repo_writes.py can exercise it offline.
"""
import json, os, re, sys

_SCRIPT_EXTS = {".py", ".sh", ".js", ".ts", ".mjs", ".cjs"}
_SCRATCH_EXTS = {".err", ".log", ".tmp", ".temp", ".bak", ".orig", ".swp", ".swo"}
_CODE_DIRS = ("helpers/", "test/", ".claude/hooks/")
_UNSAFE_SHELL = set("$`*?~{}[]!")            # target we can't resolve statically → skip (allow)

# Redirect targets: `>`, `>>`, `2>`, `1>>`, `&>`, `&>>`. The captured target must start with a
# non-`&` char, so `2>&1` (fd dup, target `&1`) never matches. `tee [-a] FILE` too.
_REDIR = re.compile(r"(?:\d+|&)?>>?\s*([^\s|&;<>()]+)")
_TEE = re.compile(r"\btee\b(?:\s+-a\b)?\s+([^\s|&;<>()]+)")


def _repo_root():
    # Claude sets CLAUDE_PROJECT_DIR; fall back to this file's location (.claude/hooks/..).
    root = os.environ.get("CLAUDE_PROJECT_DIR") \
        or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.abspath(root)


def _rel_in_repo(path, root):
    """Return the repo-relative posix path if `path` lands inside the repo, else None."""
    ap = os.path.abspath(os.path.join(root, path) if not os.path.isabs(path) else path)
    try:
        rel = os.path.relpath(ap, root)
    except ValueError:                        # different drive on Windows → outside the repo
        return None
    if rel.startswith(".."):                  # outside the repo (OS temp/scratchpad) → allowed
        return None
    return rel.replace(os.sep, "/")


def _check_rel(rel, block_source):
    """Path rules shared by every write path. Returns a deny reason, or None to allow.
    `block_source` gates the source/ blanket rule (Write/Edit only — shell redirects into
    source/edit/ are the documented pipeline)."""
    ext = os.path.splitext(rel)[1].lower()
    if ext in _SCRATCH_EXTS:
        return (f"Refusing to write a scratch/debug file ({rel}) inside the repo. "
                f"'.err/.log/.tmp/…' output never belongs in the project — it dirties the tree "
                f"and makes runs non-reproducible. Send it to /dev/null or the OS temp/scratchpad "
                f"dir, and run SKILL.md commands verbatim without adding ad-hoc redirects.")
    if ext in _SCRIPT_EXTS and not rel.startswith(_CODE_DIRS):
        return (f"Refusing to create a script ({rel}) outside helpers/, test/, .claude/hooks/. "
                f"Don't improvise throwaway scripts in the repo: a recurring mechanical step "
                f"belongs in a tested helper (see helpers/compile_cuts.py); one-off scratch goes "
                f"to the OS temp dir, never the project.")
    if block_source and (rel == "source" or rel.startswith("source/")):
        return ("Writing into source/ is not allowed — it holds the untouchable original audio. "
                "All artifacts go under <audio_dir>/edit/.")
    return None


def _shell_targets(command):
    """Yield redirect / tee targets from a shell command, skipping ones with shell expansion
    (a `$VAR`/glob/`~` target can't be resolved statically, so we don't guess)."""
    for m in list(_REDIR.finditer(command)) + list(_TEE.finditer(command)):
        tgt = m.group(1).strip("'\"")
        if tgt and not (set(tgt) & _UNSAFE_SHELL):
            yield tgt


def decide(tool_name, tool_input, root):
    """Return a deny reason string, or None to allow. Pure — no I/O."""
    tool_input = tool_input or {}
    if tool_name in ("Write", "Edit"):
        path = tool_input.get("file_path")
        if not path:
            return None
        rel = _rel_in_repo(path, root)
        return _check_rel(rel, block_source=True) if rel is not None else None
    if tool_name in ("Bash", "PowerShell"):
        command = tool_input.get("command") or ""
        for tgt in _shell_targets(command):
            rel = _rel_in_repo(tgt, root)
            if rel is not None:
                reason = _check_rel(rel, block_source=False)  # source/edit redirects are the pipeline
                if reason:
                    return reason
        return None
    return None


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)                           # never break a tool call on a malformed payload
    reason = decide(data.get("tool_name"), data.get("tool_input"), _repo_root())
    if reason:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "deny",
            "permissionDecisionReason": reason}}))
    sys.exit(0)


if __name__ == "__main__":
    main()
