"""Offline checks for the PreToolUse repo-write guard (.claude/hooks/guard_repo_writes.py).

Locks the two things that matter: the guard denies improvised scratch (the `.err` redirect that
prompted this rule, plus script/source violations) and it does NOT block the documented pipeline
(shell redirects into source/edit/, /dev/null, temp, $VAR targets, real helper/test edits)."""
import importlib.util
import os

_GUARD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      ".claude", "hooks", "guard_repo_writes.py")
_spec = importlib.util.spec_from_file_location("guard_repo_writes", _GUARD)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)

ROOT = "/repo"


def deny(tool, inp):
    return guard.decide(tool, inp, ROOT)


# --- the rule this change exists for: scratch redirects via the shell ---
def test_bash_stderr_redirect_to_err_denied():
    assert deny("Bash", {"command": "python -m helpers.repeats t.json 2>rp.err"})

def test_bash_stdout_redirect_to_log_denied():
    assert deny("Bash", {"command": "python x.py > debug.log"})

def test_bash_append_and_tee_scratch_denied():
    assert deny("Bash", {"command": "echo hi >> notes.tmp"})
    assert deny("Bash", {"command": "python x.py | tee out.err"})

def test_powershell_redirect_scratch_denied():
    assert deny("PowerShell", {"command": "python x.py > run.log"})

def test_write_scratch_ext_denied_anywhere():
    assert deny("Write", {"file_path": "source/edit/work/foo.err"})
    assert deny("Write", {"file_path": "foo.log"})


# --- must NOT break the documented pipeline ---
def test_pipeline_redirects_into_source_edit_allowed():
    assert deny("Bash", {"command": "python -m helpers.pack t.json > source/edit/work/packed.md"}) is None
    assert deny("Bash", {"command": "python -m helpers.repeats t.json > source/edit/work/repeats.json"}) is None

def test_dev_null_and_fd_dup_allowed():
    assert deny("Bash", {"command": "python x.py 2>/dev/null"}) is None
    assert deny("Bash", {"command": "python x.py > out.txt 2>&1"}) is None  # out.txt not scratch

def test_shell_variable_target_skipped():
    assert deny("Bash", {"command": 'python x.py > "$TMPDIR/transcribe.log"'}) is None

def test_temp_dir_outside_repo_allowed():
    assert deny("Bash", {"command": "python x.py 2>/tmp/scratch.err"}) is None

def test_plain_command_no_redirect_allowed():
    assert deny("Bash", {"command": "ffprobe -v error source/x.wav"}) is None


# --- existing rules still hold ---
def test_new_script_outside_code_dirs_denied():
    assert deny("Write", {"file_path": "scratch.py"})
    assert deny("Bash", {"command": "cat > helpers/../evil.sh"})

def test_script_in_code_dirs_allowed():
    assert deny("Write", {"file_path": "helpers/new_helper.py"}) is None
    assert deny("Write", {"file_path": "test/test_x.py"}) is None

def test_write_into_source_denied_but_bash_redirect_there_ok():
    assert deny("Write", {"file_path": "source/foo.txt"})                       # Write/Edit blocked
    assert deny("Bash", {"command": "echo x > source/edit/out/chapters.txt"}) is None  # pipeline ok

def test_docs_and_json_allowed():
    assert deny("Write", {"file_path": "docs/plan.md"}) is None
    assert deny("Edit", {"file_path": "helpers/render.py"}) is None
