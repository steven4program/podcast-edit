"""Settings reach the helpers from .env — the setup instructions promise this.

Never touches the real repo-root .env: it holds a live API key, and a test that moves a
secret aside can leave it moved if the run is interrupted.
"""
import os

import pytest

import helpers


def test_require_env_returns_an_exported_value(monkeypatch):
    monkeypatch.setenv("PE_TEST_KEY", "abc123")
    assert helpers.require_env("PE_TEST_KEY") == "abc123"


def test_require_env_explains_itself_instead_of_raising_keyerror(monkeypatch):
    # os.environ["..."] raised a bare KeyError that reads like a code bug, so a user with a
    # correctly filled .env had nothing to go on. The message must name the setting and .env.
    monkeypatch.delenv("PE_TEST_KEY", raising=False)
    with pytest.raises(SystemExit) as e:
        helpers.require_env("PE_TEST_KEY", "hint line")
    msg = str(e.value)
    assert "PE_TEST_KEY" in msg and ".env" in msg and "hint line" in msg


def test_require_env_treats_empty_as_missing(monkeypatch):
    # .env.example ships `GEMINI_API_KEY=` — an empty value is not a usable setting.
    monkeypatch.setenv("PE_TEST_KEY", "")
    with pytest.raises(SystemExit):
        helpers.require_env("PE_TEST_KEY")


def test_env_path_anchors_to_the_repo_not_the_cwd(tmp_path, monkeypatch):
    # The helpers are run from the audio directory, so a cwd-relative lookup finds nothing.
    monkeypatch.chdir(tmp_path)
    p = helpers.env_path()
    assert os.path.basename(p) == ".env"
    assert os.path.isfile(os.path.join(os.path.dirname(p), "pyproject.toml"))


def test_dotenv_values_are_loaded(tmp_path, monkeypatch):
    # The bug: nothing ever called load_dotenv, so a filled-in .env was dead weight and
    # transcription died on a bare KeyError in a fresh shell.
    monkeypatch.delenv("PE_FROM_DOTENV", raising=False)
    f = tmp_path / ".env"
    f.write_text("PE_FROM_DOTENV=from_file\n", encoding="utf-8")
    assert helpers._load_env(str(f)) == str(f)
    assert os.environ["PE_FROM_DOTENV"] == "from_file"


def test_exported_variable_wins_over_dotenv(tmp_path, monkeypatch):
    # A real environment (CI secret, shell export) must beat a checked-out file.
    monkeypatch.setenv("PE_FROM_DOTENV", "from_shell")
    f = tmp_path / ".env"
    f.write_text("PE_FROM_DOTENV=from_file\n", encoding="utf-8")
    helpers._load_env(str(f))
    assert os.environ["PE_FROM_DOTENV"] == "from_shell"


def test_missing_dotenv_file_is_not_an_error(tmp_path):
    # Exported-only setups have no .env at all; importing helpers must still work.
    assert helpers._load_env(str(tmp_path / "nope.env")) is None


def test_projects_own_dotenv_is_searched_first(tmp_path, monkeypatch):
    # A non-editable install cannot find .env via the package, so the cwd walk is the
    # fallback -- but it must never let a stray .env upstream of the audio directory
    # shadow the project's own file.
    monkeypatch.chdir(tmp_path)
    paths = helpers._candidate_env_paths()
    assert paths[0] == helpers.env_path()
    assert str(tmp_path / ".env") in paths


def test_cwd_walk_finds_dotenv_in_a_parent_directory(tmp_path, monkeypatch):
    # `pip install .` (no -e) puts the package in site-packages: env_path() points at a
    # .env nobody created, so the walk up from the cwd is the only way to find the real one.
    nested = tmp_path / "episode" / "tracks"
    nested.mkdir(parents=True)
    (tmp_path / ".env").write_text("PE_WALK_KEY=found\n", encoding="utf-8")
    monkeypatch.chdir(nested)
    monkeypatch.delenv("PE_WALK_KEY", raising=False)
    monkeypatch.setattr(helpers, "env_path", lambda: str(tmp_path / "site-packages" / ".env"))
    assert helpers._load_env() == str(tmp_path / ".env")
    assert os.environ["PE_WALK_KEY"] == "found"


def test_cwd_chain_terminates_at_the_filesystem_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    chain = list(helpers._cwd_chain())
    assert chain[0] == os.path.abspath(str(tmp_path))
    assert os.path.dirname(chain[-1]) == chain[-1]   # reached the root, did not loop


def test_names_in_env_file_ignores_comments_and_blanks(tmp_path):
    f = tmp_path / ".env"
    f.write_text("A=1\n\n# B=2\n  # C=3\nD =4\nnot_an_assignment\n", encoding="utf-8")
    assert helpers._names_in_env_file(str(f)) == {"A", "D"}


def test_names_in_env_file_survives_a_missing_file(tmp_path):
    assert helpers._names_in_env_file(str(tmp_path / "nope.env")) == set()


@pytest.mark.parametrize("label,body", [
    ("lf",          b"PE_BOM_KEY=v\n"),                 # mac/linux
    ("crlf",        b"PE_BOM_KEY=v\r\n"),               # windows
    ("bom_crlf",    b"\xef\xbb\xbfPE_BOM_KEY=v\r\n"),   # windows Notepad
    ("bom_lf",      b"\xef\xbb\xbfPE_BOM_KEY=v\n"),
])
def test_dotenv_loads_regardless_of_line_endings_or_bom(tmp_path, monkeypatch, label, body):
    # A BOM became part of the FIRST setting's name, so the key silently never loaded and
    # the user -- who had filled .env in correctly -- was told to fill in .env.
    monkeypatch.delenv("PE_BOM_KEY", raising=False)
    f = tmp_path / f"{label}.env"
    f.write_bytes(body)
    assert helpers._load_env(str(f)) == str(f)
    assert os.environ["PE_BOM_KEY"] == "v"
    assert helpers._names_in_env_file(str(f)) == {"PE_BOM_KEY"}


def test_filled_in_env_with_dotenv_uninstalled_names_the_real_cause(monkeypatch):
    # The dead end: pull the repo, fill in .env correctly, skip `pip install -e .` -> the
    # file is silently ignored and the generic message tells you to fill in the file you
    # already filled in. It must point at the uninstalled package instead.
    monkeypatch.delenv("PE_TEST_KEY", raising=False)
    monkeypatch.setattr(helpers, "_DOTENV_LOADED", False)
    monkeypatch.setattr(helpers, "_names_in_env_file", lambda p: {"PE_TEST_KEY"})
    with pytest.raises(SystemExit) as e:
        helpers.require_env("PE_TEST_KEY")
    msg = str(e.value)
    assert "python-dotenv" in msg and "pip install" in msg
    assert "Add it to the .env file" not in msg   # not the misleading advice


def test_genuinely_absent_key_still_gets_the_fill_in_dotenv_advice(monkeypatch):
    monkeypatch.delenv("PE_TEST_KEY", raising=False)
    monkeypatch.setattr(helpers, "_DOTENV_LOADED", False)
    monkeypatch.setattr(helpers, "_names_in_env_file", lambda p: set())
    with pytest.raises(SystemExit) as e:
        helpers.require_env("PE_TEST_KEY")
    assert "Add it to the .env file" in str(e.value)
