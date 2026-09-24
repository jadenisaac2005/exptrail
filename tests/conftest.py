import subprocess
import warnings

import pytest


def git(cwd, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """An empty cwd outside any git repo."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EXPTRAIL_ROOT", raising=False)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    return tmp_path


@pytest.fixture
def repo(workdir):
    """A clean git repo with one commit, as cwd."""
    git(workdir, "init", "-q")
    (workdir / "train.py").write_text("print('hi')\n")
    (workdir / ".gitignore").write_text("*.npz\n")
    git(workdir, "add", ".")
    git(workdir, "commit", "-qm", "init")
    git(workdir, "remote", "add", "origin", "https://user:secret@github.com/me/proj.git")
    return workdir


@pytest.fixture
def quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
