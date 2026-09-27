"""What counts as a dirty tree, and what ends up in git_diff.patch."""

import hashlib
import json
import subprocess
import warnings

import pytest

from exptrail import ConfigWarning, DirtyTreeWarning, Run
from exptrail import meta as meta_mod
from exptrail.cli import main

from .conftest import git


def read_meta(run):
    return json.loads((run.dir / "meta.json").read_text())["git"]


def patch_of(run):
    return (run.dir / "git_diff.patch").read_text()


def run_clean(name="clean", **kwargs):
    """Start and finish a run, failing if any warning is raised."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with Run(name, **kwargs) as run:
            run.summary(acc=1.0)
    return run


def commit_all(repo, msg="more"):
    git(repo, "add", ".")
    git(repo, "commit", "-qm", msg)


def test_untracked_imported_module_is_dirty_and_saved(repo, tmp_path_factory):
    # Stage something first, so "index unchanged" is a meaningful check.
    (repo / "train.py").write_text("import difftok\nprint(difftok.tok('x'))\n")
    git(repo, "add", "train.py")
    module = "def tok(s):\n    return s.split()\n"
    (repo / "difftok.py").write_text(module)
    index_before = (repo / ".git" / "index").read_bytes()
    status_before = git(repo, "status", "--porcelain")

    with pytest.warns(DirtyTreeWarning, match=r"difftok\.py \(untracked\)"):
        with Run("newmod") as run:
            pass

    info = read_meta(run)
    assert info["dirty"] is True
    assert info["dirty_files"] == {"tracked": ["train.py"], "untracked": ["difftok.py"]}
    assert info["ignored_files"] == []
    patch = patch_of(run)
    assert "new file mode" in patch
    assert "".join(f"+{line}\n" for line in module.splitlines()) in patch
    assert "+import difftok" in patch

    # The user's index is byte-for-byte untouched: difftok.py is still untracked.
    assert (repo / ".git" / "index").read_bytes() == index_before
    assert git(repo, "status", "--porcelain") == status_before + "\n?? runs/"
    assert "?? difftok.py" in status_before

    # commit + patch really does reproduce the working tree.
    clone = tmp_path_factory.mktemp("clone")
    git(clone, "clone", "-q", str(repo), ".")
    git(clone, "apply", str(run.dir / "git_diff.patch"))
    assert (clone / "difftok.py").read_text() == module
    assert (clone / "train.py").read_text() == (repo / "train.py").read_text()


def test_untracked_csv_is_ignored(repo):
    (repo / "results.csv").write_text("a,b\n1,2\n")
    (repo / "out.log").write_text("hello\n")
    run = run_clean()
    info = read_meta(run)
    assert info["dirty"] is False
    assert info["dirty_files"] == {"tracked": [], "untracked": []}
    assert "changed_files" not in info
    assert not (run.dir / "git_diff.patch").exists()


def test_untracked_code_in_gitignore_or_virtualenv_is_ignored(repo):
    (repo / ".gitignore").write_text("*.npz\nbuild/\n")
    commit_all(repo)
    (repo / "build").mkdir()
    (repo / "build" / "gen.py").write_text("x = 1\n")
    venv = repo / "venv" / "lib" / "site-packages"
    venv.mkdir(parents=True)
    (repo / "venv" / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (venv / "pkg.py").write_text("y = 2\n")
    assert read_meta(run_clean())["dirty"] is False


def test_artifacts_in_runs_folder_are_not_code(repo, quiet):
    (repo / "params.yaml").write_text("lr: 0.1\n")
    commit_all(repo)
    with Run("first") as run:
        run.save_artifact("params.yaml")
        run.save_artifact("train.py")
    assert read_meta(run_clean("second"))["dirty"] is False


def test_ignored_tracked_change_is_clean_but_in_patch(repo):
    (repo / "LOG.md").write_text("# log\n")
    commit_all(repo)
    (repo / "LOG.md").write_text("# log\nleaderboard: 0.9807\n")

    run = run_clean(dirty_ignore=["*.md"])
    info = read_meta(run)
    assert info["dirty"] is False
    assert info["ignored_files"] == ["LOG.md"]
    assert info["dirty_ignore"] == ["*.md"]
    assert "+leaderboard: 0.9807" in patch_of(run)


def test_dirty_ignore_path_patterns(repo):
    (repo / "submissions" / "week1").mkdir(parents=True)
    (repo / "submissions" / "week1" / "LOG.txt").write_text("a\n")
    commit_all(repo)
    (repo / "submissions" / "week1" / "LOG.txt").write_text("b\n")
    (repo / "submissions" / "make_sub.py").write_text("print(1)\n")  # untracked code, ignored

    info = read_meta(run_clean(dirty_ignore=["submissions/**"]))
    assert info["dirty"] is False
    assert sorted(info["ignored_files"]) == ["submissions/make_sub.py", "submissions/week1/LOG.txt"]

    # An ignored change alongside a real one: dirty, and only the real one is blamed.
    (repo / "train.py").write_text("print('changed')\n")
    with pytest.warns(DirtyTreeWarning):
        with Run("mixed", dirty_ignore=["submissions/"]) as run:
            pass
    info = read_meta(run)
    assert info["dirty_files"] == {"tracked": ["train.py"], "untracked": []}
    assert "submissions/week1/LOG.txt" in info["ignored_files"]


def test_oversized_untracked_code_is_hashed(repo):
    big = ("x = 1\n" * 200_000).encode()  # 1.2 MB
    assert len(big) > meta_mod.MAX_UNTRACKED_BYTES
    (repo / "big.py").write_bytes(big)
    (repo / "small.py").write_text("y = 2\n")

    with pytest.warns(DirtyTreeWarning):
        with Run("big") as run:
            pass

    info = read_meta(run)
    assert info["dirty_files"]["untracked"] == ["small.py", "big.py"]
    assert info["untracked_too_large"] == [
        {"path": "big.py", "size": len(big), "sha256": hashlib.sha256(big).hexdigest()}
    ]
    patch = patch_of(run)
    assert "x = 1" not in patch and "+y = 2" in patch
    assert f"big.py  size={len(big)}  sha256={hashlib.sha256(big).hexdigest()}" in patch
    assert (repo / ".git" / "index").exists() and "big.py" not in git(repo, "ls-files")


def test_kwarg_validation():
    with pytest.raises(TypeError, match="dirty_ignore"):
        Run("bad", dirty_ignore="*.md")
    with pytest.raises(TypeError, match="untracked_code"):
        Run("bad", untracked_code=["*.py", 3])


def write_pyproject(repo, body):
    (repo / "pyproject.toml").write_text(body)
    commit_all(repo, "pyproject")


needs_tomllib = pytest.mark.skipif(meta_mod.tomllib is None, reason="tomllib needs Python 3.11+")


@needs_tomllib
def test_pyproject_settings_are_used(repo):
    write_pyproject(repo, '[tool.exptrail]\ndirty_ignore = ["*.md"]\nuntracked_code = ["*.py", "*.csv"]\n')
    (repo / "NOTES.md").write_text("x\n")
    commit_all(repo)
    (repo / "NOTES.md").write_text("y\n")
    info = read_meta(run_clean())
    assert info["dirty"] is False
    assert info["ignored_files"] == ["NOTES.md"]
    assert info["untracked_code"] == ["*.py", "*.csv"]

    (repo / "data.csv").write_text("1\n")  # now counts as code
    with pytest.warns(DirtyTreeWarning, match="data.csv"):
        with Run("csv"):
            pass


@needs_tomllib
def test_kwargs_override_pyproject(repo):
    write_pyproject(repo, '[tool.exptrail]\ndirty_ignore = ["*.md"]\n')
    (repo / "NOTES.md").write_text("x\n")
    commit_all(repo)
    (repo / "NOTES.md").write_text("y\n")
    with pytest.warns(DirtyTreeWarning, match="NOTES.md"):
        with Run("override", dirty_ignore=[]) as run:
            pass
    assert read_meta(run)["dirty_ignore"] == []

    (repo / "train.py").write_text("print('changed')\n")
    info = read_meta(run_clean(dirty_ignore=["*.md", "train.py"]))
    assert sorted(info["ignored_files"]) == ["NOTES.md", "train.py"]


@needs_tomllib
def test_invalid_pyproject(repo):
    write_pyproject(repo, "[tool.exptrail\n")
    with pytest.warns(ConfigWarning, match="could not parse"):
        with Run("badtoml"):
            pass
    write_pyproject(repo, '[tool.exptrail]\ndirty_ignore = "*.md"\n')
    with pytest.raises(TypeError, match=r"\[tool.exptrail\] dirty_ignore"):
        with Run("badtype"):
            pass


def test_python310_warns_once_and_ignores_pyproject(repo, monkeypatch):
    monkeypatch.setattr(meta_mod, "tomllib", None)
    monkeypatch.setattr(meta_mod, "_warned_no_tomllib", False)
    write_pyproject(repo, '[tool.exptrail]\ndirty_ignore = ["*.md"]\n')
    (repo / "NOTES.md").write_text("x\n")
    commit_all(repo)
    (repo / "NOTES.md").write_text("y\n")

    with pytest.warns(ConfigWarning, match="ignored on Python 3.10") as record:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            with pytest.warns(DirtyTreeWarning):  # *.md from pyproject was not applied
                with Run("py310"):
                    pass
    assert sum(issubclass(w.category, ConfigWarning) for w in record) == 1

    # Only once per process; keyword arguments still work.
    run = run_clean(dirty_ignore=["*.md"])
    assert read_meta(run)["dirty"] is False


def test_python310_no_warning_without_section(repo, monkeypatch):
    monkeypatch.setattr(meta_mod, "tomllib", None)
    monkeypatch.setattr(meta_mod, "_warned_no_tomllib", False)
    write_pyproject(repo, '[project]\nname = "x"\n\n[tool.other]\na = 1\n')
    run_clean()


def test_verify_strict_passes_when_only_ignored_files_changed(repo, quiet):
    (repo / "LOG.md").write_text("# log\n")
    commit_all(repo)
    (repo / "LOG.md").write_text("# log\nscore: 0.5\n")
    with Run("ignored-only", dirty_ignore=["*.md"]) as run:
        run.summary(test_acc=0.5)
    assert main(["table", "--metric", "test_acc", "--readme", "README.md"]) == 0
    assert "(dirty)" not in (repo / "README.md").read_text()
    assert main(["verify", "README.md", "--strict"]) == 0


def test_patch_survives_non_utf8_content(repo):
    (repo / "latin.py").write_bytes(b"s = '\xe9'\n")
    with pytest.warns(DirtyTreeWarning):
        with Run("latin") as run:
            pass
    assert b"+s = '\xe9'" in (run.dir / "git_diff.patch").read_bytes()
    subprocess.run(["git", "apply", "--check", "-R", str(run.dir / "git_diff.patch")],
                   cwd=repo, check=True)


def test_untracked_scan_failure_keeps_git_info(repo, monkeypatch):
    real_git = meta_mod._git
    timeouts = []

    def flaky_git(args, cwd, env=None, timeout=15):
        if "ls-files" in args:
            timeouts.append(timeout)
            raise subprocess.TimeoutExpired(["git", *args], timeout)
        return real_git(args, cwd, env, timeout)

    monkeypatch.setattr(meta_mod, "_git", flaky_git)
    (repo / "train.py").write_text("print('changed')\n")
    (repo / "newmod.py").write_text("x = 1\n")

    with pytest.warns(DirtyTreeWarning, match=r"(?s)untracked files unchecked.*could not be checked"):
        with Run("slowscan") as run:
            pass

    assert timeouts == [60]
    info = read_meta(run)
    assert info["available"] is True
    assert info["commit"] == git(repo, "rev-parse", "HEAD")
    assert info["branch"] and info["remote"] == "https://github.com/me/proj.git"
    assert info["dirty"] is True
    assert info["untracked_scan"] == "failed: timed out after 60s"
    assert info["dirty_files"] == {"tracked": ["train.py"], "untracked": []}
    patch = patch_of(run)  # tracked-only patch is kept
    assert "+print('changed')" in patch and "newmod" not in patch


def test_untracked_scan_failure_on_clean_tree_is_dirty(repo, monkeypatch):
    real_git = meta_mod._git

    def flaky_git(args, cwd, env=None, timeout=15):
        if "ls-files" in args:
            raise subprocess.TimeoutExpired(["git", *args], timeout)
        return real_git(args, cwd, env, timeout)

    monkeypatch.setattr(meta_mod, "_git", flaky_git)
    with pytest.warns(DirtyTreeWarning, match="could not be checked") as record:
        with Run("slowscan-clean") as run:
            pass
    assert "saved to" not in str(record[0].message)
    info = read_meta(run)
    assert info["dirty"] is True and info["untracked_scan"].startswith("failed")
    assert not (run.dir / "git_diff.patch").exists()


def write_big(repo, name="big.py"):
    big = ("x = 1\n" * 200_000).encode()  # 1.2 MB
    (repo / name).write_bytes(big)
    return big


def test_hash_only_change_writes_no_patch(repo):
    big = write_big(repo)
    with pytest.warns(DirtyTreeWarning) as record:
        with Run("hashonly") as run:
            pass

    message = str(record[0].message)
    assert "recorded by sha256 in meta.json: big.py." in message
    assert "git_diff.patch" not in message
    assert not (run.dir / "git_diff.patch").exists()
    info = read_meta(run)
    assert info["dirty"] is True
    assert "diff_file" not in info
    assert info["dirty_files"] == {"tracked": [], "untracked": ["big.py"]}
    assert info["untracked_too_large"] == [
        {"path": "big.py", "size": len(big), "sha256": hashlib.sha256(big).hexdigest()}
    ]


def test_hashed_file_plus_tracked_change_patch_applies(repo, tmp_path_factory):
    write_big(repo)
    (repo / "train.py").write_text("print('changed')\n")
    with pytest.warns(DirtyTreeWarning, match="recorded by sha256 in meta.json: big.py") as record:
        with Run("hashplus") as run:
            pass
    assert "git_diff.patch" in str(record[0].message)
    assert read_meta(run)["diff_file"] == "git_diff.patch"

    clone = tmp_path_factory.mktemp("clone")
    git(clone, "clone", "-q", str(repo), ".")
    git(clone, "apply", str(run.dir / "git_diff.patch"))
    assert (clone / "train.py").read_text() == "print('changed')\n"
    assert not (clone / "big.py").exists()


def test_dirty_warning_line_layout(repo, monkeypatch):
    (repo / "train.py").write_text("print('changed')\n")
    with pytest.warns(DirtyTreeWarning) as record:
        with Run("layout"):
            pass
    lines = str(record[0].message).splitlines()
    assert lines[1] == lines[-1] == "!" * 72
    assert lines[2] == "exptrail: WORKING TREE IS DIRTY (train.py)"
    assert lines[3].startswith("Run ")

    real_git = meta_mod._git

    def flaky_git(args, cwd, env=None, timeout=15):
        if "ls-files" in args:
            raise subprocess.TimeoutExpired(["git", *args], timeout)
        return real_git(args, cwd, env, timeout)

    monkeypatch.setattr(meta_mod, "_git", flaky_git)
    with pytest.warns(DirtyTreeWarning) as record:
        with Run("layout-failed"):
            pass
    lines = str(record[0].message).splitlines()
    assert lines[2] == "exptrail: WORKING TREE IS DIRTY (train.py, untracked files unchecked)"
    assert lines[3] == "Untracked files could not be checked (timed out after 60s), so"
    assert lines[5].startswith("commit ") and lines[5].endswith(".")
    assert lines[6].startswith("Tracked changes were saved to ")
