import csv
import json
import sys
import warnings

import pytest

from exptrail import DirtyTreeWarning, NoGitWarning, Run, track

from .conftest import git


def read(run, name):
    return json.loads((run.dir / name).read_text())


def rows(run):
    with (run.dir / "metrics.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def test_lifecycle_writes_all_files(repo):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a clean repo must not warn
        with Run("momentum-lr0.01", config={"lr": 0.01}, tags=["a"], seeds={"np": 0}) as run:
            for epoch in range(3):
                run.log(step=epoch, train_loss=1.0 / (epoch + 1), val_acc=0.5 + epoch / 10)
            run.summary(test_acc=0.9807)

    assert run.dir.parent == repo / "runs"
    assert run.dir.name.endswith("_momentum-lr0.01")
    assert read(run, "config.json") == {"lr": 0.01}
    assert read(run, "summary.json") == {"test_acc": 0.9807}
    meta = read(run, "meta.json")
    assert meta["status"] == "finished"
    assert meta["tags"] == ["a"] and meta["seeds"] == {"np": 0}
    assert meta["git"]["commit"] == git(repo, "rev-parse", "HEAD")
    assert meta["git"]["dirty"] is False
    assert meta["git"]["remote"] == "https://github.com/me/proj.git"  # token stripped
    assert meta["end_time"] and meta["duration_s"] >= 0
    assert meta["python"] and meta["hostname"] and isinstance(meta["argv"], list)
    assert not (run.dir / "git_diff.patch").exists()
    data = rows(run)
    assert [r["step"] for r in data] == ["0", "1", "2"]
    assert float(data[2]["val_acc"]) == pytest.approx(0.7)


def test_metrics_flushed_before_run_ends(workdir, quiet):
    with Run("flush") as run:
        run.log(step=0, loss=1.5)
        # read while the run is still open
        assert rows(run) == [{"step": "0", "loss": "1.5"}]
        assert read(run, "meta.json")["status"] == "running"
        run.summary(acc=0.5)
        assert read(run, "summary.json") == {"acc": 0.5}
        assert run.summary_data == {"acc": 0.5}


def test_new_metric_midrun_expands_header(workdir, quiet):
    with Run("expand") as run:
        run.log(loss=1.0)
        run.log(loss=0.5, val_acc=0.8)
        run.log(loss=0.25)
    data = rows(run)
    assert list(data[0]) == ["step", "loss", "val_acc"]
    assert [r["val_acc"] for r in data] == ["", "0.8", ""]
    assert [r["step"] for r in data] == ["0", "1", "2"]  # auto-increment


def test_log_rejects_non_numeric(workdir, quiet):
    with Run("bad") as run:
        with pytest.raises(TypeError):
            run.log(loss="high")


def test_crash_marks_failed_and_saves_traceback(workdir, quiet):
    with pytest.raises(ZeroDivisionError):
        with Run("crash") as run:
            run.log(step=0, loss=2.0)
            run.summary(partial=1)
            1 / 0
    meta = read(run, "meta.json")
    assert meta["status"] == "failed"
    assert "ZeroDivisionError" in meta["error"]
    tb = (run.dir / "traceback.txt").read_text()
    assert "Traceback" in tb and "1 / 0" in tb
    assert rows(run)[0]["loss"] == "2.0"  # logged data survives
    assert read(run, "summary.json") == {"partial": 1}


def test_keyboard_interrupt_marks_interrupted(workdir, quiet):
    with pytest.raises(KeyboardInterrupt):
        with Run("ctrlc") as run:
            raise KeyboardInterrupt
    assert read(run, "meta.json")["status"] == "interrupted"


def test_dirty_tree_warns_and_saves_diff(repo):
    (repo / "train.py").write_text("print('changed')\n")
    with pytest.warns(DirtyTreeWarning, match="DIRTY"):
        with Run("dirty") as run:
            run.summary(acc=1.0)
    meta = read(run, "meta.json")
    assert meta["git"]["dirty"] is True
    assert meta["git"]["changed_files"] == ["train.py"]
    patch = (run.dir / meta["git"]["diff_file"]).read_text()
    assert "+print('changed')" in patch and "-print('hi')" in patch


def test_untracked_files_do_not_make_tree_dirty(repo):
    (repo / "scratch.txt").write_text("x")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with Run("untracked") as run:
            pass
    assert read(run, "meta.json")["git"]["dirty"] is False


def test_no_git_is_recorded_gracefully(workdir, monkeypatch):
    monkeypatch.setenv("PATH", "")  # simulate Colab-without-git
    with pytest.warns(NoGitWarning):
        with Run("nogit") as run:
            run.summary(acc=0.1)
    meta = read(run, "meta.json")
    assert meta["git"] == {"available": False, "reason": "git executable not found"}
    assert meta["status"] == "finished"


def test_package_versions_only_track_frameworks(workdir, quiet):
    with Run("pkgs") as run:
        pass
    packages = read(run, "meta.json")["packages"]
    assert set(packages) <= {"numpy", "torch", "scikit-learn"}


@pytest.fixture
def fake_home(repo, monkeypatch):
    """Make the repo live under $HOME and run with a home-relative argv."""
    home = repo.parent
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(sys, "argv", [str(repo / "train.py"), f"--out={repo}/o.npz", "--lr", "0.1"])
    monkeypatch.setattr(sys, "executable", str(home / ".venv" / "bin" / "python"))
    return home


def test_paths_redacted_by_default(repo, fake_home):
    with Run("redact") as run:
        pass
    meta = read(run, "meta.json")
    rel = repo.name
    assert meta["cwd"] == f"~/{rel}"
    assert meta["python_executable"] == "~/.venv/bin/python"
    assert meta["argv"] == [f"~/{rel}/train.py", f"--out=~/{rel}/o.npz", "--lr", "0.1"]
    assert meta["git"]["root"] == f"~/{rel}"
    assert str(fake_home) not in (run.dir / "meta.json").read_text()


def test_redact_paths_opt_out(repo, fake_home):
    with Run("raw", redact_paths=False) as run:
        pass
    meta = read(run, "meta.json")
    assert meta["cwd"] == str(repo)
    assert meta["python_executable"] == str(fake_home / ".venv" / "bin" / "python")
    assert meta["argv"][0] == str(repo / "train.py")


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://user:tok@github.com/o/r.git", "https://github.com/o/r.git"),
        ("https://tok@github.com/o/r.git", "https://github.com/o/r.git"),
        ("ssh://user:pw@host/o/r.git", "ssh://host/o/r.git"),
        ("ssh://git@github.com/o/r.git", "ssh://git@github.com/o/r.git"),
        ("git@github.com:o/r.git", "git@github.com:o/r.git"),
    ],
)
def test_clean_remote_strips_credentials(url, expected):
    from exptrail.meta import _clean_remote

    assert _clean_remote(url) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("/home/me", "~"),
        ("/home/me/proj/x.py", "~/proj/x.py"),
        ("--out=/home/me/o", "--out=~/o"),
        ("/home/me2/x", "/home/me2/x"),  # different user, same prefix
        ("/mnt/home/me/x", "/mnt/home/me/x"),  # not at the start of a path
        ("--lr", "--lr"),
    ],
)
def test_redact_home(text, expected):
    from exptrail.meta import redact_home

    assert redact_home(text, home="/home/me") == expected
    assert redact_home(text, home="/") == text  # home at filesystem root: leave alone


def test_not_a_repo(workdir):
    with pytest.warns(NoGitWarning, match="not inside a git repository"):
        with Run("norepo") as run:
            pass
    assert read(run, "meta.json")["git"]["available"] is False


def test_same_name_same_second_gets_unique_folder(workdir, quiet):
    with Run("dup") as a:
        pass
    with Run("dup") as b:
        pass
    assert a.dir != b.dir


def test_root_from_env_and_name_slug(workdir, monkeypatch, quiet):
    monkeypatch.setenv("EXPTRAIL_ROOT", str(workdir / "elsewhere"))
    with Run("lr=0.1 / bs 32") as run:
        pass
    assert run.dir.parent == workdir / "elsewhere"
    assert run.dir.name.endswith("_lr-0.1-bs-32")


def test_metrics_csv_exists_without_log_calls(workdir, quiet):
    with Run("nolog") as run:
        assert (run.dir / "metrics.csv").read_text().strip() == "step"
        run.summary(acc=1.0)
    assert rows(run) == []


def test_save_artifact_rejects_escaping_names(workdir, quiet):
    (workdir / "model.npz").write_bytes(b"w")
    with Run("art") as run:
        for bad in ("../meta.json", "/tmp/x", "a/b", ".."):
            with pytest.raises(ValueError):
                run.save_artifact("model.npz", name=bad)
        assert run.save_artifact("model.npz", name="best.npz").name == "best.npz"
    assert read(run, "meta.json")["name"] == "art"


def test_save_artifact_file_and_dir(workdir, quiet):
    (workdir / "model.npz").write_bytes(b"weights")
    (workdir / "ckpt").mkdir()
    (workdir / "ckpt" / "a.bin").write_bytes(b"a")
    with Run("art") as run:
        dest = run.save_artifact("model.npz")
        run.save_artifact(workdir / "ckpt")
    assert dest.read_bytes() == b"weights"
    assert (run.dir / "artifacts" / "ckpt" / "a.bin").exists()


def test_numpy_values_are_serialised(workdir, quiet):
    np = pytest.importorskip("numpy")
    with Run("np", config={"lr": np.float64(0.1), "X": np.zeros((50, 50)), "k": np.arange(3)}) as run:
        run.log(loss=np.float32(0.5))
        run.summary(acc=np.float64(0.75), n=np.int64(3))
    assert read(run, "config.json") == {"lr": 0.1, "X": "<ndarray shape=(50, 50)>", "k": [0, 1, 2]}
    assert read(run, "summary.json") == {"acc": 0.75, "n": 3}


def test_using_run_before_start_errors():
    with pytest.raises(RuntimeError):
        Run("x").log(loss=1)


def test_track_decorator(workdir, quiet):
    @track(tags=["deco"])
    def train(lr=0.1, epochs=2, run=None):
        for e in range(epochs):
            run.log(step=e, loss=lr / (e + 1))
        return {"test_acc": 0.9}

    assert train(lr=0.5) == {"test_acc": 0.9}
    (run_dir,) = (workdir / "runs").iterdir()
    assert run_dir.name.endswith("_train")
    assert json.loads((run_dir / "config.json").read_text()) == {"lr": 0.5, "epochs": 2}
    assert json.loads((run_dir / "summary.json").read_text()) == {"test_acc": 0.9}


def test_track_decorator_with_positional_run(workdir, quiet):
    @track
    def train(lr, run=None):
        run.log(loss=lr)
        return {"acc": 1.0}

    assert train(0.1, None) == {"acc": 1.0}  # run slot filled positionally
    (run_dir,) = (workdir / "runs").iterdir()
    assert json.loads((run_dir / "config.json").read_text()) == {"lr": 0.1}


def test_track_decorator_without_parens_and_failure(workdir, quiet):
    @track
    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        boom()
    (run_dir,) = (workdir / "runs").iterdir()
    assert json.loads((run_dir / "meta.json").read_text())["status"] == "failed"
