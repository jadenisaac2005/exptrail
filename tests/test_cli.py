import pytest

from exptrail import Run
from exptrail.cli import main
from exptrail.store import find_run


@pytest.fixture
def two_runs(workdir, quiet):
    out = []
    for name, lr in [("a", 0.1), ("b", 0.01)]:
        with Run(name, config={"lr": lr, "epochs": 5}) as run:
            for step in range(5):
                run.log(step=step, val_acc=step * lr)
            run.summary(test_acc=lr * 5)
        out.append(run)
    return out


def test_ls(two_runs, capsys):
    assert main(["ls"]) == 0
    out = capsys.readouterr().out
    assert two_runs[0].dir.name in out and "finished" in out and "test_acc=0.5" in out


def test_ls_empty(workdir, capsys):
    assert main(["ls"]) == 0
    assert "no runs" in capsys.readouterr().out


def test_show(two_runs, capsys):
    assert main(["show", "a"]) == 0
    out = capsys.readouterr().out
    assert '"lr": 0.1' in out and '"test_acc": 0.5' in out and '"hostname"' in out


def test_compare_shows_only_differing_config(two_runs, capsys):
    assert main(["compare", "a", "b"]) == 0
    out = capsys.readouterr().out
    config_part = out.split("## summary")[0]
    assert "lr" in config_part and "epochs" not in config_part
    assert main(["compare", "a", "b", "--all"]) == 0
    assert "epochs" in capsys.readouterr().out.split("## summary")[0]


def test_find_run_by_prefix_path_and_ambiguity(two_runs, workdir):
    a = two_runs[0]
    root = workdir / "runs"
    assert find_run(a.dir.name, root).path == a.dir
    assert find_run(str(a.dir), root).path == a.dir
    assert find_run("_a", root).path == a.dir
    with pytest.raises(LookupError, match="ambiguous"):
        find_run("2", root)
    assert main(["show", "zzz"]) == 2


def test_plot(two_runs, workdir, capsys):
    pytest.importorskip("matplotlib")
    out = workdir / "curves.png"
    assert main(["plot", "a", "b", "--metric", "val_acc", "-o", str(out)]) == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert main(["plot", "a", "--metric", "nope", "-o", str(out)]) == 1


def test_root_flag(two_runs, workdir, capsys):
    assert main(["--root", str(workdir / "empty"), "ls"]) == 0
    assert "no runs" in capsys.readouterr().out
