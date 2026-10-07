"""Which summary metrics `exptrail ls` shows: --metric > the run's primary > the first 3."""

import json

import pytest

from exptrail import Run, track
from exptrail.cli import main


def ls_line(capsys, *args):
    assert main(["ls", *args]) == 0
    lines = capsys.readouterr().out.splitlines()
    return {line.split()[0].split("_", 1)[1]: line for line in lines[2:] if line.strip()}


@pytest.fixture
def runs(workdir, quiet):
    with Run("plain") as run:
        run.summary(loss=0.3, acc=0.8, f1=0.7, lb=0.6)
    with Run("headline", primary="lb") as run:
        run.summary(loss=0.3, acc=0.8, f1=0.7, lb=0.6)
    with Run("unlogged", primary="lb") as run:
        run.summary(loss=0.3)


def test_default_order_unchanged_without_primary(runs, capsys):
    line = ls_line(capsys)["plain"]
    assert "loss=0.3, acc=0.8, f1=0.7" in line and "lb" not in line


def test_primary_shown_first_then_old_order(runs, capsys):
    lines = ls_line(capsys)
    assert "lb=0.6, loss=0.3, acc=0.8" in lines["headline"] and "f1" not in lines["headline"]
    assert "lb=—, loss=0.3" in lines["unlogged"]


@pytest.mark.parametrize("flag", ["--metric", "-m"])
def test_metric_flag_beats_primary(runs, capsys, flag):
    lines = ls_line(capsys, flag, "f1", flag, "acc")
    for name in ("plain", "headline"):
        assert "f1=0.7, acc=0.8" in lines[name] and "lb" not in lines[name] and "loss" not in lines[name]
    assert "f1=—, acc=—" in lines["unlogged"]


def test_primary_stored_in_meta(workdir, quiet):
    with Run("p", primary="lb") as run:
        pass
    assert json.loads((run.dir / "meta.json").read_text())["primary_metric"] == "lb"
    with Run("q") as run:
        pass
    assert "primary_metric" not in json.loads((run.dir / "meta.json").read_text())


def test_track_primary(workdir, quiet, capsys):
    @track(primary="b")
    def train():
        return {"a": 1, "b": 2}

    train()
    assert "b=2, a=1" in ls_line(capsys)["train"]


def test_primary_must_be_str():
    with pytest.raises(TypeError, match="primary"):
        Run("bad", primary=["lb"])
