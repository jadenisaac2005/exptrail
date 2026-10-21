"""Linking a run to the run it builds on."""

import json

import pytest

from exptrail import Run, track
from exptrail.cli import main
from exptrail.store import SavedRun


def meta(run):
    return json.loads((run.dir / "meta.json").read_text())


@pytest.fixture
def model(workdir, quiet):
    with Run("model") as run:
        run.summary(val_acc=0.9)
    return run


def test_parent_recorded_and_resolved(model, capsys):
    with Run("predict", parent=model.dir.name) as child:
        pass
    assert meta(child)["parent"] == model.dir.name
    assert SavedRun(child.dir).parent_run().path == model.dir

    assert main(["show", "predict"]) == 0
    out = capsys.readouterr().out
    assert f"\nparent: {model.dir.name}\n" in out and "not found" not in out

    assert main(["compare", "model", "predict"]) == 0
    row = [line for line in capsys.readouterr().out.splitlines() if line.startswith("parent")]
    assert len(row) == 1 and model.dir.name in row[0] and "—" in row[0]


def test_path_is_stored_as_folder_name_only(model):
    with Run("predict", parent=str(model.dir)) as child:
        pass
    assert meta(child)["parent"] == model.dir.name


def test_missing_parent_is_recorded_without_warning(workdir, capsys):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", message="exptrail: not inside a git repository")
        with Run("orphan", parent="20990101-000000_nope") as child:
            pass
    assert meta(child)["parent"] == "20990101-000000_nope"
    assert SavedRun(child.dir).parent_run() is None
    assert main(["show", "orphan"]) == 0
    assert "parent: 20990101-000000_nope (not found in this runs folder)" in capsys.readouterr().out
    assert main(["compare", "orphan"]) == 0
    assert "(not found in this runs folder)" in capsys.readouterr().out


def test_no_parent_no_parent_lines(model, capsys):
    assert "parent" not in meta(model)
    assert main(["show", "model"]) == 0
    assert "parent:" not in capsys.readouterr().out
    assert main(["compare", "model", "model"]) == 0
    assert "parent" not in capsys.readouterr().out


def test_track_parent(model):
    @track(parent=model.dir.name)
    def rescore():
        return {"lb": 0.8}

    rescore()
    child = [p for p in model.dir.parent.iterdir() if p.name.endswith("_rescore")][0]
    assert SavedRun(child).parent == model.dir.name


@pytest.mark.parametrize("bad", [3, ["a"], b"model"])
def test_parent_must_be_str(bad):
    with pytest.raises(TypeError, match="parent"):
        Run("bad", parent=bad)


@pytest.mark.parametrize("bad", ["", ".", ".."])
def test_parent_must_name_a_folder(bad):
    with pytest.raises(ValueError, match="parent"):
        Run("bad", parent=bad)
