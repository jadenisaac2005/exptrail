"""exptrail annotate: summary values added after a run has ended."""

import json
from datetime import datetime

import pytest

from exptrail import Run
from exptrail.cli import main, parse_pair
from exptrail.store import SavedRun

from .test_liveness import fake_run, remote


def read(run_dir, name):
    return json.loads((run_dir / name).read_text())


@pytest.fixture
def done(workdir, quiet):
    with Run("done") as run:
        run.summary(val_acc=0.9)
    return run.dir


def test_annotate_writes_summary_and_provenance(done, capsys):
    assert main(["annotate", "done", "lb=0.913", "rank=12", "ok=true", "note=public lb", "x=null"]) == 0
    assert read(done, "summary.json") == {
        "val_acc": 0.9, "lb": 0.913, "rank": 12, "ok": True, "note": "public lb", "x": None}
    notes = read(done, "meta.json")["annotations"]
    assert set(notes) == {"lb", "rank", "ok", "note", "x"}
    assert notes["lb"]["by"] == "hand" and "previous" not in notes["lb"]
    assert datetime.fromisoformat(notes["lb"]["at"]).utcoffset().total_seconds() == 0
    assert read(done, "meta.json")["status"] == "finished"
    assert "annotated" in capsys.readouterr().out


def test_annotate_refuses_overwrite_without_force(done, capsys):
    before = {n: (done / n).read_bytes() for n in ("summary.json", "meta.json")}
    assert main(["annotate", "done", "val_acc=0.95"]) == 1
    assert "--force" in capsys.readouterr().err
    assert {n: (done / n).read_bytes() for n in before} == before

    assert main(["annotate", "done", "val_acc=0.95", "--force"]) == 0
    assert read(done, "summary.json")["val_acc"] == 0.95
    assert read(done, "meta.json")["annotations"]["val_acc"]["previous"] == 0.9
    assert "was 0.9" in capsys.readouterr().out


def test_annotate_refuses_running_run_without_force(workdir, capsys):
    run_dir = fake_run(workdir / "runs", "live", remote(), heartbeat_age=0)
    before = (run_dir / "summary.json").read_bytes()
    assert main(["annotate", "live", "lb=1"]) == 1
    assert "running" in capsys.readouterr().err
    assert (run_dir / "summary.json").read_bytes() == before
    assert main(["annotate", "live", "lb=1", "--force"]) == 0
    assert read(run_dir, "summary.json") == {"acc": 0.5, "lb": 1}
    assert read(run_dir, "meta.json")["status"] == "running"  # annotate never changes the status


@pytest.mark.parametrize("bad", ["novalue", "=5", " =5"])
def test_annotate_rejects_malformed_pairs(done, capsys, bad):
    before = (done / "summary.json").read_bytes()
    assert main(["annotate", "done", "lb=1", bad]) != 0
    assert "key=value" in capsys.readouterr().err
    assert (done / "summary.json").read_bytes() == before


def test_parse_pair():
    assert parse_pair("a=1.5") == ("a", 1.5)
    assert parse_pair("a=1e-3") == ("a", 0.001)
    assert parse_pair("a=false") == ("a", False)
    assert parse_pair("a=b=c") == ("a", "b=c")
    assert parse_pair("a=") == ("a", "")
    assert parse_pair('a="7"') == ("a", "7")


def test_saved_run_annotate(done):
    run = SavedRun(done)
    assert run.annotate({"lb": 0.8}) == {}
    assert run.summary["lb"] == 0.8 and set(run.annotations) == {"lb"}
    with pytest.raises(ValueError, match="already has lb"):
        run.annotate({"lb": 0.7})
    assert run.annotate({"lb": 0.7}, force=True) == {"lb": 0.8}
    with pytest.raises(ValueError):
        run.annotate({})
    assert not list(done.glob("*.tmp"))


def test_show_marks_annotated_keys(done, capsys):
    SavedRun(done).annotate({"lb": 0.8})
    assert main(["show", "done"]) == 0
    out = capsys.readouterr().out
    summary = out.split("## summary")[1].split("## meta")[0]
    date = read(done, "meta.json")["annotations"]["lb"]["at"][:10]
    assert f"lb: (added later, {date})" in summary
    assert "val_acc:" not in summary.split("}")[1]  # only annotated keys are marked


def test_annotate_failure_between_writes_keeps_provenance(done, monkeypatch):
    from exptrail import store

    real = store._write_json
    calls = []

    def flaky(path, data):
        calls.append(path.name)
        if len(calls) == 2:
            raise OSError("disk full")
        real(path, data)

    monkeypatch.setattr(store, "_write_json", flaky)
    with pytest.raises(OSError):
        SavedRun(done).annotate({"lb": 0.913, "rank": 12})
    assert calls == ["meta.json", "summary.json"]
    added = set(read(done, "summary.json")) - {"val_acc"}
    assert added <= set(read(done, "meta.json").get("annotations", {}))
