"""Run folders written by exptrail 0.1.0 (none of the 0.1.1 meta keys) still read fine."""

import hashlib
import shutil
from pathlib import Path

from exptrail.cli import main
from exptrail.store import SavedRun

EXAMPLE_RUNS = Path(__file__).parent.parent / "examples" / "digits" / "runs"


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_010_runs_load_in_ls_show_compare(workdir, capsys):
    root = workdir / "runs"
    shutil.copytree(EXAMPLE_RUNS, root)
    runs = [SavedRun(p) for p in sorted(root.iterdir())]
    assert len(runs) == 3
    for run in runs:
        assert {"annotations", "primary_metric", "parent"}.isdisjoint(run.meta)
        assert run.annotations == {} and run.primary_metric is None
        assert run.parent is None and run.parent_run() is None
    before = snapshot(root)

    assert main(["ls"]) == 0
    out = capsys.readouterr().out
    assert all(r.id in out for r in runs) and "test_acc=0.9611" in out
    for run in runs:
        assert main(["show", run.id]) == 0
        out = capsys.readouterr().out
        assert "parent:" not in out and "added later" not in out
    assert main(["compare", *(r.id for r in runs)]) == 0
    out = capsys.readouterr().out
    assert "## provenance" in out and "parent" not in out

    assert snapshot(root) == before
