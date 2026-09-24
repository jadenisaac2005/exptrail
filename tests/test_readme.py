import json

import pytest

from exptrail import Run
from exptrail.cli import main
from exptrail.readme import END, START, cell_matches, replace_block


@pytest.fixture
def three_runs(repo, quiet):
    runs = []
    for name, lr, acc in [("sgd", 0.1, 0.9512345), ("momentum", 0.05, 0.9807), ("adam", 0.001, 0.97)]:
        with Run(name, config={"lr": lr, "opt": name}) as run:
            run.log(step=0, val_acc=acc - 0.1)
            run.summary(test_acc=acc, epochs=10)
        runs.append(run)
    return runs


def table(*extra):
    return main(["table", "--metric", "test_acc", "--readme", "README.md", *extra])


def test_table_inserts_block_with_links(three_runs, repo):
    (repo / "README.md").write_text("# Project\n\nIntro.\n")
    assert table("--config", "lr") == 0
    text = (repo / "README.md").read_text()
    assert text.startswith("# Project\n\nIntro.\n\n" + START)
    assert text.count(START) == 1 and text.count(END) == 1
    run = three_runs[1]
    commit = json.loads((run.dir / "meta.json").read_text())["git"]["commit"]
    row = f"| [momentum](runs/{run.dir.name}/) | 0.05 | 0.9807 | [`{commit[:7]}`](https://github.com/me/proj/commit/{commit}) |"
    assert row in text
    assert "| run | lr | test_acc | commit |" in text
    assert "0.9512 |" in text  # precision 4


def test_table_is_idempotent_and_preserves_surroundings(three_runs, repo):
    readme = repo / "README.md"
    readme.write_text(f"# P\n\nBefore.\n\n{START}\nold junk\n{END}\n\nAfter.\n")
    assert table() == 0
    first = readme.read_text()
    assert "old junk" not in first
    assert first.startswith("# P\n\nBefore.\n\n") and first.endswith(f"{END}\n\nAfter.\n")
    assert table() == 0
    assert readme.read_text() == first  # byte-identical on re-run
    assert first.count(START) == 1


def test_table_replaces_rows_when_runs_change(three_runs, repo):
    table("--runs", "*_sgd")
    text = (repo / "README.md").read_text()
    assert "[sgd]" in text and "[adam]" not in text
    table("--runs", "*_adam")
    text = (repo / "README.md").read_text()
    assert "[adam]" in text and "[sgd]" not in text and text.count(START) == 1


def test_table_path_glob_and_failed_runs_excluded(three_runs, repo, quiet):
    with pytest.raises(RuntimeError):
        with Run("broken") as run:
            run.summary(test_acc=0.99)
            raise RuntimeError
    table("--runs", "runs/*")
    text = (repo / "README.md").read_text()
    assert "[broken]" not in text and "[sgd]" in text
    table("--runs", "runs/*", "--include-failed")
    assert "[broken]" in (repo / "README.md").read_text()


def test_format_value_small_floats():
    from exptrail.readme import format_value

    assert format_value(1.234e-6, 4) == "1.234e-06"
    assert format_value(0.0, 4) == "0.0000"
    assert format_value(1e-5, None) == "1e-05"


def test_table_rejects_multiple_blocks():
    with pytest.raises(ValueError):
        replace_block(f"{START}\n{END}\n{START}\n{END}\n", "x")


def test_table_no_matching_runs(repo):
    assert table("--runs", "nothing*") == 1


def test_verify_passes_on_generated_table(three_runs, repo, capsys):
    table("--config", "lr", "--metric", "epochs")
    assert main(["verify", "README.md"]) == 0
    assert "OK: README.md — 3 rows, 9 values checked" in capsys.readouterr().out


@pytest.mark.parametrize(
    "old,new",
    [
        ("| 0.9807 |", "| 0.9817 |"),  # tampered metric
        ("| 0.05 | 0.9807", "| 0.01 | 0.9807"),  # tampered config value
        ("| 0.05 | 0.9807", "| 0.1 | 0.9807"),  # config values must match exactly
        ("| 10 | [`", "| 12 | [`"),  # tampered integer metric
    ],
)
def test_verify_catches_tampered_number(three_runs, repo, capsys, old, new):
    table("--config", "lr", "--metric", "epochs")
    readme = repo / "README.md"
    text = readme.read_text()
    assert old in text
    readme.write_text(text.replace(old, new, 1))
    assert main(["verify", "README.md"]) == 1
    assert "MISMATCH" in capsys.readouterr().err


def test_verify_catches_edited_summary(three_runs, repo, capsys):
    table()
    summary = three_runs[0].dir / "summary.json"
    summary.write_text(json.dumps({"test_acc": 0.5, "epochs": 10}))
    assert main(["verify", "README.md"]) == 1
    assert "test_acc shows 0.9512 but summary.json has 0.5" in capsys.readouterr().err


def test_verify_catches_wrong_commit_and_missing_run(three_runs, repo, capsys):
    table()
    readme = repo / "README.md"
    commit = json.loads((three_runs[0].dir / "meta.json").read_text())["git"]["commit"]
    readme.write_text(readme.read_text().replace(f"`{commit[:7]}`", "`deadbee`"))
    assert main(["verify", "README.md"]) == 1
    table()
    readme.write_text(readme.read_text().replace(three_runs[2].dir.name, "20990101-000000_adam"))
    assert main(["verify", "README.md"]) == 1
    err = capsys.readouterr().err
    assert "commit shows" in err and "not found" in err


def test_verify_catches_added_row(three_runs, repo):
    table()
    readme = repo / "README.md"
    readme.write_text(readme.read_text().replace(END, "| [mine](runs/fake/) | 0.9999 | `abc1234` |\n" + END))
    assert main(["verify", "README.md"]) == 1


def test_verify_fails_without_block(workdir):
    (workdir / "README.md").write_text("# nothing\n")
    assert main(["verify", "README.md"]) == 1
    assert main(["verify", "missing.md"]) == 2


def test_verify_strict_fails_on_dirty_runs(repo, quiet):
    (repo / "train.py").write_text("changed\n")
    with Run("dirty") as run:
        run.summary(test_acc=0.5)
    table()
    text = (repo / "README.md").read_text()
    assert "(dirty)" in text
    assert main(["verify", "README.md"]) == 0
    assert main(["verify", "README.md", "--strict"]) == 1


def test_no_git_rows(workdir, quiet):
    with Run("nogit") as run:
        run.summary(test_acc=0.25)
    table()
    assert "| 0.2500 | no-git |" in (workdir / "README.md").read_text()
    assert main(["verify", "README.md"]) == 0


@pytest.mark.parametrize(
    "cell,value,ok",
    [
        ("0.9807", 0.98071, True),
        ("0.98", 0.98071, True),  # coarser, still a faithful rounding
        ("0.9808", 0.98071, False),
        ("1.23e-05", 1.2345e-5, True),
        ("1.24e-05", 1.2345e-5, False),
        ("3", 3, True),
        ("3", 3.4, True),
        ("4", 3.4, False),
        ("—", None, True),
        ("0.5", None, False),
        ("1.00e-05", 1e-5, True),
        ("adam", "adam", True),
        ("sgd", "adam", False),
    ],
)
def test_cell_matches(cell, value, ok):
    assert cell_matches(cell, value) is ok
