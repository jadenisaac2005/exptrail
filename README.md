# exptrail

Local, framework-agnostic experiment logging so that **every number in your README can be traced to a saved run and re-checked**.

README results tables drift: configs get overwritten, loss curves aren't saved, and nobody remembers which commit produced 0.9807. `exptrail` writes each run to its own folder (config, per-step metrics, final numbers, git commit + diff, environment), generates the README table *from those folders*, and `exptrail verify` fails CI if the table and the runs disagree.

- Standard library only (Python 3.10+). `matplotlib` optional, for `plot`.
- Works with NumPy-from-scratch, PyTorch, sklearn, anything: you just pass numbers.
- Works without git (e.g. Colab): recorded as `no-git` with a warning, never an error.

> **Name.** Candidates checked against PyPI on 2026-09-24: **`exptrail`** (free, used), `trailrun` (free), `resultlock` (free). `runlog`, `runledger`, `runstamp` and `labbook` were taken.

## Install

```bash
pip install exptrail            # or: pip install "exptrail[plot]"
```

## Log a run

```python
from exptrail import Run

with Run(name="momentum-lr0.01", config={"lr": 0.01, "momentum": 0.9},
         tags=["digits"], seeds={"numpy": 0}) as run:
    for epoch in range(30):
        ...
        run.log(step=epoch, train_loss=loss, val_acc=acc)   # appended + flushed immediately
    run.summary(test_acc=0.9807)                              # the numbers you report
    run.save_artifact("model.npz")                            # copied into the run folder
```

Or as a decorator. The function's arguments become the config, a `run` parameter receives the active run, and a returned dict becomes the summary:

```python
from exptrail import track

@track(name="mlp")
def train(lr=0.01, epochs=30, run=None):
    ...
    return {"test_acc": acc}
```

Each run writes `runs/<UTC timestamp>_<name>/` (override with `root=` or `$EXPTRAIL_ROOT`):

| file | contents |
|---|---|
| `config.json` | the config you passed |
| `metrics.csv` | one row per `log()` call: `step`, then your metrics |
| `summary.json` | final numbers, written on every `summary()` call |
| `meta.json` | status (`running`/`finished`/`failed`/`interrupted`), git commit/branch/remote/dirty flag, Python + numpy/torch/scikit-learn versions, hostname, argv, cwd, seeds, start/end time |
| `git_diff.patch` | uncommitted changes, only if the tree was dirty |
| `traceback.txt` | only if the run raised |
| `artifacts/` | anything passed to `save_artifact()` |

Paths in `meta.json` (`cwd`, `python_executable`, `argv`, `git.root`) have your home directory replaced with `~`, so committed runs don't leak your username. Pass `Run(..., redact_paths=False)` to keep them verbatim. Credentials in the git remote URL are always stripped.

A dirty working tree (modified *tracked* files; untracked files don't count) triggers a loud `DirtyTreeWarning`: that run's numbers can't be reproduced from a commit, so commit first if you plan to report them.

## CLI

```bash
exptrail ls                                   # name, status, key metrics, commit (* = dirty)
exptrail show momentum-lr0.01                 # config + summary + meta
exptrail compare sgd-lr0.1 momentum-lr0.01    # config diff + metric table
exptrail plot sgd-lr0.1 momentum-lr0.01 --metric val_acc -o val_acc.png
exptrail table --metric test_acc --runs '*digits*' --readme README.md
exptrail verify README.md                     # non-zero exit on any mismatch
```

Runs can be referred to by folder name, path, run name (latest wins) or a unique substring. `--root DIR` points any command at another runs directory.

**`table`** writes a markdown table between `<!-- results:start -->` and `<!-- results:end -->` marker lines (appended if the markers aren't there yet). Markers only count when each is on a line of its own, and there must be exactly one pair. Each row links to the run folder and commit. Re-running it replaces the block; it never duplicates it. Add columns with repeatable `--metric` and `--config` flags. Floats are shown to `--precision` decimals (default 4). Only finished runs are included unless you pass `--include-failed`.

**`verify`** re-reads every run linked from the block. It fails if a displayed metric doesn't match `summary.json` at the displayed precision, if a config value doesn't exactly match `config.json`, if the commit differs from `meta.json`, or if a linked run folder is missing. Runs that were dirty, failed or git-less produce warnings, and `--strict` turns them into failures. Put it in CI:

```yaml
- run: pip install exptrail && exptrail verify README.md
```

For links in the table to work on GitHub, commit the run folders you report.

## Example

[`examples/digits`](examples/digits) trains a NumPy MLP on sklearn's digits with 3 optimiser configs and regenerates its README table. Its results, generated with `exptrail --root examples/digits/runs table --metric test_acc --config lr --config momentum --readme README.md`:

<!-- results:start -->
<!-- Generated by exptrail. Do not edit by hand; `exptrail verify` checks these numbers. -->
<!-- command: exptrail --root examples/digits/runs table --metric test_acc --config lr --config momentum --readme README.md -->
| run | lr | momentum | test_acc | commit |
|---|---|---|---|---|
| [momentum-lr0.01](examples/digits/runs/20260924-055315_momentum-lr0.01/) | 0.01 | 0.9 | 0.9611 | [`8f3c779`](https://github.com/jadenisaac2005/Experiment-logger/commit/8f3c779efeb6512c2e2137cfb908bb184778ebfe) |
| [sgd-lr0.1](examples/digits/runs/20260924-055315_sgd-lr0.1/) | 0.1 | 0.0 | 0.9611 | [`8f3c779`](https://github.com/jadenisaac2005/Experiment-logger/commit/8f3c779efeb6512c2e2137cfb908bb184778ebfe) |
| [momentum-lr0.05](examples/digits/runs/20260924-055316_momentum-lr0.05/) | 0.05 | 0.9 | 0.9694 | [`8f3c779`](https://github.com/jadenisaac2005/Experiment-logger/commit/8f3c779efeb6512c2e2137cfb908bb184778ebfe) |
<!-- results:end -->

## License

MIT
