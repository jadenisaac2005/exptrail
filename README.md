# exptrail

Local, framework-agnostic experiment logging so that **every number in your README can be traced to a saved run and re-checked**.

README results tables drift: configs get overwritten, loss curves aren't saved, and nobody remembers which commit produced 0.9807. `exptrail` writes each run to its own folder (config, per-step metrics, final numbers, git commit + diff, environment), generates the README table *from those folders*, and `exptrail verify` fails CI if the table and the runs disagree.

- Standard library only (Python 3.10+). `matplotlib` optional, for `plot`.
- Works with NumPy-from-scratch, PyTorch, sklearn, anything: you just pass numbers.
- Works without git (e.g. Colab): recorded as `no-git` with a warning, never an error.

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
| `meta.json` | status (`running`/`finished`/`failed`/`interrupted`), git commit/branch/remote/dirty flag (and which files set it), Python + numpy/torch/scikit-learn versions, hostname, argv, cwd, seeds, start/end time |
| `git_diff.patch` | uncommitted changes (tracked edits plus new untracked code files), only if there were any |
| `traceback.txt` | only if the run raised |
| `artifacts/` | anything passed to `save_artifact()` |

Paths in `meta.json` (`cwd`, `python_executable`, `argv`, `git.root`) have your home directory replaced with `~`, so committed runs don't leak your username. Pass `Run(..., redact_paths=False)` to keep them verbatim. Credentials in the git remote URL are always stripped.

A dirty working tree triggers a loud `DirtyTreeWarning`: that run's numbers can't be reproduced from a commit, so commit first if you plan to report them.

### What counts as dirty

- **Modified tracked files** make a run dirty, unless they match `dirty_ignore`.
- **Untracked files** make a run dirty only if they look like code, i.e. match `untracked_code`. The default patterns are `*.py`, `*.ipynb`, `*.pyx`, `*.yaml`, `*.yml` and `*.toml`. Other untracked files (CSVs, logs, model outputs, the `runs/` folder) are ignored entirely. Files your `.gitignore` excludes never count, and neither does anything inside a virtualenv (a folder containing `pyvenv.cfg`).
- **`dirty_ignore`** lists glob patterns for changes that shouldn't mark a run dirty, such as a results log you edit by hand. Those changes don't set the flag, don't warn, and don't fail `verify --strict`, but they are still written to `git_diff.patch`, so the record stays complete.

```python
Run("momentum-lr0.01", dirty_ignore=["*.md", "submissions/**"],
    untracked_code=["*.py", "*.yaml"])      # replaces the default code patterns
```

A pattern without `/` (`*.md`) matches that file name in any directory. A pattern with `/` (`submissions/**`) matches the path from the repo root, and a trailing `/` (`submissions/`) means everything under that folder.

`git_diff.patch` includes untracked code files as new-file diffs, so `git apply git_diff.patch` on the recorded commit rebuilds the tree the run used. Your git index is never touched: the files are added to a temporary copy of it. Untracked files over 1 MB are recorded by path, size and sha256 in `meta.json` (and in the patch header, when there is a patch) instead of being inlined. If such a file is the only change, no `git_diff.patch` is written, since there would be nothing for `git apply` to apply.

`meta.json` records what was found under `git`: `dirty_files` (`tracked` and `untracked` lists of the files that made the run dirty), `ignored_files` (changed files that matched `dirty_ignore`), `changed_files` (every changed file that was recorded, in the patch or by hash), `untracked_too_large` (hashed files, if any), and the `dirty_ignore`/`untracked_code` patterns that were in effect.

If listing untracked files fails (it times out after 60 s, e.g. on a huge un-ignored data folder or a repo on a network drive), the commit, branch and tracked changes are still recorded, `untracked_scan` says why the scan failed, and the run is treated as dirty because it can't be shown to be clean.

On Python 3.11+ you can set the same keys once for the whole repo in `pyproject.toml`. Keyword arguments override it:

```toml
[tool.exptrail]
dirty_ignore = ["*.md", "submissions/**"]
untracked_code = ["*.py", "*.ipynb", "*.yaml"]
```

Python 3.10 has no built-in TOML parser and exptrail has no dependencies, so on 3.10 a `[tool.exptrail]` section is ignored with a one-time `ConfigWarning`. Pass the keyword arguments instead.

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

**`table`** writes a markdown table between `<!-- results:start -->` and `<!-- results:end -->` marker lines (appended if the markers aren't there yet). Markers only count when each is on a line of its own, and there must be exactly one pair. Each row links to the run folder and commit. Re-running it replaces the block; it never duplicates it. Add columns with repeatable `--metric` and `--config` flags. Floats are shown to `--precision` decimals (default 4). Only finished runs are included unless you pass `--include-failed`. Links to run folders are relative to the README; pass `--absolute-links` to link them on GitHub/GitLab instead (needed for a README shown on PyPI, which can't resolve relative links).

**`verify`** re-reads every run linked from the block. It fails if a displayed metric doesn't match `summary.json` at the displayed precision, if a config value doesn't exactly match `config.json`, if the commit differs from `meta.json`, or if a linked run folder is missing. Runs that were dirty, failed or git-less produce warnings, and `--strict` turns them into failures. Put it in CI:

```yaml
- run: pip install exptrail && exptrail verify README.md
```

For links in the table to work on GitHub, commit the run folders you report.

## Example

[`examples/digits`](https://github.com/jadenisaac2005/exptrail/tree/HEAD/examples/digits) trains a NumPy MLP on sklearn's digits with 3 optimiser configs and regenerates its README table. Its results, generated with `exptrail --root examples/digits/runs table --metric test_acc --config lr --config momentum --absolute-links --readme README.md`:

<!-- results:start -->
<!-- Generated by exptrail. Do not edit by hand; `exptrail verify` checks these numbers. -->
<!-- command: exptrail --root examples/digits/runs table --metric test_acc --config lr --config momentum --absolute-links --readme README.md -->
| run | lr | momentum | test_acc | commit |
|---|---|---|---|---|
| [momentum-lr0.01](https://github.com/jadenisaac2005/exptrail/tree/HEAD/examples/digits/runs/20260924-055909_momentum-lr0.01/) | 0.01 | 0.9 | 0.9611 | [`42124f2`](https://github.com/jadenisaac2005/exptrail/commit/42124f208a0411b96a60bc720978ac602389a53e) |
| [sgd-lr0.1](https://github.com/jadenisaac2005/exptrail/tree/HEAD/examples/digits/runs/20260924-055909_sgd-lr0.1/) | 0.1 | 0.0 | 0.9611 | [`42124f2`](https://github.com/jadenisaac2005/exptrail/commit/42124f208a0411b96a60bc720978ac602389a53e) |
| [momentum-lr0.05](https://github.com/jadenisaac2005/exptrail/tree/HEAD/examples/digits/runs/20260924-055910_momentum-lr0.05/) | 0.05 | 0.9 | 0.9694 | [`42124f2`](https://github.com/jadenisaac2005/exptrail/commit/42124f208a0411b96a60bc720978ac602389a53e) |
<!-- results:end -->

## License

MIT
