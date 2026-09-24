"""The Run object: one experiment, one folder on disk."""

from __future__ import annotations

import csv
import functools
import inspect
import json
import math
import os
import re
import shutil
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .meta import environment, git_info, redact_home


class DirtyTreeWarning(UserWarning):
    """The working tree has uncommitted changes, so the run can't be tied to a commit."""


class NoGitWarning(UserWarning):
    """No git repository was found, so the run can't be tied to a commit."""


def default_root() -> Path:
    return Path(os.environ.get("EXPTRAIL_ROOT", "runs"))


def to_jsonable(value: Any) -> Any:
    """Best-effort conversion of configs/summaries to JSON (numpy, torch, paths...)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    shape = getattr(value, "shape", None)
    if shape is not None and hasattr(value, "tolist"):
        if len(shape) == 0:  # numpy scalar / 0-d tensor
            return to_jsonable(value.tolist())
        size = math.prod(shape)
        if size <= 32:
            return to_jsonable(value.tolist())
        return f"<{type(value).__name__} shape={tuple(shape)}>"
    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except Exception:
            pass
    return repr(value)


def _to_number(key: str, value: Any) -> float | int:
    if hasattr(value, "item"):  # numpy scalar, 0-d torch tensor
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        raise TypeError(f"metric {key!r} must be numeric, got {type(value).__name__}") from None


def _write_json(path: Path, data: Any) -> None:
    """Atomic write so a crash never leaves a half-written JSON file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=False) + "\n")
    os.replace(tmp, path)


def _now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def _slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    return slug or "run"


class Run:
    """Log one experiment to ``<root>/<timestamp>_<name>/``.

    Use as a context manager::

        with Run("momentum-lr0.01", config={"lr": 0.01}) as run:
            run.log(step=epoch, train_loss=loss)
            run.summary(test_acc=0.98)
    """

    def __init__(
        self,
        name: str,
        config: dict | None = None,
        tags: list[str] | None = None,
        seeds: Any = None,
        root: str | os.PathLike | None = None,
        notes: str | None = None,
        redact_paths: bool = True,
    ):
        self.name = name
        self.config = dict(config or {})
        self.tags = list(tags or [])
        self.seeds = seeds
        self.notes = notes
        self.redact_paths = redact_paths
        self.root = Path(root) if root is not None else default_root()
        self.dir: Path | None = None
        self.meta: dict = {}
        self._summary: dict = {}
        self._columns: list[str] = []
        self._csv_file = None
        self._writer = None
        self._auto_step = 0
        self._t0 = 0.0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "Run":
        if self.dir is not None:
            raise RuntimeError("run already started")
        started = _now()
        self.root = self.root.resolve()  # immune to os.chdir() during the run
        self.root.mkdir(parents=True, exist_ok=True)
        base = f"{started.astimezone(timezone.utc):%Y%m%d-%H%M%S}_{_slug(self.name)}"
        for i in range(1000):
            candidate = self.root / (base if i == 0 else f"{base}-{i}")
            try:
                candidate.mkdir()
                break
            except FileExistsError:
                continue
        else:  # pragma: no cover
            raise RuntimeError(f"could not create a unique run folder for {base}")
        self.dir = candidate
        self._t0 = time.monotonic()

        git, diff = git_info(Path.cwd())
        if self.redact_paths and git.get("root"):
            git["root"] = redact_home(git["root"])
        if diff is not None:
            (self.dir / "git_diff.patch").write_text(diff)
            git["diff_file"] = "git_diff.patch"
        self.meta = {
            "name": self.name,
            "status": "running",
            "tags": self.tags,
            "notes": self.notes,
            "seeds": self.seeds,
            "start_time": started.isoformat(),
            "end_time": None,
            "duration_s": None,
            "git": git,
            **environment(self.redact_paths),
            "exptrail_version": __version__,
        }
        _write_json(self.dir / "config.json", self.config)
        _write_json(self.dir / "meta.json", self.meta)
        _write_json(self.dir / "summary.json", self._summary)
        self._warn_about_git(git)
        return self

    def _warn_about_git(self, git: dict) -> None:
        if not git.get("available"):
            warnings.warn(
                f"exptrail: {git['reason']}; run {self.dir.name} is not tied to a commit.",
                NoGitWarning,
                stacklevel=4,
            )
        elif git["dirty"]:
            files = ", ".join(git.get("changed_files", [])[:5])
            bar = "!" * 72
            warnings.warn(
                f"\n{bar}\n"
                f"exptrail: WORKING TREE IS DIRTY ({files})\n"
                f"Run {self.dir.name} used uncommitted code, so its numbers can't be\n"
                f"reproduced from commit {git['commit'][:10]}. The diff was saved to\n"
                f"{self.dir / 'git_diff.patch'}. Commit first for a reportable result.\n"
                f"{bar}",
                DirtyTreeWarning,
                stacklevel=4,
            )

    def finish(self, status: str = "finished", error: str | None = None) -> None:
        if self.dir is None:
            raise RuntimeError("run was never started")
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        if error is not None:
            (self.dir / "traceback.txt").write_text(error)
            self.meta["error"] = error.strip().splitlines()[-1] if error.strip() else ""
        ended = _now()
        self.meta.update(
            status=status,
            end_time=ended.isoformat(),
            duration_s=round(time.monotonic() - self._t0, 3),
        )
        _write_json(self.dir / "summary.json", self._summary)
        _write_json(self.dir / "meta.json", self.meta)

    def __enter__(self) -> "Run":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.finish("finished")
        else:
            status = "failed" if issubclass(exc_type, Exception) else "interrupted"
            self.finish(status, "".join(traceback.format_exception(exc_type, exc, tb)))
        return False  # never swallow the exception

    # -- logging -----------------------------------------------------------

    def _require_started(self) -> Path:
        if self.dir is None:
            raise RuntimeError("run not started; use `with Run(...) as run:` or run.start()")
        return self.dir

    def log(self, step: int | None = None, **metrics: Any) -> None:
        """Append one row to metrics.csv and flush it to disk immediately."""
        run_dir = self._require_started()
        if step is None:
            step = self._auto_step
        self._auto_step = int(step) + 1
        row = {"step": step}
        row.update({k: _to_number(k, v) for k, v in metrics.items()})

        new_keys = [k for k in row if k not in self._columns]
        path = run_dir / "metrics.csv"
        if new_keys:
            self._expand_columns(path, new_keys)
        self._writer.writerow(row)
        self._csv_file.flush()

    def _expand_columns(self, path: Path, new_keys: list[str]) -> None:
        """(Re)open metrics.csv with a header that includes ``new_keys``.

        Rewrites the file when a metric first appears mid-run, so the CSV
        always has a single consistent header.
        """
        if self._csv_file is not None:
            self._csv_file.close()
        old_rows = []
        if path.exists():
            with path.open(newline="") as f:
                old_rows = list(csv.DictReader(f))
        self._columns += new_keys
        tmp = path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._columns, restval="")
            writer.writeheader()
            writer.writerows(old_rows)
        os.replace(tmp, path)
        self._csv_file = path.open("a", newline="")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self._columns, restval="")

    def summary(self, **values: Any) -> None:
        """Record final numbers (the ones you report). Written to disk immediately."""
        run_dir = self._require_started()
        for k, v in values.items():
            try:
                self._summary[k] = _to_number(k, v)
            except TypeError:
                self._summary[k] = to_jsonable(v)
        _write_json(run_dir / "summary.json", self._summary)

    @property
    def summary_data(self) -> dict:
        """A copy of the summary values recorded so far."""
        return dict(self._summary)

    def save_artifact(self, path: str | os.PathLike, name: str | None = None) -> Path:
        """Copy a file or directory into ``<run>/artifacts/``."""
        run_dir = self._require_started()
        src = Path(path)
        dest_dir = run_dir / "artifacts"
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / (name or src.name)
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest)
        return dest


def track(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    config: dict | None = None,
    tags: list[str] | None = None,
    seeds: Any = None,
    root: str | os.PathLike | None = None,
    redact_paths: bool = True,
):
    """Decorator form of :class:`Run`.

    If ``config`` is omitted, the function's arguments become the config.
    If the function has a ``run`` parameter, the active Run is passed in.
    If it returns a dict, that dict is recorded as the summary.
    """

    def decorate(func: Callable) -> Callable:
        sig = inspect.signature(func)
        wants_run = "run" in sig.parameters

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cfg = config
            if cfg is None:
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                cfg = {k: v for k, v in bound.arguments.items() if k != "run"}
            with Run(name or func.__name__, config=cfg, tags=tags, seeds=seeds, root=root,
                     redact_paths=redact_paths) as run:
                if wants_run:
                    kwargs["run"] = run
                result = func(*args, **kwargs)
                if isinstance(result, dict):
                    run.summary(**result)
                return result

        return wrapper

    return decorate(fn) if fn is not None else decorate
