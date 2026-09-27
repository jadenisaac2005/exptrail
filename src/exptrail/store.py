"""Read runs back from disk."""

from __future__ import annotations

import csv
import glob
import json
from dataclasses import dataclass
from pathlib import Path

from .liveness import Liveness, assess


@dataclass
class SavedRun:
    path: Path

    def read_json(self, name: str) -> dict:
        """Strict read: raises if the file is missing, corrupt or not an object."""
        data = json.loads((self.path / name).read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{name} is not a JSON object")
        return data

    def _json(self, name: str) -> dict:
        """Tolerant read for display commands."""
        try:
            return self.read_json(name)
        except (OSError, ValueError):
            return {}

    @property
    def id(self) -> str:
        return self.path.name

    @property
    def config(self) -> dict:
        return self._json("config.json")

    @property
    def summary(self) -> dict:
        return self._json("summary.json")

    @property
    def meta(self) -> dict:
        return self._json("meta.json")

    @property
    def name(self) -> str:
        return self.meta.get("name", self.id)

    @property
    def stored_status(self) -> str:
        """The status as written in meta.json."""
        return self.meta.get("status", "unknown")

    @property
    def liveness(self) -> Liveness | None:
        """For a stored ``running`` status, whether it really is. Re-assessed on every
        access (never cached), so a long-lived SavedRun notices the job dying. Reads only."""
        return assess(self.path, self.meta) if self.stored_status == "running" else None

    @property
    def status(self) -> str:
        """Status worked out at read time: a ``running`` run may be ``dead`` or ``stale``."""
        live = self.liveness
        return live.status if live else self.stored_status

    def status_display(self) -> str:
        """Status with its evidence, e.g. ``stale (last heartbeat 14m ago)``."""
        live = self.liveness
        if live:
            return live.display()
        marked = self.meta.get("marked")
        if isinstance(marked, dict) and marked.get("by") == "hand":
            return f"{self.stored_status} (marked by hand)"
        return self.stored_status

    @property
    def commit(self) -> str | None:
        return self.meta.get("git", {}).get("commit")

    @property
    def dirty(self) -> bool:
        return bool(self.meta.get("git", {}).get("dirty"))

    def short_commit(self) -> str:
        if not self.commit:
            return "no-git"
        return self.commit[:7] + ("*" if self.dirty else "")

    def metrics(self) -> list[dict[str, float]]:
        path = self.path / "metrics.csv"
        if not path.exists():
            return []
        with path.open(newline="") as f:
            return [
                {k: float(v) for k, v in row.items() if v not in ("", None)}
                for row in csv.DictReader(f)
            ]


def is_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "meta.json").exists()


def list_runs(root: Path) -> list[SavedRun]:
    if not root.is_dir():
        return []
    return [SavedRun(p) for p in sorted(root.iterdir()) if is_run_dir(p)]


def find_run(ref: str, root: Path) -> SavedRun:
    """Resolve a run by path, folder name, run name (latest wins) or unique prefix."""
    path = Path(ref)
    if is_run_dir(path):
        return SavedRun(path)
    runs = list_runs(root)  # sorted by timestamp, oldest first
    exact = [r for r in runs if r.id == ref] or [r for r in runs if r.name == ref]
    if exact:
        return exact[-1]
    hits = [r for r in runs if ref in r.id]
    if len(hits) == 1:
        return hits[0]
    if hits:
        names = ", ".join(r.id for r in hits[:5])
        raise LookupError(f"{ref!r} is ambiguous: {names}")
    raise LookupError(f"no run matching {ref!r} in {root}")


def glob_runs(pattern: str, root: Path) -> list[SavedRun]:
    """Runs matching a glob. Patterns without a slash are matched inside ``root``."""
    full = pattern if ("/" in pattern or "\\" in pattern) else str(root / pattern)
    return [SavedRun(Path(p)) for p in sorted(glob.glob(full)) if is_run_dir(Path(p))]
