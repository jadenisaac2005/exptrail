"""Read runs back from disk."""

from __future__ import annotations

import csv
import glob
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SavedRun:
    path: Path

    def _json(self, name: str) -> dict:
        try:
            return json.loads((self.path / name).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
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
    def status(self) -> str:
        return self.meta.get("status", "unknown")

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
