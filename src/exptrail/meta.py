"""Capture the environment a run executed in: git state, Python, packages, host."""

from __future__ import annotations

import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from importlib import metadata
from pathlib import Path

TRACKED_PACKAGES = ("numpy", "torch", "scikit-learn")


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=15
    )


def _clean_remote(url: str) -> str:
    """Strip credentials from a remote URL so tokens never land in meta.json.

    Any ``user:password@`` is removed, as is a bare ``token@`` on http(s).
    A plain SSH user (``ssh://git@host``, ``git@host:org/repo``) is kept.
    """
    def strip(m: re.Match) -> str:
        scheme, userinfo = m.group(1), m.group(2)
        if ":" in userinfo or scheme.lower().startswith("http"):
            return scheme
        return m.group(0)

    return re.sub(r"^([a-zA-Z][a-zA-Z0-9+.-]*://)([^@/]+)@", strip, url.strip())


def git_info(cwd: Path) -> tuple[dict, str | None]:
    """Return (info, diff). ``diff`` is the uncommitted patch, or None if clean.

    Only tracked files count towards "dirty": untracked files (including the
    runs directory itself) don't change what the committed code computes.
    """
    if shutil.which("git") is None:
        return {"available": False, "reason": "git executable not found"}, None
    try:
        top = _git(["rev-parse", "--show-toplevel"], cwd)
        if top.returncode != 0:
            return {"available": False, "reason": "not inside a git repository"}, None
        head = _git(["rev-parse", "HEAD"], cwd)
        if head.returncode != 0:
            return {"available": False, "reason": "repository has no commits"}, None
        status = _git(["status", "--porcelain", "--untracked-files=no"], cwd).stdout
        branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd).stdout.strip()
        remote = _git(["remote", "get-url", "origin"], cwd)
        info = {
            "available": True,
            "commit": head.stdout.strip(),
            "branch": branch,
            "dirty": bool(status.strip()),
            "root": top.stdout.strip(),
            "remote": _clean_remote(remote.stdout) if remote.returncode == 0 else None,
        }
        diff = None
        if info["dirty"]:
            diff = _git(["diff", "HEAD", "--binary"], cwd).stdout
            info["changed_files"] = [line[3:] for line in status.splitlines()]
        return info, diff
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": f"git failed: {exc}"}, None


def package_versions(names=TRACKED_PACKAGES) -> dict[str, str]:
    """Versions of installed packages, without importing them (torch is slow)."""
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


def redact_home(text: str, home: str | None = None) -> str:
    """Replace the user's home directory with ``~`` so usernames don't leak."""
    home = (home if home is not None else os.path.expanduser("~")).rstrip("/\\")
    if not home or home == "~":
        return text  # no home, or home is the filesystem root
    # Only at the start of the string or of an option value (--out=/home/me/x),
    # and only as a whole path component (/home/me2 is left alone).
    pattern = r"(?:^|(?<=[=:,\s]))" + re.escape(home) + r"(?=[/\\]|$|[\s,:])"
    return re.sub(pattern, "~", text)


def environment(redact_paths: bool = True) -> dict:
    clean = redact_home if redact_paths else (lambda s: s)
    return {
        "python": sys.version.split()[0],
        "python_executable": clean(sys.executable),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "argv": [clean(a) for a in sys.argv],
        "cwd": clean(str(Path.cwd())),
        "packages": package_versions(),
    }
