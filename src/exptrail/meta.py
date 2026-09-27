"""Capture the environment a run executed in: git state, Python, packages, host."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import warnings
from importlib import metadata
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None

TRACKED_PACKAGES = ("numpy", "torch", "scikit-learn")


def _git(
    args: list[str], cwd: Path, env: dict | None = None, timeout: float = 15
) -> subprocess.CompletedProcess:
    # --no-optional-locks: never let a read opportunistically rewrite the user's index.
    # --literal-pathspecs: a file called `data[1].py` is a path, not a glob.
    # surrogateescape keeps non-UTF-8 bytes in a diff intact through str.
    return subprocess.run(
        ["git", "--no-optional-locks", "--literal-pathspecs", *args],
        cwd=cwd, capture_output=True, text=True, encoding="utf-8",
        errors="surrogateescape", timeout=timeout, env=env,
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


DEFAULT_UNTRACKED_CODE = ("*.py", "*.ipynb", "*.pyx", "*.yaml", "*.yml", "*.toml")
MAX_UNTRACKED_BYTES = 1_000_000
# Listing untracked files walks the whole work tree, which can be slow with a big
# un-ignored data folder or a repo on a network drive (Colab + Google Drive).
UNTRACKED_SCAN_TIMEOUT = 60


class ConfigWarning(UserWarning):
    """exptrail settings in pyproject.toml could not be used."""


_warned_no_tomllib = False


def check_patterns(name: str, value) -> list[str]:
    if isinstance(value, str) or not all(isinstance(v, str) for v in value):
        raise TypeError(f"{name} must be a list of glob pattern strings, got {value!r}")
    return list(value)


def _pyproject_settings(root: Path) -> dict:
    """``[tool.exptrail]`` from ``<root>/pyproject.toml`` (empty if absent or unreadable)."""
    global _warned_no_tomllib
    path = root / "pyproject.toml"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    if tomllib is None:  # Python 3.10: no stdlib TOML parser, and we add no dependencies
        if re.search(r"^\s*\[\s*tool\s*\.\s*exptrail\s*[\].]", text, re.M) and not _warned_no_tomllib:
            _warned_no_tomllib = True
            warnings.warn(
                f"exptrail: [tool.exptrail] in {path} is ignored on Python 3.10 "
                "(no tomllib). Pass dirty_ignore=/untracked_code= to Run() instead.",
                ConfigWarning,
                stacklevel=5,  # the user's Run(...) line
            )
        return {}
    try:
        section = tomllib.loads(text).get("tool", {}).get("exptrail", {})
    except tomllib.TOMLDecodeError as exc:
        warnings.warn(f"exptrail: could not parse {path}: {exc}", ConfigWarning, stacklevel=5)
        return {}
    out = {}
    for key in ("dirty_ignore", "untracked_code"):
        if key in section:
            out[key] = check_patterns(f"[tool.exptrail] {key}", section[key])
    return out


def _matches(path: str, patterns: list[str]) -> bool:
    """gitignore-flavoured glob match on a repo-relative POSIX path.

    A pattern without ``/`` (``*.md``) matches the file name in any directory;
    one with ``/`` (``submissions/**``) matches the whole path from the repo root.
    A trailing ``/`` means "everything under this directory".
    """
    name = path.rsplit("/", 1)[-1]
    for pat in patterns:
        pat = pat.lstrip("/")
        if pat.endswith("/"):
            pat += "**"
        if "/" in pat:
            if fnmatch.fnmatchcase(path, pat):
                return True
        elif fnmatch.fnmatchcase(name, pat):
            return True
    return False


def _split_z(out: str) -> list[str]:
    return [p for p in out.split("\0") if p]


def _in_virtualenv(path: str, top: Path, cache: dict) -> bool:
    """True if ``path`` sits inside an (unignored) virtualenv, which is never user code."""
    parts = path.split("/")[:-1]
    for i in range(1, len(parts) + 1):
        d = "/".join(parts[:i])
        if d not in cache:
            cache[d] = (top / d / "pyvenv.cfg").is_file()
        if cache[d]:
            return True
    return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _diff_with_untracked(top: Path, untracked: list[str], timeout: float) -> str:
    """``git diff HEAD`` plus ``untracked`` shown as new files.

    The untracked files are marked intent-to-add in a throwaway copy of the
    index, so the user's real index is never touched.
    """
    if not untracked:
        return _git(["diff", "HEAD", "--binary"], top).stdout
    index = _git(["rev-parse", "--git-path", "index"], top, timeout=timeout).stdout.strip()
    index_path = top / index  # absolute paths survive the join unchanged
    with tempfile.TemporaryDirectory(prefix="exptrail-") as tmp:
        tmp_index = Path(tmp) / "index"
        env = {**os.environ, "GIT_INDEX_FILE": str(tmp_index)}
        if index_path.is_file():
            shutil.copyfile(index_path, tmp_index)
        else:
            _git(["read-tree", "HEAD"], top, env, timeout)
        added = _git(["add", "--intent-to-add", "--", *untracked], top, env, timeout)
        diff = _git(["diff", "HEAD", "--binary"], top, env, timeout).stdout
    if added.returncode != 0:
        reason = " ".join(added.stderr.split())
        diff = f"# exptrail: could not include untracked files in this patch: {reason}\n" + diff
    return diff


def _scan_untracked(
    root: Path, patterns: list[str], skip: list[str], max_bytes: int, timeout: float
) -> tuple[list[str], list[dict]]:
    """Untracked code files: (paths to inline in the patch, [{path, size, sha256}] too big to inline)."""
    out = _git(["ls-files", "--others", "--exclude-standard", "-z"], root, timeout=timeout)
    if out.returncode != 0:
        raise subprocess.CalledProcessError(out.returncode, "git ls-files", stderr=out.stderr)
    venv_cache: dict = {}
    inline, hashed = [], []
    for p in _split_z(out.stdout):
        if (
            not _matches(p, patterns)
            or any(p.startswith(d) for d in skip)
            or _in_virtualenv(p, root, venv_cache)
        ):
            continue
        f = root / p
        if not f.is_file():  # a symlink to a directory, a socket, ...
            continue
        size = f.stat().st_size
        if size > max_bytes:
            hashed.append({"path": p, "size": size, "sha256": _sha256(f)})
        else:
            inline.append(p)
    return inline, hashed


def _describe_failure(exc: BaseException) -> str:
    if isinstance(exc, subprocess.TimeoutExpired):
        return f"timed out after {exc.timeout:g}s"
    if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
        return f"git exited with {exc.returncode}: {' '.join(exc.stderr.split())}"
    return str(exc) or type(exc).__name__


def git_info(
    cwd: Path,
    dirty_ignore: list[str] | None = None,
    untracked_code: list[str] | None = None,
    max_untracked_bytes: int = MAX_UNTRACKED_BYTES,
    exclude: list[Path] = (),
) -> tuple[dict, str | None]:
    """Return (info, diff). ``diff`` is the uncommitted patch, or None if there is none.

    A run is dirty if a tracked file changed (unless it matches ``dirty_ignore``)
    or an untracked, non-gitignored file matches ``untracked_code``. Other
    untracked files (CSVs, logs, the runs directory itself) are ignored. The
    patch holds every tracked change, ignored or not, plus untracked code files
    up to ``max_untracked_bytes`` each; larger ones are recorded by sha256.

    ``dirty_ignore``/``untracked_code`` of None fall back to ``[tool.exptrail]``
    in the repo's pyproject.toml, then to the defaults. Untracked files under
    an ``exclude`` directory (the runs folder, with its saved artifacts) never count.

    If the untracked scan fails (e.g. times out on a huge work tree), the
    commit and tracked-file info are kept, the patch holds tracked changes only,
    ``untracked_scan`` says why, and the run counts as dirty since it can't be
    shown to be clean.
    """
    if dirty_ignore is not None:
        dirty_ignore = check_patterns("dirty_ignore", dirty_ignore)
    if untracked_code is not None:
        untracked_code = check_patterns("untracked_code", untracked_code)
    if shutil.which("git") is None:
        return {"available": False, "reason": "git executable not found"}, None
    try:
        top = _git(["rev-parse", "--show-toplevel"], cwd)
        if top.returncode != 0:
            return {"available": False, "reason": "not inside a git repository"}, None
        root = Path(top.stdout.strip())
        head = _git(["rev-parse", "HEAD"], root)
        if head.returncode != 0:
            return {"available": False, "reason": "repository has no commits"}, None

        settings = _pyproject_settings(root)
        if dirty_ignore is None:
            dirty_ignore = settings.get("dirty_ignore", [])
        if untracked_code is None:
            untracked_code = settings.get("untracked_code", list(DEFAULT_UNTRACKED_CODE))

        tracked = _split_z(_git(["diff", "HEAD", "--name-only", "--no-renames", "-z"], root).stdout)
        skip = []
        for d in exclude:
            try:
                skip.append(Path(d).resolve().relative_to(root.resolve()).as_posix() + "/")
            except ValueError:  # outside the repo
                pass
        # The untracked scan gets its own try: if it fails, keep everything else.
        scan_error = None
        try:
            inline, hashed = _scan_untracked(
                root, untracked_code, skip, max_untracked_bytes, UNTRACKED_SCAN_TIMEOUT
            )
            diff = (
                _diff_with_untracked(root, inline, UNTRACKED_SCAN_TIMEOUT)
                if tracked or inline or hashed else None
            )
        except (OSError, subprocess.SubprocessError) as exc:
            scan_error = _describe_failure(exc)
            inline, hashed = [], []
            diff = _git(["diff", "HEAD", "--binary"], root).stdout if tracked else None
        recorded_untracked = inline + [h["path"] for h in hashed]

        dirty_tracked = [p for p in tracked if not _matches(p, dirty_ignore)]
        dirty_untracked = [p for p in recorded_untracked if not _matches(p, dirty_ignore)]
        ignored = [p for p in tracked + recorded_untracked if _matches(p, dirty_ignore)]

        branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root).stdout.strip()
        remote = _git(["remote", "get-url", "origin"], root)
        info = {
            "available": True,
            "commit": head.stdout.strip(),
            "branch": branch,
            "dirty": bool(dirty_tracked or dirty_untracked or scan_error),
            "root": str(root),
            "remote": _clean_remote(remote.stdout) if remote.returncode == 0 else None,
            "dirty_files": {"tracked": dirty_tracked, "untracked": dirty_untracked},
            "ignored_files": ignored,
            "dirty_ignore": dirty_ignore,
            "untracked_code": untracked_code,
        }
        if scan_error:
            info["untracked_scan"] = f"failed: {scan_error}"
        if tracked or recorded_untracked:
            info["changed_files"] = tracked + recorded_untracked
        if hashed:
            info["untracked_too_large"] = hashed
        if not diff:
            # Nothing git can apply (e.g. the only change is a file recorded by
            # hash): write no patch rather than one that `git apply` rejects.
            return info, None
        if hashed:
            note = [
                "# exptrail: untracked files over the size limit "
                f"({max_untracked_bytes} bytes) are recorded by hash, not included below:"
            ]
            note += [f"#   {h['path']}  size={h['size']}  sha256={h['sha256']}" for h in hashed]
            diff = "\n".join(note) + "\n" + diff
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
