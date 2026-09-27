"""Is a run whose meta.json says ``running`` really still running?

The usual ways ML jobs die (SIGKILL, the out-of-memory killer, a Colab
disconnect, a laptop losing power) run no cleanup code, so nothing can write
a "died" status. Instead, a run leaves evidence while it is alive:

- at start, who it is: hostname, boot ID, pid and the process start time;
- every ``heartbeat_s`` seconds, a daemon thread touches ``<run>/heartbeat``.

Readers combine the two into a status at read time, without writing anything.
On the same machine the answer is certain (``running`` or ``dead``). From
another machine, or where the process can't be inspected, only the heartbeat's
age is available, so an old heartbeat means ``stale``: probably dead.
"""

from __future__ import annotations

import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

IS_WINDOWS = os.name == "nt"
DEFAULT_HEARTBEAT_S = 30.0
STALE_FACTOR = 3
HEARTBEAT_FILE = "heartbeat"
# kern.boottime (macOS) shifts when the wall clock is stepped; a real reboot
# closer than this to the previous boot is still caught by the pid check.
BOOT_TOLERANCE_S = 300.0


def _positive_seconds(value, what: str) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number of seconds, got {value!r}") from None
    if not (math.isfinite(seconds) and seconds > 0):
        raise ValueError(f"{what} must be a positive number of seconds, got {value!r}")
    return seconds


def _env_seconds(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    return _positive_seconds(raw, f"${name}") if raw else None


def heartbeat_interval(value=None) -> float:
    """``value``, else ``$EXPTRAIL_HEARTBEAT_S``, else 30 seconds."""
    if value is not None:
        return _positive_seconds(value, "heartbeat_s")
    return _env_seconds("EXPTRAIL_HEARTBEAT_S") or DEFAULT_HEARTBEAT_S


def stale_after(interval: float) -> float:
    """``$EXPTRAIL_STALE_S``, else 3x the run's heartbeat interval."""
    return _env_seconds("EXPTRAIL_STALE_S") or STALE_FACTOR * interval


# -- identity of this machine and of a process ------------------------------

def _run(args: list[str]) -> str:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=5,
                             env={**os.environ, "LC_ALL": "C"})
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout if out.returncode == 0 else ""


def _linux_stat(pid: int) -> list[str] | None:
    """Fields 3.. of /proc/<pid>/stat (the command name may contain spaces)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return stat[stat.rindex(")") + 2:].split()


def process_start_time(pid: int) -> str | None:
    """An opaque token that changes if ``pid`` is reused by another process."""
    if sys.platform.startswith("linux"):
        fields = _linux_stat(pid)
        # field 22: start time in clock ticks since boot
        return fields[19] if fields and len(fields) > 19 else None
    if sys.platform == "darwin":
        return " ".join(_run(["ps", "-o", "lstart=", "-p", str(pid)]).split()) or None
    return None


def _is_zombie(pid: int) -> bool:
    if sys.platform.startswith("linux"):
        fields = _linux_stat(pid)
        return bool(fields) and fields[0] == "Z"
    return False


def boot_id() -> str | None:
    """Changes on every reboot, so a recorded pid can't be mistaken for a new one."""
    if sys.platform.startswith("linux"):
        try:
            return Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
        except OSError:
            return None
    if sys.platform == "darwin":
        m = re.search(r"sec\s*=\s*(\d+),\s*usec\s*=\s*(\d+)", _run(["sysctl", "-n", "kern.boottime"]))
        return f"boottime:{m.group(1)}.{int(m.group(2)):06d}" if m else None
    return None


def _same_boot(a: str, b: str) -> bool:
    if a.startswith("boottime:") and b.startswith("boottime:"):
        try:
            return abs(float(a[9:]) - float(b[9:])) < BOOT_TOLERANCE_S
        except ValueError:
            pass
    return a == b


def pid_alive(pid: int) -> bool | None:
    """True/False, or None when it can't be checked."""
    if IS_WINDOWS:
        return None  # never os.kill(pid, 0) on Windows: signal 0 is CTRL_C_EVENT there
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None  # os.kill(0 or -1, ...) would address a whole process group
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return None
    return True


def record(heartbeat_s: float) -> dict:
    """What a starting run writes into meta.json under ``liveness``."""
    pid = os.getpid()
    return {
        "hostname": socket.gethostname(),
        "pid": pid,
        "proc_start": process_start_time(pid),
        "boot_id": boot_id(),
        "heartbeat_s": heartbeat_s,
    }


# -- heartbeat --------------------------------------------------------------

class Heartbeat:
    """Daemon thread that touches ``path`` every ``interval`` seconds.

    Only the file's mtime is used, so it never races with meta.json writes,
    and it beats whether or not the job calls log().
    """

    def __init__(self, path: Path, interval: float):
        self.path = path
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.path.touch()
        self._thread = threading.Thread(target=self._beat, name="exptrail-heartbeat", daemon=True)
        self._thread.start()

    def _beat(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                os.utime(self.path)
            except OSError:
                pass  # folder moved or disk hiccup: keep trying, never crash the job

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            # os.utime can block for a long time on a network or FUSE mount (Google
            # Drive on Colab). Never hang finish() on it: the thread is a daemon, so
            # abandoning it is safe, and it exits once the pending utime returns.
            self._thread.join(timeout=min(self.interval, 5.0))
        self._thread = None


# -- read-time status -------------------------------------------------------

@dataclass(frozen=True)
class Liveness:
    status: str  # running | dead | stale
    reason: str = ""
    certain: bool = False

    def display(self) -> str:
        return f"{self.status} ({self.reason})" if self.reason else self.status


def _ago(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f}m"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.0f}h"
    return f"{seconds / 86400:.0f}d"


def _check_local(rec: dict) -> Liveness | None:
    """Certain answer when the run was on this machine, this boot; else None."""
    if not rec.get("hostname") or rec["hostname"] != socket.gethostname():
        return None
    recorded, current = rec.get("boot_id"), boot_id()
    if not isinstance(recorded, str) or not recorded or not current:
        return None
    if not _same_boot(recorded, current):
        return Liveness("dead", "machine rebooted", True)
    pid = rec.get("pid")
    alive = pid_alive(pid)
    if alive is None:
        return None
    if not alive or _is_zombie(pid):
        return Liveness("dead", "pid gone", True)
    started, now_started = rec.get("proc_start"), process_start_time(pid)
    if not started or not now_started:
        return None
    if str(started) != now_started:
        return Liveness("dead", "pid reused", True)
    return Liveness("running", "", True)


def _last_sign_of_life(run_dir: Path) -> tuple[float, str] | None:
    try:
        return (run_dir / HEARTBEAT_FILE).stat().st_mtime, "heartbeat"
    except OSError:
        pass
    # runs from before heartbeats existed: the newest file they wrote
    times = []
    for name in ("meta.json", "metrics.csv", "summary.json"):
        try:
            times.append((run_dir / name).stat().st_mtime)
        except OSError:
            pass
    return (max(times), "activity") if times else None


def assess(run_dir: Path, meta: dict, now: float | None = None) -> Liveness:
    """Status of a run whose stored status is ``running``. Reads only."""
    rec = meta.get("liveness")
    rec = rec if isinstance(rec, dict) else {}
    if not IS_WINDOWS:
        local = _check_local(rec)
        if local is not None:
            return local
    try:
        interval = _positive_seconds(rec.get("heartbeat_s"), "heartbeat_s")
    except ValueError:
        interval = DEFAULT_HEARTBEAT_S
    host = rec.get("hostname") or meta.get("hostname")
    where = f"on {host}, " if host and host != socket.gethostname() else ""
    seen = _last_sign_of_life(run_dir)
    if seen is None:
        return Liveness("stale", f"{where}no heartbeat")
    last, what = seen
    age = max(0.0, (time.time() if now is None else now) - last)
    status = "stale" if age > stale_after(interval) else "running"
    return Liveness(status, f"{where}last {what} {_ago(age)} ago")
