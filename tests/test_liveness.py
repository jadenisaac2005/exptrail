"""Run lifecycle (start/finish, SIGTERM, atexit) and read-time liveness."""

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
import warnings
from pathlib import Path

import pytest

from exptrail import NoGitWarning, Run, liveness, track
from exptrail.cli import main
from exptrail.liveness import assess, boot_id, process_start_time
from exptrail.readme import render_table, write_table
from exptrail.store import SavedRun

posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX signals and pids")
needs_identity = pytest.mark.skipif(
    os.name == "nt" or boot_id() is None or process_start_time(os.getpid()) is None,
    reason="boot ID / process start time not available on this platform",
)
OTHER_HOST = "exptrail-test-host-that-does-not-exist"


def meta_of(run_dir):
    return json.loads((Path(run_dir) / "meta.json").read_text())


def fake_run(root, name, liveness_rec, heartbeat_age=None, status="running"):
    """A run folder as a (possibly killed) process on some host would leave it."""
    run_dir = Path(root) / f"20260101-000000_{name}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text("{}")
    (run_dir / "summary.json").write_text('{"acc": 0.5}')
    (run_dir / "meta.json").write_text(json.dumps(
        {"name": name, "status": status, "git": {}, "liveness": liveness_rec}))
    if heartbeat_age is not None:
        hb = run_dir / "heartbeat"
        hb.touch()
        t = time.time() - heartbeat_age
        os.utime(hb, (t, t))
    return run_dir


def here(**overrides):
    """A liveness record for this very process."""
    rec = {"hostname": socket.gethostname(), "pid": os.getpid(),
           "proc_start": process_start_time(os.getpid()), "boot_id": boot_id(), "heartbeat_s": 30}
    rec.update(overrides)
    return rec


def remote(**overrides):
    rec = {"hostname": OTHER_HOST, "pid": 4242, "proc_start": "1", "boot_id": "x", "heartbeat_s": 30}
    rec.update(overrides)
    return rec


def gone_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def heartbeat_threads():
    return [t for t in threading.enumerate() if t.name == "exptrail-heartbeat"]


def spawn(script, cwd):
    return subprocess.Popen([sys.executable, "-c", textwrap.dedent(script)], cwd=cwd,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


# -- part 1: explicit lifecycle ----------------------------------------------

def test_start_finish_without_with(workdir, quiet):
    run = Run("manual", heartbeat_s=0.05).start()
    assert (run.dir / "heartbeat").exists() and heartbeat_threads()
    run.log(loss=1.0)
    run.summary(acc=0.5)
    run.finish()
    meta = meta_of(run.dir)
    assert meta["status"] == "finished" and meta["duration_s"] >= 0
    assert not heartbeat_threads()  # stopped and joined, nothing lingers
    assert not (run.dir / "heartbeat").exists()
    run.finish("failed")  # idempotent: the first call wins
    assert meta_of(run.dir) == meta
    with pytest.raises(RuntimeError, match="already finished"):
        run.log(loss=2.0)


def test_finish_rejects_unknown_status(workdir, quiet):
    run = Run("bad-status").start()
    with pytest.raises(ValueError):
        run.finish("dead")
    run.finish()


def test_finish_inside_with_block_is_kept(workdir, quiet):
    with pytest.raises(RuntimeError):
        with Run("early") as run:
            run.finish("interrupted")
            raise RuntimeError("after finish")
    assert meta_of(run.dir)["status"] == "interrupted"
    assert not (run.dir / "traceback.txt").exists()


def test_warning_as_error_in_start_cleans_up(workdir, default_sigterm):
    """A git warning turned into an error must not leave a half-started run behind."""
    run = Run("strict")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(NoGitWarning):
            run.start()
    assert not heartbeat_threads()
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    meta = meta_of(run.dir)
    assert meta["status"] == "failed" and "NoGitWarning" in meta["error"]
    assert (run.dir / "traceback.txt").exists()
    assert run._finished  # the atexit hook, if it were still registered, does nothing
    assert SavedRun(run.dir).status == "failed"


def test_with_and_track_leave_no_heartbeat_thread(workdir, quiet):
    with Run("w"):
        assert heartbeat_threads()

    @track
    def f():
        assert heartbeat_threads()
        return {"acc": 1.0}

    f()
    assert not heartbeat_threads()


def test_atexit_marks_interrupted(workdir):
    proc = spawn("""
        from exptrail import Run
        run = Run("forgot").start()
        run.log(loss=1.0)
        print(run.dir, flush=True)
    """, workdir)
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, err
    meta = meta_of(out.strip())
    assert meta["status"] == "interrupted"
    assert meta["status_reason"] == "exited without finish()"
    assert main(["ls"]) == 0


def test_atexit_does_nothing_after_finish(workdir):
    proc = spawn("""
        from exptrail import Run
        run = Run("tidy").start()
        run.finish()
        print(run.dir, flush=True)
    """, workdir)
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, err
    assert meta_of(out.strip())["status"] == "finished"


@pytest.fixture
def default_sigterm():
    previous = signal.signal(signal.SIGTERM, signal.SIG_DFL)
    yield
    signal.signal(signal.SIGTERM, previous)


def test_sigterm_handler_installed_and_restored(workdir, quiet, default_sigterm):
    run = Run("term").start()
    assert signal.getsignal(signal.SIGTERM) == run._on_sigterm
    run.finish()
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def test_sigterm_handler_never_overrides_users(workdir, quiet, default_sigterm):
    def mine(signum, frame):
        pass

    signal.signal(signal.SIGTERM, mine)
    run = Run("term").start()
    assert signal.getsignal(signal.SIGTERM) is mine
    run.finish()
    assert signal.getsignal(signal.SIGTERM) is mine


def test_sigterm_handler_not_installed_off_main_thread(workdir, quiet, default_sigterm):
    runs = []
    t = threading.Thread(target=lambda: runs.append(Run("thread").start()))
    t.start()
    t.join()
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    runs[0].finish()


def test_sigterm_handler_left_alone_if_user_replaced_it(workdir, quiet, default_sigterm):
    run = Run("term").start()

    def later(signum, frame):
        pass

    signal.signal(signal.SIGTERM, later)
    run.finish()
    assert signal.getsignal(signal.SIGTERM) is later


@posix_only
def test_sigterm_marks_interrupted_then_dies_by_default(workdir):
    proc = spawn("""
        import time
        from exptrail import Run
        run = Run("term").start()
        print(run.dir, flush=True)
        time.sleep(60)
    """, workdir)
    run_dir = proc.stdout.readline().strip()
    proc.send_signal(signal.SIGTERM)
    proc.communicate(timeout=60)
    assert proc.returncode == -signal.SIGTERM  # default behaviour: killed by the signal
    meta = meta_of(run_dir)
    assert meta["status"] == "interrupted" and meta["status_reason"] == "SIGTERM"


# -- part 2: liveness record and heartbeat ----------------------------------

def test_liveness_record_in_meta(workdir, quiet, monkeypatch):
    monkeypatch.setenv("EXPTRAIL_HEARTBEAT_S", "7")
    with Run("rec") as run:
        rec = meta_of(run.dir)["liveness"]
    assert rec["hostname"] == socket.gethostname()
    assert rec["pid"] == os.getpid()
    assert rec["heartbeat_s"] == 7.0
    if sys.platform.startswith("linux"):
        assert rec["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        assert rec["proc_start"].isdigit()
    assert Run("x", heartbeat_s=2).heartbeat_s == 2.0  # argument beats the environment


@pytest.mark.parametrize("bad", ["0", "-1", "soon", "nan"])
def test_bad_heartbeat_interval_rejected(workdir, monkeypatch, bad):
    monkeypatch.setenv("EXPTRAIL_HEARTBEAT_S", bad)
    with pytest.raises(ValueError, match="EXPTRAIL_HEARTBEAT_S"):
        Run("x")


def test_heartbeat_beats_without_log_calls(workdir, quiet):
    run = Run("silent", heartbeat_s=0.05).start()
    hb = run.dir / "heartbeat"
    first = hb.stat().st_mtime_ns
    deadline = time.monotonic() + 10
    while hb.stat().st_mtime_ns == first and time.monotonic() < deadline:
        time.sleep(0.02)
    assert hb.stat().st_mtime_ns > first
    assert SavedRun(run.dir).status == "running"
    run.finish()


def test_finish_does_not_hang_on_blocked_utime(workdir, quiet, monkeypatch):
    """A heartbeat stuck in os.utime (slow network mount) must not block finish()."""
    run = Run("blocked", heartbeat_s=0.05).start()
    entered, release = threading.Event(), threading.Event()
    real_utime = os.utime

    def slow_utime(path, *args, **kwargs):
        if Path(path).name == "heartbeat":
            entered.set()
            release.wait(30)
        return real_utime(path, *args, **kwargs)

    monkeypatch.setattr(os, "utime", slow_utime)
    try:
        assert entered.wait(10)  # the heartbeat thread is now stuck
        t0 = time.monotonic()
        run.finish()
        assert time.monotonic() - t0 < 2  # joined with a timeout of min(interval, 5)
        assert meta_of(run.dir)["status"] == "finished"
    finally:
        release.set()
    for t in heartbeat_threads():  # the abandoned daemon exits once utime returns
        t.join(10)
    assert not heartbeat_threads()


def test_long_silent_run_stays_running(workdir):
    """80 minutes without a single log() call: the heartbeat alone keeps it running."""
    run_dir = fake_run(workdir / "runs", "silent", remote(heartbeat_s=30), heartbeat_age=0)
    hb = run_dir / "heartbeat"
    t0 = time.time()
    for elapsed in range(0, 80 * 60 + 1, 30):  # the thread touches the file every 30 s
        os.utime(hb, (t0 + elapsed, t0 + elapsed))
        now = t0 + elapsed + 29
        assert assess(run_dir, meta_of(run_dir), now=now).status == "running"
    assert assess(run_dir, meta_of(run_dir), now=t0 + 80 * 60 + 91).status == "stale"


# -- part 3: status at read time --------------------------------------------

@needs_identity
def test_sigkilled_run_is_dead(workdir):
    proc = spawn("""
        import time
        from exptrail import Run
        run = Run("victim").start()
        print(run.dir, flush=True)
        time.sleep(60)
    """, workdir)
    run_dir = Path(proc.stdout.readline().strip())
    try:
        assert SavedRun(run_dir).status == "running"
        assert SavedRun(run_dir).liveness.certain
    finally:
        proc.kill()  # SIGKILL: no cleanup code runs
        proc.communicate(timeout=60)
    run = SavedRun(run_dir)
    assert run.stored_status == "running"  # nothing wrote a status...
    assert run.status == "dead"  # ...but the evidence says it died
    assert run.status_display() == "dead (pid gone)"


@needs_identity
def test_saved_run_is_reassessed_on_every_read(workdir):
    """A long-lived SavedRun notices the job dying; nothing is cached."""
    proc = spawn("""
        import time
        from exptrail import Run
        run = Run("victim").start()
        print(run.dir, flush=True)
        time.sleep(60)
    """, workdir)
    run = SavedRun(Path(proc.stdout.readline().strip()))
    try:
        assert run.status == "running"
    finally:
        proc.kill()
        proc.communicate(timeout=60)
    assert run.status == "dead"  # same object, fresh assessment


def test_unknown_host_old_heartbeat_is_stale(workdir):
    run = SavedRun(fake_run(workdir, "old", remote(), heartbeat_age=14 * 60))
    assert run.status == "stale"
    assert not run.liveness.certain
    assert run.status_display() == f"stale (on {OTHER_HOST}, last heartbeat 14m ago)"


def test_unknown_host_fresh_heartbeat_is_running(workdir):
    run = SavedRun(fake_run(workdir, "fresh", remote(), heartbeat_age=5))
    assert run.status == "running"
    assert "last heartbeat" in run.status_display()


def test_stale_after_env_override(workdir, monkeypatch):
    run_dir = fake_run(workdir, "env", remote(), heartbeat_age=60)
    assert SavedRun(run_dir).status == "running"  # default: 3 x 30 s
    monkeypatch.setenv("EXPTRAIL_STALE_S", "45")
    assert SavedRun(run_dir).status == "stale"


def test_missing_fields_fall_back_to_heartbeat(workdir):
    # e.g. a run recorded before liveness records existed
    assert SavedRun(fake_run(workdir, "a", {}, heartbeat_age=5)).status == "running"
    old = fake_run(workdir, "b", {})
    t = time.time() - 3600
    for f in old.iterdir():
        os.utime(f, (t, t))
    assert SavedRun(old).status_display() == "stale (last activity 60m ago)"


@needs_identity
def test_live_process_on_this_host_is_running(workdir):
    run = SavedRun(fake_run(workdir, "me", here(), heartbeat_age=3600))
    assert run.status == "running" and run.liveness.certain  # heartbeat age irrelevant


@needs_identity
def test_same_pid_different_start_time_is_dead(workdir):
    run = SavedRun(fake_run(workdir, "reused", here(proc_start="12345-not-it"), heartbeat_age=0))
    assert run.status_display() == "dead (pid reused)"


@needs_identity
def test_different_boot_id_same_host_is_dead(workdir):
    other_boot = "boottime:1.000000" if sys.platform == "darwin" else "00000000-0000-0000-0000-000000000000"
    run = SavedRun(fake_run(workdir, "rebooted", here(boot_id=other_boot), heartbeat_age=0))
    assert run.status_display() == "dead (machine rebooted)"


@needs_identity
def test_gone_pid_same_host_is_dead(workdir):
    run = SavedRun(fake_run(workdir, "gone", here(pid=gone_pid()), heartbeat_age=0))
    assert run.status_display() == "dead (pid gone)"


@pytest.mark.parametrize("pid", [0, -1, True, "123", None])
def test_bogus_pids_never_signalled(workdir, monkeypatch, pid):
    monkeypatch.setattr(os, "kill", lambda *a: pytest.fail("os.kill called"))
    run = SavedRun(fake_run(workdir, "bogus", here(pid=pid), heartbeat_age=0))
    assert run.status == "running"  # heartbeat path


def test_windows_never_calls_os_kill(workdir, monkeypatch):
    monkeypatch.setattr(liveness, "IS_WINDOWS", True)
    monkeypatch.setattr(os, "kill", lambda *a: pytest.fail("os.kill(pid, 0) is CTRL_C_EVENT on Windows"))
    fresh = SavedRun(fake_run(workdir, "win-fresh", here(pid=gone_pid()), heartbeat_age=5))
    assert fresh.status == "running" and not fresh.liveness.certain
    old = SavedRun(fake_run(workdir, "win-old", here(), heartbeat_age=3600))
    assert old.status_display() == "stale (last heartbeat 60m ago)"


def test_finished_runs_are_not_assessed(workdir):
    run = SavedRun(fake_run(workdir, "done", remote(), heartbeat_age=10**6, status="finished"))
    assert run.status == "finished" and run.liveness is None


def snapshot(root):
    return {
        str(p.relative_to(root)): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_read_commands_modify_nothing(workdir, quiet, capsys):
    root = workdir / "runs"
    with Run("done") as run:
        run.summary(acc=0.5)
    fake_run(root, "stale", remote(), heartbeat_age=3600)
    fake_run(root, "dead", here(pid=gone_pid()), heartbeat_age=0)
    fake_run(root, "live", here(), heartbeat_age=0)
    runs = [SavedRun(p) for p in sorted(root.iterdir())]
    write_table(workdir / "README.md", render_table(runs, ["acc"], workdir / "README.md"))
    ids = [r.id for r in runs]
    before = snapshot(root)
    time.sleep(0.01)

    assert main(["ls"]) == 0
    for rid in ids:
        assert main(["show", rid]) == 0
    assert main(["compare", *ids]) == 0
    assert main(["verify", "README.md"]) == 1
    assert main(["verify", "--strict", "README.md"]) == 1

    assert snapshot(root) == before
    out = capsys.readouterr().out
    assert f"stale (on {OTHER_HOST}, last heartbeat 60m ago)" in out


# -- part 4: mark ------------------------------------------------------------

@needs_identity
def test_mark_refuses_live_run_without_force(workdir, capsys):
    run_dir = fake_run(workdir / "runs", "live", here(), heartbeat_age=0)
    before = (run_dir / "meta.json").read_bytes()
    assert main(["mark", "live", "failed"]) == 1
    assert "--force" in capsys.readouterr().err
    assert (run_dir / "meta.json").read_bytes() == before
    assert main(["mark", "live", "failed", "--force"]) == 0
    meta = meta_of(run_dir)
    assert meta["status"] == "failed"
    assert meta["marked"]["by"] == "hand" and meta["marked"]["at"]
    assert meta["marked"]["previous_status"] == "running"


def test_mark_refuses_heartbeat_running_run(workdir):
    fake_run(workdir / "runs", "remote-live", remote(), heartbeat_age=0)
    assert main(["mark", "remote-live", "interrupted"]) == 1


def test_mark_stale_run(workdir, capsys):
    run_dir = fake_run(workdir / "runs", "old", remote(), heartbeat_age=3600)
    assert main(["mark", "old", "interrupted"]) == 0
    assert "was stale" in capsys.readouterr().out
    meta = meta_of(run_dir)
    assert meta["status"] == "interrupted"
    assert meta["marked"]["shown_as"].startswith("stale")
    run = SavedRun(run_dir)
    assert run.status == "interrupted" and run.status_display() == "interrupted (marked by hand)"


def test_mark_rejects_unknown_status(workdir):
    fake_run(workdir / "runs", "old", remote(), heartbeat_age=3600)
    with pytest.raises(SystemExit):
        main(["mark", "old", "dead"])


# -- part 5: verify ----------------------------------------------------------

def test_verify_fails_on_stale_run(workdir, capsys):
    root = workdir / "runs"
    run = SavedRun(fake_run(root, "stale", remote(), heartbeat_age=3600))
    write_table(workdir / "README.md", render_table([run], ["acc"], workdir / "README.md"))
    assert main(["verify", "README.md"]) == 1
    assert f"stale: status is stale (on {OTHER_HOST}, last heartbeat 60m ago), not finished" in capsys.readouterr().err


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_verify_fails_on_unfinished_stored_status(workdir, capsys, status):
    run = SavedRun(fake_run(workdir / "runs", status, remote(), status=status))
    write_table(workdir / "README.md", render_table([run], ["acc"], workdir / "README.md"))
    assert main(["verify", "README.md"]) == 1
    assert f"status is {status}" in capsys.readouterr().err


# -- warnings point at the user's line ----------------------------------------

def _one_warning(record):
    ours = [w for w in record if issubclass(w.category, NoGitWarning)]
    assert len(ours) == 1, [str(w.message) for w in record]
    return ours[0]


def test_warning_points_at_with_line(workdir):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        line = sys._getframe().f_lineno + 1
        with Run("w"):
            pass
    w = _one_warning(record)
    assert (w.filename, w.lineno) == (__file__, line)


def test_warning_points_at_start_line(workdir):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        run = Run("s")
        line = sys._getframe().f_lineno + 1
        run.start()
        run.finish()
    w = _one_warning(record)
    assert (w.filename, w.lineno) == (__file__, line)


def test_warning_points_at_track_call(workdir):
    @track
    def train():
        return {"acc": 1.0}

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        line = sys._getframe().f_lineno + 1
        train()
    w = _one_warning(record)
    assert (w.filename, w.lineno) == (__file__, line)
