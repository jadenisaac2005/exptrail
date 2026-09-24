"""Command-line interface: ls, show, compare, plot, table, verify."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from .readme import render_table, verify_readme, write_table
from .run import default_root
from .store import SavedRun, find_run, glob_runs, list_runs


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return "—" if value is None else str(value)


def print_table(header: list[str], rows: list[list[str]], out=None) -> None:
    out = out or sys.stdout
    widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    for i, row in enumerate([header, *rows]):
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip(), file=out)
        if i == 0:
            print("  ".join("-" * w for w in widths), file=out)


def _resolve(refs: list[str], root: Path) -> list[SavedRun]:
    return [find_run(r, root) for r in refs]


def cmd_ls(args) -> int:
    runs = list_runs(args.root)
    if not runs:
        print(f"no runs in {args.root}")
        return 0
    rows = []
    for run in runs:
        summary = run.summary
        keys = args.metric or list(summary)[:3]
        metrics = ", ".join(f"{k}={_fmt(summary.get(k))}" for k in keys)
        rows.append([run.id, run.status, metrics, run.short_commit()])
    print_table(["run", "status", "summary", "commit"], rows)
    if any(r.dirty for r in runs):
        print("\n* = ran on a dirty working tree")
    return 0


def cmd_show(args) -> int:
    run = find_run(args.run, args.root)
    print(f"# {run.id}  ({run.path})\n")
    for title, data in (("config", run.config), ("summary", run.summary), ("meta", run.meta)):
        print(f"## {title}")
        print(json.dumps(data, indent=2))
        print()
    return 0


def cmd_compare(args) -> int:
    runs = _resolve(args.runs, args.root)
    ids = [r.id for r in runs]
    configs = [r.config for r in runs]
    keys = list(dict.fromkeys(k for c in configs for k in c))
    differing = [k for k in keys if len({json.dumps(c.get(k), sort_keys=True) for c in configs}) > 1]
    shown = keys if args.all else differing
    print("## config" + ("" if args.all else " (differing keys only; --all for everything)"))
    if shown:
        print_table(["key", *ids], [[k, *(_fmt(c.get(k)) for c in configs)] for k in shown])
    else:
        print("(identical)")
    summaries = [r.summary for r in runs]
    mkeys = list(dict.fromkeys(k for s in summaries for k in s))
    print("\n## summary")
    print_table(["metric", *ids], [[k, *(_fmt(s.get(k)) for s in summaries)] for k in mkeys])
    print("\n## provenance")
    print_table(["", *ids], [["status", *(r.status for r in runs)], ["commit", *(r.short_commit() for r in runs)]])
    return 0


def cmd_plot(args) -> int:
    from .plot import plot_runs

    runs = _resolve(args.runs, args.root)
    out = Path(args.out or f"{args.metric}.png")
    n = plot_runs(runs, args.metric, out, x=args.x)
    if n == 0:
        print(f"error: no run has logged {args.metric!r}", file=sys.stderr)
        return 1
    print(f"wrote {out} ({n} run{'s' if n != 1 else ''})")
    return 0


def cmd_table(args) -> int:
    runs = []
    for pattern in args.runs:
        runs += [r for r in glob_runs(pattern, args.root) if r.path not in {x.path for x in runs}]
    if not args.include_failed:
        runs = [r for r in runs if r.status == "finished"]
    if not runs:
        print("error: no finished runs match " + " ".join(args.runs), file=sys.stderr)
        return 1
    readme = Path(args.readme)
    command = "exptrail " + shlex.join(args.argv)
    block = render_table(runs, args.metric, readme, args.config or [], args.precision, command)
    changed = write_table(readme, block)
    print(f"{'updated' if changed else 'unchanged'}: {readme} ({len(runs)} run{'s' if len(runs) != 1 else ''})")
    for r in runs:
        if r.dirty:
            print(f"warning: {r.id} ran on a dirty working tree", file=sys.stderr)
    return 0


def cmd_verify(args) -> int:
    failed = False
    for readme in args.readme:
        res = verify_readme(Path(readme), strict=args.strict)
        for w in res.warnings:
            print(f"warning: {w}", file=sys.stderr)
        for e in res.errors:
            print(f"MISMATCH: {e}", file=sys.stderr)
        status = "OK" if res.ok else "FAILED"
        print(f"{status}: {readme} — {res.rows} rows, {res.checked} values checked")
        failed |= not res.ok
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="exptrail", description=__doc__)
    p.add_argument("--root", type=Path, default=None, help="runs directory (default: $EXPTRAIL_ROOT or ./runs)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ls", help="list runs")
    s.add_argument("--metric", action="append", help="summary metric to show (repeatable)")
    s.set_defaults(func=cmd_ls)

    s = sub.add_parser("show", help="show config, summary and meta of a run")
    s.add_argument("run")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("compare", help="config diff and metric table for several runs")
    s.add_argument("runs", nargs="+")
    s.add_argument("--all", action="store_true", help="show all config keys, not just differing ones")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("plot", help="overlay a metric's curves to a PNG")
    s.add_argument("runs", nargs="+")
    s.add_argument("--metric", required=True)
    s.add_argument("--x", default="step", help="x-axis column (step or elapsed)")
    s.add_argument("--out", "-o", help="output PNG (default: <metric>.png)")
    s.set_defaults(func=cmd_plot)

    s = sub.add_parser("table", help="write a results table into a README")
    s.add_argument("--metric", action="append", required=True, help="summary metric column (repeatable)")
    s.add_argument("--runs", nargs="+", default=["*"], help="glob(s) of run folders (default: all)")
    s.add_argument("--readme", default="README.md")
    s.add_argument("--config", action="append", help="config key to show as a column (repeatable)")
    s.add_argument("--precision", type=int, default=4, help="decimals for floats (default 4)")
    s.add_argument("--include-failed", action="store_true")
    s.set_defaults(func=cmd_table)

    s = sub.add_parser("verify", help="check README results against saved runs")
    s.add_argument("readme", nargs="+")
    s.add_argument("--strict", action="store_true", help="also fail on dirty, failed or git-less runs")
    s.set_defaults(func=cmd_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    args.root = args.root or default_root()
    args.argv = argv
    try:
        return args.func(args)
    except (LookupError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
