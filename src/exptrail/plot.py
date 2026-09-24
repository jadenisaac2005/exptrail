"""Overlay metric curves from several runs (needs matplotlib)."""

from __future__ import annotations

from pathlib import Path

from .store import SavedRun


def plot_runs(runs: list[SavedRun], metric: str, out: Path, x: str = "step") -> int:
    """Plot ``metric`` against ``x`` for each run; return how many runs had data."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("plotting needs matplotlib: pip install 'exptrail[plot]'") from None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = 0
    for run in runs:
        points = [(r[x], r[metric]) for r in run.metrics() if x in r and metric in r]
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, label=f"{run.name} ({run.short_commit()})", linewidth=1.8)
        plotted += 1
    ax.set_xlabel(x)
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return plotted
