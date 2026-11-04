from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .algorithms import RunHistory


def save_histories(histories: Iterable[RunHistory], output_dir: str | Path) -> None:
    histories = list(histories)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for history in histories:
        for j, iteration in enumerate(history.iterations):
            rows.append(
                {
                    "algorithm": history.algorithm,
                    "iteration": iteration,
                    "objective": history.objectives[j],
                    "optimality_gap": history.gaps[j],
                    "riemannian_grad_norm": history.grad_norms[j],
                    "elapsed_seconds": history.elapsed_seconds[j],
                    "active_clients": history.active_clients[j],
                }
            )

    csv_path = output / "metrics.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    has_gaps = any(
        history.gaps and any(value == value for value in history.gaps)
        for history in histories
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for history in histories:
        values = history.gaps if has_gaps else history.objectives
        if values and any(value == value for value in values):
            plotted = [max(v, 1e-16) for v in values] if has_gaps else values
            ax.semilogy(history.iterations, plotted, label=history.algorithm)
    ax.set_xlabel("Communication rounds")
    ax.set_ylabel("Optimality gap" if has_gaps else "Objective")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / ("optimality_gap.png" if has_gaps else "objective.png"), dpi=180)
    plt.close(fig)
