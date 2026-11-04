from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .algorithms import RunHistory


def mean_histories(
    histories: Iterable[RunHistory], *, algorithm: str | None = None
) -> RunHistory:
    """Average the recorded curves from repeated runs with matching checkpoints."""
    histories = list(histories)
    if not histories:
        raise ValueError("At least one run history is required.")
    reference_iterations = histories[0].iterations
    if any(history.iterations != reference_iterations for history in histories[1:]):
        raise ValueError("Repeated runs must use identical checkpoint iterations.")

    def average(field: str) -> list[float]:
        columns = zip(*(getattr(history, field) for history in histories))
        return [
            sum(float(value) for value in column) / len(histories)
            for column in columns
        ]

    return RunHistory(
        algorithm=algorithm or histories[0].algorithm,
        iterations=list(reference_iterations),
        objectives=average("objectives"),
        gaps=average("gaps"),
        grad_norms=average("grad_norms"),
        elapsed_seconds=average("elapsed_seconds"),
        active_clients=[
            int(round(value)) for value in average("active_clients")
        ],
        final_point=None,
        participation_counts=None,
        gradient_samples=int(
            round(sum(history.gradient_samples for history in histories) / len(histories))
        ),
    )


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
