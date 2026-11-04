from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rfedags import (
    StiefelPCAProblem,
    run_rfedags,
    run_rfedavg,
    run_rfedsvrg,
    run_rfedproj,
    run_zo_rfedproj,
)
from rfedags.data import make_synthetic_pca_data
from rfedags.io_utils import save_histories


ALGORITHMS = {
    "rfedags": run_rfedags,
    "rfedavg": run_rfedavg,
    "rfedsvrg": run_rfedsvrg,
    "rfedproj": run_rfedproj,
    "zo-rfedproj": run_zo_rfedproj,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="RFedAGS Stiefel PCA: synthetic experiment.")
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--local-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=6e-3)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--algorithms",
        default="rfedags",
        help="Comma-separated: rfedags,rfedavg,rfedsvrg,rfedproj,zo-rfedproj.",
    )
    parser.add_argument("--output-dir", default="results/synthetic")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    n_agents, samples_per_agent, d, r = 40, 100, 100, 5
    clients = make_synthetic_pca_data(
        n_agents, samples_per_agent, d, seed=args.seed + 1
    )
    problem = StiefelPCAProblem(clients, r)
    x_generator = torch.Generator(device="cpu").manual_seed(args.seed + 2)
    X0 = problem.manifold.random_point(dtype=dtype, device=device, generator=x_generator)
    p_generator = torch.Generator(device="cpu").manual_seed(args.seed + 3)
    probabilities = torch.rand(n_agents, generator=p_generator, dtype=torch.float64)

    optimal_cost = problem.exact_optimal_cost(X0)
    print(f"Global optimum objective (spectral reference): {optimal_cost.item():.8e}")
    print(
        f"Config: d={d}, r={r}, N={n_agents}, S={samples_per_agent}, "
        f"K={args.local_steps}, B={args.batch_size}, alpha={args.learning_rate}, "
        f"T={args.rounds}"
    )

    histories = []
    names = [name.strip().lower() for name in args.algorithms.split(",") if name.strip()]
    for name in names:
        if name not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{name}'. Choices: {list(ALGORITHMS)}")
        print(f"\nRunning {name} ...")
        kwargs = dict(
            rounds=args.rounds,
            local_steps=args.local_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            probabilities=probabilities,
            check_every=args.check_every,
            seed=args.seed + 10,
            optimal_cost=optimal_cost,
        )
        if name == "rfedags":
            history = run_rfedags(
                problem, X0, participation_mode="frequency", **kwargs
            )
        else:
            history = ALGORITHMS[name](problem, X0, **kwargs)
        histories.append(history)
        print(
            f"{name}: final objective={history.objectives[-1]:.8e}, "
            f"final gap={history.gaps[-1]:.8e}, "
            f"orthogonality error={problem.manifold.orthogonality_error(history.final_point).item():.3e}"
        )

    save_histories(histories, Path(args.output_dir))
    print(f"\nSaved metrics and plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
