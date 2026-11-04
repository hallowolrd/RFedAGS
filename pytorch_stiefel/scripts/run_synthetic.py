from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rfedags import (
    StiefelPCAProblem, run_rfedags, run_rfedavg, run_rfedsvrg,
    run_rfedproj, run_zo_rfedproj,
)
from rfedags.data import make_synthetic_pca_data
from rfedags.io_utils import mean_histories, save_histories

ALGORITHMS = {
    "rfedags": run_rfedags,
    "rfedavg": run_rfedavg,
    "rfedsvrg": run_rfedsvrg,
    "rfedproj": run_rfedproj,
    "zo-rfedproj": run_zo_rfedproj,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="RFedAGS Stiefel PCA: MATLAB-aligned synthetic experiment.")
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--local-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=6e-3)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--algorithms", default="rfedags,rfedavg,rfedsvrg,rfedproj,zo-rfedproj")
    parser.add_argument("--output-dir", default="results/synthetic")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    n_agents, samples, d, r = 40, 100, 100, 5

    # One fixed dataset, initial point and participation-probability vector
    # are shared by every repeat and every algorithm, as in the MATLAB script.
    clients = make_synthetic_pca_data(
        n_agents, samples, d, seed=args.seed, dtype=dtype
    )
    problem = StiefelPCAProblem(clients, r)
    xgen = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    X0 = problem.manifold.random_point(dtype=dtype, device=device, generator=xgen)
    pgen = torch.Generator(device="cpu").manual_seed(args.seed + 2)
    probabilities = torch.rand(n_agents, generator=pgen, dtype=torch.float64)
    optimal_cost = problem.exact_optimal_cost(X0)

    print(f"Reference optimum (exact PCA spectrum): {optimal_cost.item():.10e}")
    print(f"Config: d={d}, r={r}, N={n_agents}, S={samples}, K={args.local_steps}, "
          f"B={args.batch_size}, alpha={args.learning_rate}, rounds={args.rounds}, repeats={args.repeats}")

    all_summaries = []
    for name in [x.strip().lower() for x in args.algorithms.split(",") if x.strip()]:
        if name not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm {name}; choose from {list(ALGORITHMS)}")
        run_histories = []
        for repeat in range(args.repeats):
            print(f"Running {name}: repeat {repeat + 1}/{args.repeats}")
            kwargs = dict(
                rounds=args.rounds, local_steps=args.local_steps,
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                probabilities=probabilities, check_every=args.check_every,
                seed=args.seed + 100 + repeat, optimal_cost=optimal_cost,
            )
            if name == "rfedags":
                history = run_rfedags(problem, X0, participation_mode="frequency", **kwargs)
            else:
                history = ALGORITHMS[name](problem, X0, **kwargs)
            history.algorithm = f"{name}-seed{repeat + 1}"
            run_histories.append(history)
        all_summaries.extend(run_histories)
        all_summaries.append(mean_histories(run_histories, algorithm=f"{name}-mean"))
    save_histories(all_summaries, Path(args.output_dir))
    print(f"Saved individual runs and mean curves to {args.output_dir}")


if __name__ == "__main__":
    main()
