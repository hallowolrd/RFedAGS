from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rfedags import StiefelPCAProblem, run_rfedags, run_rfedavg
from rfedags.io_utils import save_histories


def main() -> None:
    parser = argparse.ArgumentParser(description="RFedAGS versus tangent-mean RFedAvg.")
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="results/ags_vs_tm")
    args = parser.parse_args()

    n_agents, samples, d, r = 2, 60, 50, 1
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    clients = [
        0.5 * torch.randn(samples, d, generator=generator, dtype=torch.float64),
        400.0 * torch.randn(samples, d, generator=generator, dtype=torch.float64),
    ]
    problem = StiefelPCAProblem(clients, r)
    xgen = torch.Generator(device=device).manual_seed(args.seed + 1)
    X0 = problem.manifold.random_point(dtype=torch.float64, device=args.device, generator=xgen)
    probabilities = torch.ones(n_agents, dtype=torch.float64)
    optimal_cost = problem.exact_optimal_cost(X0)
    common = dict(
        rounds=args.rounds, local_steps=50, batch_size=samples,
        learning_rate=1e-7, probabilities=probabilities, check_every=10,
        seed=args.seed + 2, optimal_cost=optimal_cost,
    )
    ags = run_rfedags(problem, X0, participation_mode="full", **common)
    tangent_mean = run_rfedavg(problem, X0, **common)
    save_histories([ags, tangent_mean], Path(args.output_dir))
    print(f"Reference optimum: {optimal_cost.item():.10e}")
    for history in (ags, tangent_mean):
        print(f"{history.algorithm}: final gap={history.gaps[-1]:.8e}, "
              f"orthogonality error={problem.manifold.orthogonality_error(history.final_point).item():.3e}")


if __name__ == "__main__":
    main()
