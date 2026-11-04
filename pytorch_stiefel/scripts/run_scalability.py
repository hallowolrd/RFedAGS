from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rfedags import StiefelPCAProblem, run_rfedags
from rfedags.data import make_synthetic_pca_data
from rfedags.io_utils import save_histories


def cases(mode: str):
    if mode == "agents":
        return [(n, 1000, 100, 5) for n in (60, 80, 100, 150, 200)]
    if mode == "samples":
        return [(100, s, 100, 5) for s in (400, 800, 1200, 1600, 2000)]
    if mode == "dimension":
        return [
            (50, 1000, 1000, 5),
            (50, 1000, 2000, 5),
            (50, 1000, 2000, 10),
            (50, 1000, 4000, 5),
            (50, 1000, 4000, 10),
        ]
    raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="RFedAGS Stiefel PCA scalability experiments.")
    parser.add_argument("--mode", choices=["agents", "samples", "dimension"], required=True)
    parser.add_argument("--rounds", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="results/scalability")
    args = parser.parse_args()

    device = torch.device(args.device)
    for case_id, (n_agents, samples, d, r) in enumerate(cases(args.mode)):
        print(
            f"\nCase {case_id + 1}: N={n_agents}, S={samples}, d={d}, r={r}, "
            f"T={args.rounds}"
        )
        clients = make_synthetic_pca_data(
            n_agents,
            samples,
            d,
            seed=args.seed + case_id + 1,
            dtype=torch.float32,
        )
        problem = StiefelPCAProblem(clients, r)
        x_generator = torch.Generator(device="cpu").manual_seed(args.seed + 100 + case_id)
        X0 = problem.manifold.random_point(
            dtype=torch.float32, device=device, generator=x_generator
        )
        p_generator = torch.Generator(device="cpu").manual_seed(args.seed + 200 + case_id)
        probabilities = torch.rand(n_agents, generator=p_generator, dtype=torch.float64)

        history = run_rfedags(
            problem,
            X0,
            rounds=args.rounds,
            local_steps=5,
            batch_size=max(1, samples // 2),
            learning_rate=2e-2,
            probabilities=probabilities,
            participation_mode="frequency",
            check_every=10,
            seed=args.seed + 300 + case_id,
            optimal_cost=None,
        )
        case_name = f"N{n_agents}_S{samples}_d{d}_r{r}"
        save_histories(
            [history],
            Path(args.output_dir) / args.mode / case_name,
        )
        print(
            f"Final objective={history.objectives[-1]:.8e}; "
            f"orthogonality error={problem.manifold.orthogonality_error(history.final_point).item():.3e}"
        )


if __name__ == "__main__":
    main()
