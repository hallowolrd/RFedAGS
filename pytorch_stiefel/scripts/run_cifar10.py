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
from rfedags.data import load_cifar10_noniid
from rfedags.io_utils import save_histories


ALGORITHMS = {
    "rfedags": run_rfedags,
    "rfedavg": run_rfedavg,
    "rfedsvrg": run_rfedsvrg,
    "rfedproj": run_rfedproj,
    "zo-rfedproj": run_zo_rfedproj,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="RFedAGS Stiefel PCA: CIFAR-10 experiment.")
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--local-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument(
        "--algorithms",
        default="rfedags",
        help="Comma-separated: rfedags,rfedavg,rfedsvrg,rfedproj,zo-rfedproj.",
    )
    parser.add_argument("--compute-reference", action="store_true")
    parser.add_argument("--output-dir", default="results/cifar10")
    args = parser.parse_args()

    device = torch.device(args.device)
    n_agents, samples_per_agent, d, r = 50, 1000, 3072, 4
    clients = load_cifar10_noniid(
        args.data_root,
        n_agents=n_agents,
        shards_per_client=2,
        normalize=not args.no_normalize,
        download=True,
        seed=args.seed + 1,
    )
    if any(client.shape[0] != samples_per_agent for client in clients):
        sizes = [client.shape[0] for client in clients]
        raise RuntimeError(
            f"Expected 1000 samples per client, got sizes {sizes}. "
            "Adjust shard partitioning or experiment constants."
        )

    problem = StiefelPCAProblem(clients, r)
    x_generator = torch.Generator(device="cpu").manual_seed(args.seed + 2)
    X0 = problem.manifold.random_point(
        dtype=torch.float32, device=device, generator=x_generator
    )
    p_generator = torch.Generator(device="cpu").manual_seed(args.seed + 3)
    probabilities = torch.rand(n_agents, generator=p_generator, dtype=torch.float64)

    optimal_cost = problem.exact_optimal_cost(X0) if args.compute_reference else None
    print(
        f"Config: d={d}, r={r}, N={n_agents}, S={samples_per_agent}, "
        f"K={args.local_steps}, B={args.batch_size}, alpha={args.learning_rate}, "
        f"T={args.rounds}, normalized={not args.no_normalize}"
    )
    if optimal_cost is not None:
        print(f"Global optimum objective (spectral reference): {optimal_cost.item():.8e}")
    else:
        print("Spectral reference disabled; optimality-gap values will be NaN.")

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
            f"final grad norm={history.grad_norms[-1]:.8e}, "
            f"orthogonality error={problem.manifold.orthogonality_error(history.final_point).item():.3e}"
        )

    save_histories(histories, Path(args.output_dir))
    print(f"\nSaved metrics and plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
