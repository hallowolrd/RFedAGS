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
from rfedags.data import load_cifar10_noniid
from rfedags.io_utils import mean_histories, save_histories

ALGORITHMS = {
    "rfedags": run_rfedags, "rfedavg": run_rfedavg,
    "rfedsvrg": run_rfedsvrg, "rfedproj": run_rfedproj,
    "zo-rfedproj": run_zo_rfedproj,
}


def load_matlab_clients(path: str, n_agents: int, samples: int, d: int,
                        dtype: torch.dtype) -> list[torch.Tensor]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError("Install scipy to load MATLAB .mat data: pip install scipy") from exc
    payload = loadmat(path)
    if "data" not in payload:
        keys = [key for key in payload if not key.startswith("__")]
        raise KeyError(f"Expected a 'data' matrix in the .mat file; available variables: {keys}")
    data = torch.as_tensor(payload["data"], dtype=dtype)
    if data.shape == (d, n_agents * samples):
        data = data.T.contiguous()
    if data.shape != (n_agents * samples, d):
        raise ValueError(f"Expected data shape {(n_agents*samples, d)} or its transpose; got {tuple(data.shape)}")
    return [data[i*samples:(i+1)*samples].clone() for i in range(n_agents)]


def main() -> None:
    parser = argparse.ArgumentParser(description="RFedAGS Stiefel PCA: CIFAR-10.")
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--mat-file", default=None,
                        help="Original CIFAR10_train_50_noniid_slight.mat; expects variable 'data'.")
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--local-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--algorithms", default="rfedags,rfedavg,rfedsvrg,rfedproj,zo-rfedproj")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--output-dir", default="results/cifar10")
    args = parser.parse_args()

    dtype = torch.float64
    n_agents, samples, d, r = 50, 1000, 3072, 4
    if args.mat_file:
        clients = load_matlab_clients(args.mat_file, n_agents, samples, d, dtype)
        data_source = f"MATLAB source: {args.mat_file}"
    else:
        clients = load_cifar10_noniid(
            args.data_root, n_agents=n_agents, shards_per_client=2,
            normalize=not args.no_normalize, download=True,
            seed=args.seed, dtype=dtype,
        )
        data_source = "torchvision label-shard fallback (not the author's original .mat partition)"
    if any(x.shape[0] != samples for x in clients):
        raise ValueError("Every client must have exactly 1000 samples.")

    problem = StiefelPCAProblem(clients, r)
    device = torch.device(args.device)
    xgen = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    X0 = problem.manifold.random_point(dtype=dtype, device=device, generator=xgen)
    pgen = torch.Generator(device="cpu").manual_seed(args.seed + 2)
    probabilities = torch.rand(n_agents, generator=pgen, dtype=torch.float64)
    reference = None if args.skip_reference else problem.exact_optimal_cost(X0)
    print(data_source)
    if reference is None:
        print("Exact PCA reference disabled; optimality gaps will be NaN.")
    else:
        print(f"Exact PCA reference: {reference.item():.10e}")

    all_histories = []
    for name in [x.strip().lower() for x in args.algorithms.split(",") if x.strip()]:
        if name not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm {name}; choose from {list(ALGORITHMS)}")
        runs = []
        for repeat in range(args.repeats):
            kwargs = dict(
                rounds=args.rounds, local_steps=args.local_steps,
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                probabilities=probabilities, check_every=args.check_every,
                seed=args.seed + 100 + repeat, optimal_cost=reference,
            )
            if name == "rfedags":
                history = run_rfedags(problem, X0, participation_mode="frequency", **kwargs)
            else:
                history = ALGORITHMS[name](problem, X0, **kwargs)
            history.algorithm = f"{name}-seed{repeat+1}"
            runs.append(history)
        all_histories.extend(runs)
        all_histories.append(mean_histories(runs, algorithm=f"{name}-mean"))

    save_histories(all_histories, Path(args.output_dir))
    print(f"Saved individual runs and mean curves to {args.output_dir}")


if __name__ == "__main__":
    main()
