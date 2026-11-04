from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rfedags import StiefelPCAProblem, run_rfedags
from rfedags.io_utils import mean_histories, save_histories


def main() -> None:
    parser = argparse.ArgumentParser(description="MATLAB-aligned Stiefel scalability sweeps.")
    parser.add_argument("--mode", choices=["agents", "samples", "dimension"], required=True)
    parser.add_argument("--rounds", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="results/scalability")
    parser.add_argument("--skip-reference", action="store_true",
                        help="Skip exact spectral optimum; gaps will be NaN.")
    args = parser.parse_args()

    dtype = torch.float64
    device = torch.device(args.device)
    # Recreate each MATLAB sweep's fixed master dataset once, then slice it.
    if args.mode == "agents":
        max_n, master_s, d = 200, 1000, 100
        master = []
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        for i in range(1, max_n + 1):
            master.append((i / max_n) * torch.randn(master_s, d, generator=gen, dtype=dtype))
        cases = [(n, 1000, d, 5) for n in (60, 80, 100, 150, 200)]
    elif args.mode == "samples":
        max_n, master_s, d = 100, 2000, 100
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        master = [
            (i / max_n) * torch.randn(master_s, d, generator=gen, dtype=dtype)
            for i in range(1, max_n + 1)
        ]
        cases = [(100, s, d, 5) for s in (400, 800, 1200, 1600, 2000)]
    else:
        max_n, master_s, max_d = 50, 1000, 4000
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        master = [
            (i / max_n) * torch.randn(master_s, max_d, generator=gen, dtype=dtype)
            for i in range(1, max_n + 1)
        ]
        cases = [
            (50, 1000, 1000, 5),
            (50, 1000, 2000, 5),  # intended value; MATLAB block accidentally repeats (1000,5)
            (50, 1000, 2000, 10),
            (50, 1000, 4000, 5),
            (50, 1000, 4000, 10),
        ]

    summaries = []
    for case_id, (n_agents, samples, d, r) in enumerate(cases):
        if args.mode == "agents":
            # Correct the MATLAB preallocation bug: every intended client has 1000 valid samples.
            clients = [master[i][:samples].clone() for i in range(n_agents)]
        elif args.mode == "samples":
            clients = [master[i][:samples].clone() for i in range(n_agents)]
        else:
            clients = [master[i][:samples, :d].clone() for i in range(n_agents)]

        problem = StiefelPCAProblem(clients, r)
        xgen = torch.Generator(device="cpu").manual_seed(args.seed + 1000 + case_id)
        X0 = problem.manifold.random_point(dtype=dtype, device=device, generator=xgen)
        pgen = torch.Generator(device="cpu").manual_seed(args.seed + 2000)
        probabilities = torch.rand(max_n, generator=pgen, dtype=torch.float64)[:n_agents]
        reference = None if args.skip_reference else problem.exact_optimal_cost(X0)
        if reference is None:
            print("Skipping exact PCA reference for this sweep.")
        else:
            print(f"Case {case_id+1}: exact optimum={reference.item():.10e}")

        runs = []
        print(f"Case {case_id+1}/{len(cases)}: N={n_agents}, S={samples}, d={d}, r={r}")
        for repeat in range(args.repeats):
            history = run_rfedags(
                problem, X0, rounds=args.rounds, local_steps=5,
                batch_size=samples // 2, learning_rate=2e-2,
                probabilities=probabilities, participation_mode="frequency",
                check_every=10, seed=args.seed + 3000 + repeat,
                optimal_cost=reference,
            )
            history.algorithm = f"{args.mode}-{case_id+1}-seed{repeat+1}"
            runs.append(history)
        mean = mean_histories(runs, algorithm=f"{args.mode}-{case_id+1}-mean")
        case_dir = Path(args.output_dir) / args.mode / f"N{n_agents}_S{samples}_d{d}_r{r}"
        save_histories([*runs, mean], case_dir)
        summaries.append(mean)

    save_histories(summaries, Path(args.output_dir) / args.mode / "summary")
    print(f"Saved repeat-level and mean curves under {args.output_dir}/{args.mode}")


if __name__ == "__main__":
    main()
