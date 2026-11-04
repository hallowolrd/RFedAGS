from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Literal

import torch

from .problem import StiefelPCAProblem


@dataclass
class RunHistory:
    algorithm: str
    iterations: list[int] = field(default_factory=list)
    objectives: list[float] = field(default_factory=list)
    gaps: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    elapsed_seconds: list[float] = field(default_factory=list)
    active_clients: list[int] = field(default_factory=list)
    final_point: torch.Tensor | None = None
    participation_counts: torch.Tensor | None = None
    gradient_samples: int = 0


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(int(seed))


def _sample_batch(
    n_samples: int, batch_size: int, generator: torch.Generator
) -> torch.Tensor:
    size = min(int(batch_size), int(n_samples))
    return torch.randperm(n_samples, generator=generator)[:size]


def _select_clients(
    n_agents: int,
    probabilities: torch.Tensor,
    generator: torch.Generator,
    participation: Literal["bernoulli", "fixed", "full"],
    fixed_clients: int | None = None,
) -> list[int]:
    if participation == "full":
        return list(range(n_agents))
    if participation == "fixed":
        count = n_agents if fixed_clients is None else max(1, min(n_agents, fixed_clients))
        return torch.randperm(n_agents, generator=generator)[:count].tolist()

    active = torch.nonzero(
        torch.rand(n_agents, generator=generator) < probabilities,
        as_tuple=False,
    ).flatten().tolist()
    # Match the MATLAB implementation: force one responder if a round is empty.
    if not active:
        active = [int(torch.randint(n_agents, (1,), generator=generator).item())]
    return active


def _step_size(
    base_lr: float,
    round_index: int,
    decay_every: int | None,
    decay_lambda: float,
) -> float:
    if decay_every is None or decay_every <= 0 or round_index == 0:
        return float(base_lr)
    decay_count = round_index // decay_every
    if decay_count == 0:
        return float(base_lr)
    return float(base_lr) / (float(decay_lambda) + decay_count)


def _record(
    history: RunHistory,
    problem: StiefelPCAProblem,
    X: torch.Tensor,
    iteration: int,
    elapsed: float,
    active_count: int,
    optimal_cost: torch.Tensor | float | None,
) -> None:
    objective = float(problem.cost(X).detach().cpu())
    grad_norm = float(problem.grad_norm(X).detach().cpu())
    if optimal_cost is None:
        gap = float("nan")
    else:
        reference = float(
            optimal_cost.detach().cpu()
            if isinstance(optimal_cost, torch.Tensor)
            else optimal_cost
        )
        gap = max(objective - reference, 0.0)
    history.iterations.append(int(iteration))
    history.objectives.append(objective)
    history.gaps.append(gap)
    history.grad_norms.append(grad_norm)
    history.elapsed_seconds.append(float(elapsed))
    history.active_clients.append(int(active_count))


def _finish(
    history: RunHistory,
    X: torch.Tensor,
    counts: torch.Tensor,
) -> RunHistory:
    history.final_point = X.detach().clone()
    history.participation_counts = counts.detach().clone()
    return history


def _initial_history(
    name: str,
    problem: StiefelPCAProblem,
    X: torch.Tensor,
    optimal_cost: torch.Tensor | float | None,
) -> RunHistory:
    history = RunHistory(algorithm=name)
    _record(history, problem, X, 0, 0.0, 0, optimal_cost)
    return history


def run_rfedags(
    problem: StiefelPCAProblem,
    X0: torch.Tensor,
    *,
    rounds: int = 1000,
    local_steps: int = 5,
    batch_size: int = 50,
    learning_rate: float = 6e-3,
    probabilities: torch.Tensor | None = None,
    participation_mode: Literal["frequency", "true", "fixed", "full"] = "frequency",
    fixed_clients: int | None = None,
    global_step: float = 1.0,
    decay_every: int | None = None,
    decay_lambda: float = 1.0,
    check_every: int = 10,
    seed: int = 0,
    optimal_cost: torch.Tensor | float | None = None,
) -> RunHistory:
    """RFedAGS with local Riemannian SGD and projection vector transport.

    The default frequency mode mirrors RFedAGSP.m / option == 2: clients
    independently respond with probabilities p_i, cumulative response
    frequencies q_i are estimated online, and the server applies
        X_{t+1} = R_{X_t}(global_step / N * sum_{i in S_t} zeta_i / q_i).
    The stream zeta_i stores negative local gradient steps, as in the MATLAB
    implementation.
    """
    if local_steps < 1 or rounds < 1 or batch_size < 1:
        raise ValueError("rounds, local_steps and batch_size must be positive.")
    n_agents = problem.n_agents
    if probabilities is None:
        probabilities = torch.full((n_agents,), 0.5, dtype=torch.float64)
    probabilities = torch.as_tensor(probabilities, dtype=torch.float64, device="cpu")
    if probabilities.numel() != n_agents:
        raise ValueError("probabilities must have one entry per client.")
    if torch.any(probabilities <= 0) or torch.any(probabilities > 1):
        raise ValueError("Each participation probability must lie in (0, 1].")
    if participation_mode not in {"frequency", "true", "fixed", "full"}:
        raise ValueError(f"Unknown participation_mode: {participation_mode}")

    generator = _generator(seed)
    X = X0.detach().clone()
    counts = torch.zeros(n_agents, dtype=torch.long)
    history = _initial_history("RFedAGS", problem, X, optimal_cost)
    cumulative_time = 0.0
    base_lr = float(learning_rate)

    if participation_mode in {"frequency", "true"}:
        selection_mode = "bernoulli"
    else:
        selection_mode = participation_mode

    for t in range(rounds):
        lr = _step_size(base_lr, t, decay_every, decay_lambda)
        round_start = time.perf_counter()
        active = _select_clients(
            n_agents, probabilities, generator, selection_mode, fixed_clients
        )
        counts[active] += 1

        streams: dict[int, torch.Tensor] = {}
        for client_id in active:
            X_local = X.clone()
            stream = torch.zeros_like(X)
            n_local = problem.client_sizes[client_id]
            for k in range(local_steps):
                indices = _sample_batch(n_local, batch_size, generator)
                grad = problem.client_grad(client_id, X_local, indices)
                eta = -lr * grad
                X_next = problem.manifold.retract(X_local, eta)
                if k == 0:
                    stream = eta
                else:
                    stream = stream + problem.manifold.transport(X_local, X, eta)
                X_local = X_next
                history.gradient_samples += min(batch_size, n_local)
            streams[client_id] = stream

        if participation_mode == "frequency":
            # The MATLAB code updates the count with the current response and
            # divides by t+1 before aggregation.
            q = counts.to(dtype=X.dtype, device=X.device) / float(t + 1)
            delta = torch.zeros_like(X)
            for client_id in active:
                delta = delta + streams[client_id] / q[client_id]
            delta = delta / n_agents
        elif participation_mode == "true":
            p = probabilities.to(dtype=X.dtype, device=X.device)
            delta = torch.zeros_like(X)
            for client_id in active:
                delta = delta + streams[client_id] / p[client_id]
            delta = delta / n_agents
        else:
            weights = torch.tensor(
                [problem.client_sizes[i] for i in active],
                dtype=X.dtype,
                device=X.device,
            )
            total_weight = weights.sum()
            delta = torch.zeros_like(X)
            for weight, client_id in zip(weights, active):
                delta = delta + weight * streams[client_id]
            delta = delta / total_weight

        X = problem.manifold.retract(X, global_step * delta)
        cumulative_time += time.perf_counter() - round_start

        if (t + 1) % check_every == 0 or t + 1 == rounds:
            _record(
                history, problem, X, t + 1, cumulative_time, len(active), optimal_cost
            )

    return _finish(history, X, counts)


def run_rfedavg(
    problem: StiefelPCAProblem,
    X0: torch.Tensor,
    *,
    rounds: int = 1000,
    local_steps: int = 5,
    batch_size: int = 50,
    learning_rate: float = 6e-3,
    probabilities: torch.Tensor | None = None,
    check_every: int = 10,
    seed: int = 0,
    optimal_cost: torch.Tensor | float | None = None,
) -> RunHistory:
    """RFedAvg baseline: local RSGD followed by inverse-QR tangent-mean aggregation."""
    n_agents = problem.n_agents
    if probabilities is None:
        probabilities = torch.full((n_agents,), 0.5, dtype=torch.float64)
    probabilities = torch.as_tensor(probabilities, dtype=torch.float64, device="cpu")
    generator = _generator(seed)
    X = X0.detach().clone()
    counts = torch.zeros(n_agents, dtype=torch.long)
    history = _initial_history("RFedAvg", problem, X, optimal_cost)
    cumulative_time = 0.0

    for t in range(rounds):
        start = time.perf_counter()
        active = _select_clients(n_agents, probabilities, generator, "bernoulli")
        counts[active] += 1
        local_points = []
        for client_id in active:
            X_local = X.clone()
            n_local = problem.client_sizes[client_id]
            for _ in range(local_steps):
                indices = _sample_batch(n_local, batch_size, generator)
                grad = problem.client_grad(client_id, X_local, indices)
                X_local = problem.manifold.retract(X_local, -learning_rate * grad)
                history.gradient_samples += min(batch_size, n_local)
            local_points.append(X_local)

        tangent_sum = torch.zeros_like(X)
        for local_point in local_points:
            tangent_sum = tangent_sum + problem.manifold.inverse_qr_retraction(X, local_point)
        X = problem.manifold.retract(X, tangent_sum / len(active))
        cumulative_time += time.perf_counter() - start

        if (t + 1) % check_every == 0 or t + 1 == rounds:
            _record(history, problem, X, t + 1, cumulative_time, len(active), optimal_cost)

    return _finish(history, X, counts)


def run_rfedsvrg(
    problem: StiefelPCAProblem,
    X0: torch.Tensor,
    *,
    rounds: int = 1000,
    local_steps: int = 5,
    batch_size: int = 50,
    learning_rate: float = 6e-3,
    probabilities: torch.Tensor | None = None,
    check_every: int = 10,
    seed: int = 0,
    optimal_cost: torch.Tensor | float | None = None,
) -> RunHistory:
    """RFedSVRG baseline ported from RFedSVRG_.m."""
    n_agents = problem.n_agents
    if probabilities is None:
        probabilities = torch.full((n_agents,), 0.5, dtype=torch.float64)
    probabilities = torch.as_tensor(probabilities, dtype=torch.float64, device="cpu")
    generator = _generator(seed)
    X = X0.detach().clone()
    counts = torch.zeros(n_agents, dtype=torch.long)
    history = _initial_history("RFedSVRG", problem, X, optimal_cost)
    cumulative_time = 0.0

    for t in range(rounds):
        start = time.perf_counter()
        active = _select_clients(n_agents, probabilities, generator, "bernoulli")
        counts[active] += 1

        # The reference MATLAB code divides the sum of active local full
        # gradients by N, not by the number of responders.
        global_grad = torch.zeros_like(X)
        for client_id in active:
            global_grad = global_grad + problem.client_grad(client_id, X)
            history.gradient_samples += problem.client_sizes[client_id]
        global_grad = global_grad / n_agents

        local_points = []
        for client_id in active:
            X_local = X.clone()
            n_local = problem.client_sizes[client_id]
            for _ in range(local_steps):
                indices = _sample_batch(n_local, batch_size, generator)
                grad_at_local = problem.client_grad(client_id, X_local, indices)
                grad_at_global = problem.client_grad(client_id, X, indices)
                transported = problem.manifold.transport(X, X_local, grad_at_global - global_grad)
                direction = grad_at_local - transported
                X_local = problem.manifold.retract(X_local, -learning_rate * direction)
                history.gradient_samples += 2 * min(batch_size, n_local)
            local_points.append(X_local)

        tangent_sum = torch.zeros_like(X)
        for local_point in local_points:
            tangent_sum = tangent_sum + problem.manifold.inverse_qr_retraction(X, local_point)
        X = problem.manifold.retract(X, tangent_sum / len(active))
        cumulative_time += time.perf_counter() - start

        if (t + 1) % check_every == 0 or t + 1 == rounds:
            _record(history, problem, X, t + 1, cumulative_time, len(active), optimal_cost)

    return _finish(history, X, counts)


def _run_projection_method(
    problem: StiefelPCAProblem,
    X0: torch.Tensor,
    *,
    name: str,
    zeroth_order: bool,
    rounds: int,
    local_steps: int,
    batch_size: int,
    learning_rate: float,
    probabilities: torch.Tensor | None,
    check_every: int,
    seed: int,
    optimal_cost: torch.Tensor | float | None,
    global_step: float = 1.0,
    mu: float = 1e-5,
) -> RunHistory:
    n_agents = problem.n_agents
    if probabilities is None:
        probabilities = torch.full((n_agents,), 0.5, dtype=torch.float64)
    probabilities = torch.as_tensor(probabilities, dtype=torch.float64, device="cpu")
    generator = _generator(seed)
    X_mid = X0.detach().clone()
    X_old = X_mid.clone()
    counts = torch.zeros(n_agents, dtype=torch.long)
    history = _initial_history(name, problem, X_mid, optimal_cost)
    previous_active: set[int] = set()
    stored_streams: dict[int, torch.Tensor] = {}
    cumulative_time = 0.0

    for t in range(rounds):
        start = time.perf_counter()
        active = _select_clients(n_agents, probabilities, generator, "bernoulli")
        counts[active] += 1
        local_ambient: dict[int, torch.Tensor] = {}

        for client_id in active:
            if t == 0 or client_id not in previous_active:
                correction = torch.zeros_like(X_mid)
            else:
                correction = (
                    problem.manifold.polar_projection(X_old) - X_mid
                ) / (local_steps * learning_rate * global_step) - stored_streams[client_id]

            Z_hat = problem.manifold.polar_projection(X_mid)
            Z = Z_hat.clone()
            stream = torch.zeros_like(X_mid)
            n_local = problem.client_sizes[client_id]

            for _ in range(local_steps):
                indices = _sample_batch(n_local, batch_size, generator)
                if zeroth_order:
                    estimate = torch.zeros_like(X_mid)
                    for sample_id in indices.tolist():
                        direction = torch.randn(
                            X_mid.shape,
                            generator=generator,
                            dtype=X_mid.dtype,
                            device="cpu",
                        ).to(device=X_mid.device)
                        direction = direction / torch.linalg.matrix_norm(direction)
                        projected = problem.manifold.polar_projection(Z + mu * direction)
                        f_plus = problem.client_cost(
                            client_id, projected, torch.tensor([sample_id])
                        )
                        f_base = problem.client_cost(
                            client_id, Z, torch.tensor([sample_id])
                        )
                        estimate = estimate + ((f_plus - f_base) / mu) * direction
                    eta = (problem.d * problem.r / len(indices)) * estimate
                else:
                    eta = problem.client_grad(client_id, Z, indices)

                Z_hat = Z_hat - learning_rate * (eta + correction)
                Z = problem.manifold.polar_projection(Z_hat)
                stream = stream + eta
                history.gradient_samples += min(batch_size, n_local)

            stored_streams[client_id] = stream / local_steps
            local_ambient[client_id] = Z_hat

        ambient_sum = torch.zeros_like(X_mid)
        for client_id in active:
            ambient_sum = ambient_sum + local_ambient[client_id]
        X_new = (
            (1.0 - global_step) * problem.manifold.polar_projection(X_mid)
            + (global_step / len(active)) * ambient_sum
        )
        X_old = X_mid
        X_mid = X_new
        X = problem.manifold.polar_projection(X_mid)
        previous_active = set(active)
        cumulative_time += time.perf_counter() - start

        if (t + 1) % check_every == 0 or t + 1 == rounds:
            _record(history, problem, X, t + 1, cumulative_time, len(active), optimal_cost)

    return _finish(history, X, counts)


def run_rfedproj(
    problem: StiefelPCAProblem,
    X0: torch.Tensor,
    *,
    rounds: int = 1000,
    local_steps: int = 5,
    batch_size: int = 50,
    learning_rate: float = 6e-3,
    probabilities: torch.Tensor | None = None,
    check_every: int = 10,
    seed: int = 0,
    optimal_cost: torch.Tensor | float | None = None,
) -> RunHistory:
    """RFedProj baseline with polar projection, matching RFedProj_.m."""
    return _run_projection_method(
        problem,
        X0,
        name="RFedProj",
        zeroth_order=False,
        rounds=rounds,
        local_steps=local_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        probabilities=probabilities,
        check_every=check_every,
        seed=seed,
        optimal_cost=optimal_cost,
    )


def run_zo_rfedproj(
    problem: StiefelPCAProblem,
    X0: torch.Tensor,
    *,
    rounds: int = 1000,
    local_steps: int = 5,
    batch_size: int = 50,
    learning_rate: float = 6e-3,
    probabilities: torch.Tensor | None = None,
    check_every: int = 10,
    seed: int = 0,
    optimal_cost: torch.Tensor | float | None = None,
    mu: float = 1e-5,
) -> RunHistory:
    """Zeroth-order RFedProj baseline matching ZORFedProj_.m."""
    return _run_projection_method(
        problem,
        X0,
        name="ZO-RFedProj",
        zeroth_order=True,
        rounds=rounds,
        local_steps=local_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        probabilities=probabilities,
        check_every=check_every,
        seed=seed,
        optimal_cost=optimal_cost,
        mu=mu,
    )
