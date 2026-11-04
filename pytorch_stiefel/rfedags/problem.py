from __future__ import annotations

from collections.abc import Sequence

import torch

from .geometry import StiefelManifold


class StiefelPCAProblem:
    """Federated PCA objective on a Stiefel manifold.

    Each client stores row-wise samples Z_i with shape [S_i, d]. The local
    objective is f_i(X) = -mean_j ||Z_ij @ X||_2^2. The global objective is
    the unweighted mean of client objectives, matching the authors' MATLAB
    ConstructRFLPCAProb_St_ implementation.
    """

    def __init__(
        self,
        client_data: Sequence[torch.Tensor],
        r: int,
        manifold: StiefelManifold | None = None,
    ):
        if not client_data:
            raise ValueError("client_data must contain at least one client.")
        self.clients = [torch.as_tensor(x).detach().contiguous() for x in client_data]
        d = int(self.clients[0].shape[-1])
        if any(x.ndim != 2 or x.shape[1] != d or x.shape[0] == 0 for x in self.clients):
            raise ValueError("Each client dataset must be a non-empty [samples, d] tensor.")
        self.d = d
        self.r = int(r)
        self.manifold = manifold or StiefelManifold(d, r)
        if self.manifold.d != d or self.manifold.r != r:
            raise ValueError("The supplied manifold dimensions do not match the data.")
        self.n_agents = len(self.clients)
        self.client_sizes = [int(x.shape[0]) for x in self.clients]

    def _get_data(
        self,
        client_id: int,
        X: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        Z = self.clients[client_id]
        if indices is not None:
            Z = Z.index_select(0, indices.detach().to(device="cpu", dtype=torch.long))
        return Z.to(device=X.device, dtype=X.dtype, non_blocking=True)

    def client_cost(
        self,
        client_id: int,
        X: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        Z = self._get_data(client_id, X, indices)
        return -(Z @ X).square().sum(dim=1).mean()

    def client_euclidean_grad(
        self,
        client_id: int,
        X: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        Z = self._get_data(client_id, X, indices)
        return -2.0 * (Z.transpose(0, 1) @ (Z @ X)) / Z.shape[0]

    def client_grad(
        self,
        client_id: int,
        X: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        egrad = self.client_euclidean_grad(client_id, X, indices)
        return self.manifold.project_tangent(X, egrad)

    def cost(self, X: torch.Tensor) -> torch.Tensor:
        values = [self.client_cost(i, X) for i in range(self.n_agents)]
        return torch.stack(values).mean()

    def grad(self, X: torch.Tensor) -> torch.Tensor:
        values = [self.client_grad(i, X) for i in range(self.n_agents)]
        return torch.stack(values).mean(dim=0)

    def grad_norm(self, X: torch.Tensor) -> torch.Tensor:
        return torch.linalg.matrix_norm(self.grad(X))

    def exact_optimal_cost(self, X: torch.Tensor) -> torch.Tensor:
        """Exact global optimum of the PCA objective from the top-r covariance eigenvalues.

        The MATLAB scripts use centralized Riemannian L-BFGS to estimate this
        reference value. For PCA, the global optimum is available in closed form:
        the columns of X are the top-r eigenvectors of the average covariance.
        """
        covariance = torch.zeros(
            self.d, self.d, dtype=X.dtype, device=X.device
        )
        for Z_cpu in self.clients:
            Z = Z_cpu.to(device=X.device, dtype=X.dtype, non_blocking=True)
            covariance.add_(
                (Z.transpose(0, 1) @ Z) / (Z.shape[0] * self.n_agents)
            )
        eigenvalues = torch.linalg.eigvalsh(0.5 * (covariance + covariance.T))
        return -eigenvalues[-self.r :].sum()
