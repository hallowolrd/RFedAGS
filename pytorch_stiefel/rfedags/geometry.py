from __future__ import annotations

import torch


def sym(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-2, -1))


class StiefelManifold:
    """Stiefel manifold St(r, d) = {X in R^(d x r): X.T @ X = I}.

    This matches the Manopt stiefelfactory used by the MATLAB implementation:
    Euclidean/Frobenius metric, QR retraction, and projection vector transport.
    """

    def __init__(self, d: int, r: int):
        if d < r or r < 1:
            raise ValueError(f"Require d >= r >= 1, got d={d}, r={r}.")
        self.d = int(d)
        self.r = int(r)

    def random_point(
        self,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        A = torch.randn(
            self.d, self.r, dtype=dtype, device=device, generator=generator
        )
        return self.qr_unique(A)

    @staticmethod
    def qr_unique(A: torch.Tensor) -> torch.Tensor:
        """Thin QR with positive diagonal in R, matching Manopt qr_unique."""
        Q, R = torch.linalg.qr(A, mode="reduced")
        diagonal = torch.diagonal(R, dim1=-2, dim2=-1)
        signs = torch.where(diagonal >= 0, 1.0, -1.0).to(A.dtype)
        return Q * signs.unsqueeze(-2)

    def retract(
        self, X: torch.Tensor, eta: torch.Tensor, scale: float = 1.0
    ) -> torch.Tensor:
        """First-order QR retraction R_X(scale * eta)."""
        return self.qr_unique(X + scale * eta)

    @staticmethod
    def project_tangent(X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """Orthogonal projection onto T_X St(r,d) for the embedded metric."""
        return U - X @ sym(X.transpose(-2, -1) @ U)

    def transport(
        self, X_from: torch.Tensor, X_to: torch.Tensor, U: torch.Tensor
    ) -> torch.Tensor:
        """Projection transport P_{X_to}(U), exactly as Manopt M.transp."""
        del X_from  # Projection transport depends only on the destination point.
        return self.project_tangent(X_to, U)

    @staticmethod
    def polar_projection(A: torch.Tensor) -> torch.Tensor:
        """Nearest Stiefel point in Frobenius norm, used by RFedProj baselines."""
        U, _, Vh = torch.linalg.svd(A, full_matrices=False)
        return U @ Vh

    def inverse_qr_retraction(
        self, X: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        """Inverse of the QR retraction used by the RFedAvg/RFedSVRG baselines.

        Solve for upper-triangular R in
            (X.T Y) R + R.T (X.T Y).T = 2 I,
        then return U = Y R - X. This is the inverse QR retraction implemented
        in Manopt's Stiefel factory.
        """
        A = X.transpose(-2, -1) @ Y
        r = self.r
        upper = torch.triu_indices(r, r, device=X.device)
        count = upper.shape[1]

        basis = torch.zeros(count, r, r, dtype=X.dtype, device=X.device)
        basis[torch.arange(count, device=X.device), upper[0], upper[1]] = 1.0
        A_basis = torch.einsum("ij,kjl->kil", A, basis)
        sym_A_basis = 0.5 * (A_basis + A_basis.transpose(-2, -1))
        system = sym_A_basis[:, upper[0], upper[1]].transpose(0, 1)
        rhs = torch.eye(r, dtype=X.dtype, device=X.device)[upper[0], upper[1]]

        try:
            coeff = torch.linalg.solve(system, rhs)
        except RuntimeError:
            coeff = torch.linalg.lstsq(system, rhs.unsqueeze(1)).solution.squeeze(1)

        R = torch.einsum("k,kij->ij", coeff, basis)
        return Y @ R - X

    def orthogonality_error(self, X: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(self.r, dtype=X.dtype, device=X.device)
        return torch.linalg.matrix_norm(X.transpose(-2, -1) @ X - identity)

    def tangent_error(self, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        return torch.linalg.matrix_norm(
            X.transpose(-2, -1) @ U + U.transpose(-2, -1) @ X
        )
