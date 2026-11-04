import unittest

import torch

from rfedags import StiefelManifold, StiefelPCAProblem, run_rfedags
from rfedags.data import make_synthetic_pca_data


class StiefelCoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.dtype = torch.float64
        self.manifold = StiefelManifold(12, 3)
        self.X = self.manifold.random_point(dtype=self.dtype)

    def test_qr_retraction_and_tangent_projection(self):
        ambient = torch.randn_like(self.X)
        tangent = self.manifold.project_tangent(self.X, ambient)
        self.assertLess(self.manifold.tangent_error(self.X, tangent).item(), 1e-10)
        Y = self.manifold.retract(self.X, 0.1 * tangent)
        self.assertLess(self.manifold.orthogonality_error(Y).item(), 1e-10)

    def test_projection_transport(self):
        ambient = torch.randn_like(self.X)
        Y = self.manifold.retract(self.X, 0.05 * self.manifold.project_tangent(self.X, ambient))
        transported = self.manifold.transport(self.X, Y, ambient)
        self.assertLess(self.manifold.tangent_error(Y, transported).item(), 1e-10)

    def test_inverse_qr_retraction(self):
        tangent = self.manifold.project_tangent(self.X, torch.randn_like(self.X))
        tangent = 0.05 * tangent
        Y = self.manifold.retract(self.X, tangent)
        recovered = self.manifold.inverse_qr_retraction(self.X, Y)
        reconstructed = self.manifold.retract(self.X, recovered)
        self.assertLess(torch.linalg.matrix_norm(reconstructed - Y).item(), 1e-9)
        self.assertLess(self.manifold.tangent_error(self.X, recovered).item(), 1e-9)

    def test_rfedags_smoke(self):
        clients = make_synthetic_pca_data(
            n_agents=4, samples_per_agent=24, d=12, seed=9
        )
        problem = StiefelPCAProblem(clients, r=3)
        X0 = problem.manifold.random_point(dtype=self.dtype)
        result = run_rfedags(
            problem,
            X0,
            rounds=4,
            local_steps=2,
            batch_size=8,
            learning_rate=1e-3,
            probabilities=torch.ones(4, dtype=torch.float64),
            participation_mode="frequency",
            check_every=1,
            seed=11,
        )
        self.assertEqual(result.final_point.shape, (12, 3))
        self.assertEqual(len(result.iterations), 5)
        self.assertTrue(torch.isfinite(result.final_point).all())
        self.assertLess(problem.manifold.orthogonality_error(result.final_point).item(), 1e-9)


if __name__ == "__main__":
    unittest.main()
