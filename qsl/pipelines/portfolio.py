"""
Portfolio Optimization Pipeline - QAOA-based optimal asset allocation.

*** WARNING: DEMONSTRATION ONLY — uses random/synthetic data, not real finance ***
Solves the portfolio optimization problem using QAOA:
maximize expected return while minimizing risk (variance).
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PortfolioResult:
    """Result of portfolio optimization."""
    weights: np.ndarray  # Asset allocation weights
    expected_return: float
    risk: float  # Portfolio variance
    sharpe_ratio: float
    efficient_frontier: Optional[List[Tuple[float, float]]] = None


class PortfolioOptimizer:
    """
    Quantum portfolio optimization using QAOA.

    Formulates the Markowitz portfolio selection as a QUBO problem
    and solves it with QAOA.

    Args:
        returns: Expected returns for each asset
        covariance: Covariance matrix of asset returns
        risk_aversion: Risk aversion parameter (lambda in Markowitz)
        budget: Total budget constraint (number of assets to select)
    """

    def __init__(self,
                 returns: np.ndarray,
                 covariance: np.ndarray,
                 risk_aversion: float = 0.5,
                 budget: int = None):
        n = len(returns)

        if covariance.shape != (n, n):
            raise ValueError(
                f"Covariance matrix must be {n}x{n}, got {covariance.shape}"
            )

        self.returns = returns
        self.covariance = covariance
        self.risk_aversion = risk_aversion
        self.n_assets = n
        self.budget = budget or n // 2
        self._result: Optional[PortfolioResult] = None

    def _build_qubo_matrix(self) -> np.ndarray:
        """
        Build QUBO matrix for portfolio optimization.

        Minimize: -mu^T * x + gamma * x^T * Sigma * x + penalty * (sum x_i - B)^2

        Converts continuous weights to binary selection variables.
        """
        n = self.n_assets
        Q = np.zeros((n, n))

        for i in range(n):
            # Return term (negative because we minimize)
            Q[i, i] -= self.returns[i]

            # Risk term
            for j in range(n):
                if i == j:
                    Q[i, i] += self.risk_aversion * self.covariance[i, i]
                elif j > i:
                    Q[i, j] = self.risk_aversion * self.covariance[i, j] * 2

        # Budget constraint penalty: penalty * (sum_i x_i - B)^2
        # = penalty * (sum_i sum_j x_i x_j - 2*B*sum_i x_i + B^2)
        # Diagonal: penalty * (1 - 2*B), off-diagonal: penalty * 2
        penalty = 10.0
        B = self.budget
        for i in range(n):
            Q[i, i] += penalty * (1 - 2 * B)
            for j in range(i + 1, n):
                Q[i, j] += penalty * 2

        return Q

    def optimize(self, p: int = 1, verbose: bool = True) -> PortfolioResult:
        """
        Run portfolio optimization using QAOA.

        Args:
            p: Number of QAOA layers
            verbose: Print progress

        Returns:
            PortfolioResult with optimal weights
        """
        from ..algorithms.qaoa import QAOA

        Q = self._build_qubo_matrix()

        if verbose:
            print(f"\n  Portfolio Optimization (QAOA)")
            print(f"  Assets: {self.n_assets}, Budget: {self.budget}")
            print(f"  Expected Returns: {self.returns}")

        qaoa = QAOA(n_qubits=self.n_assets, cost_matrix=Q, p=p)
        params, energy = qaoa.optimize(maxiter=200, verbose=False)

        bitstring, _ = qaoa.get_optimal_bitstring()

        # Convert bitstring to weights
        weights = np.zeros(self.n_assets)
        selected_count = 0
        for i in range(self.n_assets):
            if (bitstring >> i) & 1:
                weights[i] = 1.0
                selected_count += 1

        # Normalize weights
        if selected_count > 0:
            weights /= selected_count

        # Compute portfolio metrics
        expected_return = np.dot(weights, self.returns)
        risk = np.sqrt(weights @ self.covariance @ weights)
        sharpe = expected_return / (risk + 1e-10)

        # Generate efficient frontier (simplified)
        frontier = self._compute_frontier()

        self._result = PortfolioResult(
            weights=weights,
            expected_return=expected_return,
            risk=risk,
            sharpe_ratio=sharpe,
            efficient_frontier=frontier,
        )

        if verbose:
            print(f"  Optimal allocation: {weights}")
            print(f"  Expected return: {expected_return:.4f}")
            print(f"  Risk (volatility): {risk:.4f}")
            print(f"  Sharpe ratio: {sharpe:.4f}")

        return self._result

    def _compute_frontier(self, n_points: int = 10) -> List[Tuple[float, float]]:
        """Compute efficient frontier by varying risk aversion."""
        frontier = []

        for gamma in np.linspace(0.1, 3.0, n_points):
            Q = np.zeros((self.n_assets, self.n_assets))
            penalty = 10.0

            for i in range(self.n_assets):
                Q[i, i] -= self.returns[i]
                Q[i, i] += gamma * self.covariance[i, i]
                Q[i, i] += penalty * (1 - 2 * self.budget)
                for j in range(i + 1, self.n_assets):
                    Q[i, j] = gamma * self.covariance[i, j] * 2 + penalty * 2

            relaxed = np.linalg.solve(
                Q + np.eye(self.n_assets) * 0.01,
                np.ones(self.n_assets)
            )
            w = np.maximum(relaxed, 0)
            w = w / (w.sum() + 1e-10)

            exp_ret = np.dot(w, self.returns)
            risk = np.sqrt(w @ self.covariance @ w)
            frontier.append((risk, exp_ret))

        return sorted(frontier, key=lambda x: x[0])

    @property
    def result(self) -> Optional[PortfolioResult]:
        return self._result

    @staticmethod
    def sample_problem(n_assets: int = 5, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a sample portfolio problem.

        Returns:
            (returns, covariance_matrix)
        """
        np.random.seed(seed)

        # Generate returns
        returns = np.random.uniform(0.02, 0.15, n_assets)

        # Generate correlation matrix
        A = np.random.randn(n_assets, n_assets)
        corr = A @ A.T
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)

        # Generate volatilities and covariance
        vols = np.random.uniform(0.1, 0.3, n_assets)
        cov = np.outer(vols, vols) * corr

        np.random.seed(None)
        return returns, cov
