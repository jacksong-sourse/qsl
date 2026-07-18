import numpy as np
import random
from typing import Callable, Optional
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class QAOA:
    """
    Quantum Approximate Optimization Algorithm.

    Solves combinatorial optimization problems by encoding the cost function
    into a Hamiltonian and using parameterized quantum circuits.

    Args:
        n_qubits: Number of qubits (variables)
        cost_matrix: Symmetric n x n matrix defining the problem.
            - encoding="ising": Cost = sum_i Q[i][i] * s_i
                                + sum_{i<j} Q[i][j] * s_i * s_j,
              with s_i in {-1, +1} (Ising spins)
            - encoding="qubo": Cost = sum_i Q[i][i] * x_i
                               + sum_{i<j} Q[i][j] * x_i * x_j,
              with x_i in {0, 1} (binary variables)
        p: Number of QAOA layers (depth)
        encoding: "ising" (default) or "qubo". QUBO problems are
            converted to Ising form via x_i = (1 - s_i) / 2 internally,
            so the cost Hamiltonian, expectation values and reported
            energies are always consistent.
    """

    def __init__(self, n_qubits: int, cost_matrix: np.ndarray, p: int = 1,
                 encoding: str = "ising"):
        if cost_matrix.shape != (n_qubits, n_qubits):
            raise ValueError(
                f"cost_matrix must be {n_qubits}x{n_qubits}, "
                f"got {cost_matrix.shape}"
            )
        if p < 1:
            raise ValueError(f"p must be >= 1, got {p}")
        if encoding not in ("ising", "qubo"):
            raise ValueError(
                f"encoding must be 'ising' or 'qubo', got '{encoding}'"
            )
        self.n_qubits = n_qubits
        self.cost_matrix = cost_matrix
        self.encoding = encoding
        self.p = p
        self.N = 1 << n_qubits
        self._optimal_params = None
        self._optimal_value = None
        self._optimal_bitstring = None

        # 统一转换为 Ising 形式: E = offset + sum_i h_i s_i + sum_{i<j} J_ij s_i s_j
        if encoding == "qubo":
            self._h, self._J, self._offset = self._qubo_to_ising(cost_matrix)
        else:
            self._h = np.diag(cost_matrix).astype(float).copy()
            self._J = np.array(cost_matrix, dtype=float, copy=True)
            self._offset = 0.0

    @staticmethod
    def _qubo_to_ising(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """
        将 QUBO 矩阵转换为 Ising 参数 (h, J, offset)。

        代入 x_i = (1 - s_i)/2 (x in {0,1}, s in {-1,+1}):
            E_qubo = sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j
                   = offset + sum_i h_i s_i + sum_{i<j} J_ij s_i s_j

        其中:
            J_ij   = Q_ij / 4
            h_i    = -Q_ii/2 - sum_{j!=i} Q_ij/4   (Q 对称)
            offset = sum_i Q_ii/2 + sum_{i<j} Q_ij/4
        """
        n = Q.shape[0]
        J = np.zeros((n, n))
        h = np.zeros(n)
        offset = 0.0
        for i in range(n):
            h[i] -= Q[i, i] / 2.0
            offset += Q[i, i] / 2.0
            for j in range(i + 1, n):
                J[i, j] = Q[i, j] / 4.0
                h[i] -= Q[i, j] / 4.0
                h[j] -= Q[i, j] / 4.0
                offset += Q[i, j] / 4.0
        return h, J, offset

    def _cost(self, bitstring: int) -> float:
        """
        Compute cost of a classical bitstring.

        E = offset + sum_i h_i * s_i + sum_{i<j} J_ij * s_i * s_j
        where s_i = (-1)^(bit_i) in {-1, +1}

        对于 encoding="qubo", 该值与原始 QUBO 目标函数完全一致
        (变量转换 x_i = (1 - s_i)/2 已包含在 h/J/offset 中)。
        """
        cost = self._offset
        for i in range(self.n_qubits):
            si = -1.0 if (bitstring >> i) & 1 else 1.0
            cost += self._h[i] * si
            for j in range(i + 1, self.n_qubits):
                sj = -1.0 if (bitstring >> j) & 1 else 1.0
                cost += self._J[i, j] * si * sj
        return cost

    def _apply_cost_layer(self, state: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply e^(-i*gamma*H_C) to the state vector (fully vectorized).

        Uses ZZ interactions for off-diagonal terms and Z rotations for diagonal.
        For each ZZ term J_{ij} * Z_i * Z_j:
            Apply phase = exp(-i*gamma*J_{ij}) if Z_i == Z_j else exp(i*gamma*J_{ij})
        For each Z term h_i * Z_i:
            Apply phase = exp(-i*gamma*h_i) if Z_i == +1 else exp(i*gamma*h_i)
        """
        indices = np.arange(self.N, dtype=np.int64)
        phases = np.ones(self.N, dtype=complex)

        for i in range(self.n_qubits):
            mask_i = 1 << i
            # Z_i term
            coeff = self._h[i]
            if abs(coeff) >= 1e-12:
                zi = np.where((indices & mask_i) != 0, -1.0, 1.0)
                phases *= np.exp(-1j * gamma * coeff * zi)

            for j in range(i + 1, self.n_qubits):
                mask_j = 1 << j
                coeff = self._J[i, j]
                if abs(coeff) >= 1e-12:
                    zj = np.where((indices & mask_j) != 0, -1.0, 1.0)
                    zi = np.where((indices & mask_i) != 0, -1.0, 1.0)
                    zz = zi * zj
                    phases *= np.exp(-1j * gamma * coeff * zz)

        state *= phases
        return state

    def _apply_mixer_layer(self, state: np.ndarray, beta: float) -> np.ndarray:
        """
        Apply mixer Hamiltonian e^(-i*beta*H_M) = prod_i RX_i(2*beta) (fully vectorized).

        RX(2*beta)|0> = cos(beta)|0> - i*sin(beta)|1>
        RX(2*beta)|1> = -i*sin(beta)|0> + cos(beta)|1>
        """
        c = np.cos(beta)
        s = -1j * np.sin(beta)
        indices = np.arange(self.N, dtype=np.int64)

        for i in range(self.n_qubits):
            mask_i = 1 << i
            bit_zero = (indices & mask_i) == 0
            bit_one = ~bit_zero
            a0 = state[bit_zero].copy()
            a1 = state[bit_one].copy()
            state[bit_zero] = c * a0 + s * a1
            state[bit_one] = s * a0 + c * a1
        return state

    def _simulate_circuit(self, params: np.ndarray) -> np.ndarray:
        """
        Simulate the full QAOA circuit.

        params = [gamma_0, beta_0, gamma_1, beta_1, ..., gamma_{p-1}, beta_{p-1}]
        """
        # Check for NaN in parameters
        if np.any(np.isnan(params)):
            raise ValueError("Parameters contain NaN values")

        # Initial state: |+...+> = H^⊗n |0...0>
        state = np.ones(self.N, dtype=complex) / np.sqrt(self.N)

        for layer in range(self.p):
            gamma = params[2 * layer]
            beta = params[2 * layer + 1]

            # Cost layer
            state = self._apply_cost_layer(state, gamma)
            # Mixer layer
            state = self._apply_mixer_layer(state, beta)

        # Normalize to prevent floating-point drift
        norm = np.sqrt(np.sum(np.abs(state) ** 2))
        if norm > 0:
            state /= norm

        return state

    def _expectation_value(self, params: np.ndarray) -> float:
        """
        Compute ⟨H_C⟩ = expectation value of the cost Hamiltonian (fully vectorized).

        E = sum_i h_i * ⟨Z_i⟩ + sum_{i<j} J_{ij} * ⟨Z_i Z_j⟩
        """
        state = self._simulate_circuit(params)
        probs = np.abs(state) ** 2
        indices = np.arange(self.N, dtype=np.int64)

        z_exp = np.zeros(self.n_qubits)
        zz_exp = np.zeros((self.n_qubits, self.n_qubits))

        for i in range(self.n_qubits):
            mask_i = 1 << i
            zi = np.where((indices & mask_i) != 0, -1.0, 1.0)
            z_exp[i] = np.sum(probs * zi)
            for j in range(i + 1, self.n_qubits):
                mask_j = 1 << j
                zj = np.where((indices & mask_j) != 0, -1.0, 1.0)
                zz_exp[i, j] = np.sum(probs * zi * zj)

        energy = self._offset
        for i in range(self.n_qubits):
            energy += self._h[i] * z_exp[i]
            for j in range(i + 1, self.n_qubits):
                energy += self._J[i, j] * zz_exp[i, j]

        if np.isnan(energy):
            return 0.0

        return float(energy)

    def optimize(self, maxiter: int = 200, verbose: bool = False) -> tuple[np.ndarray, float]:
        """
        Run QAOA optimization to find optimal parameters.

        Returns:
            (optimal_params, optimal_energy)
        """
        # Initial parameters (random small values)
        initial_params = np.random.uniform(-np.pi / 4, np.pi / 4, 2 * self.p)

        if SCIPY_AVAILABLE:
            result = minimize(
                self._expectation_value,
                initial_params,
                method='COBYLA',
                options={'maxiter': maxiter, 'disp': verbose}
            )
            optimal_params = result.x
            optimal_energy = result.fun
        else:
            # Simple gradient descent fallback
            optimal_params = self._gradient_descent(initial_params, maxiter, verbose)
            optimal_energy = self._expectation_value(optimal_params)

        self._optimal_params = optimal_params
        self._optimal_value = optimal_energy

        # Find best bitstring by maximum probability
        final_state = self._simulate_circuit(optimal_params)
        probs = np.abs(final_state) ** 2
        best_idx = int(np.argmax(probs))
        self._optimal_bitstring = best_idx

        return optimal_params, optimal_energy

    def _gradient_descent(self, initial_params: np.ndarray,
                          maxiter: int, verbose: bool) -> np.ndarray:
        """Simple gradient descent with finite differences as fallback."""
        params = initial_params.copy().astype(np.float64)
        lr = 0.1

        for iteration in range(maxiter):
            # Finite difference gradient
            eps = 1e-6
            grad = np.zeros_like(params)
            e0 = self._expectation_value(params)

            if np.isnan(e0):
                # If expectation is NaN, reset with small random perturbation
                params = np.random.uniform(-np.pi / 4, np.pi / 4, len(params))
                lr = 0.1
                continue

            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += eps
                e_plus = self._expectation_value(params_plus)
                grad[i] = (e_plus - e0) / eps

            params -= lr * grad

            # Decay learning rate
            lr *= 0.99

            if verbose and iteration % 20 == 0:
                print(f"  Iter {iteration}: energy = {e0:.6f}")

        return params

    def get_optimal_bitstring(self) -> tuple[int, float]:
        """
        Get the best solution found.

        Returns:
            (bitstring_as_int, corresponding_cost)
        """
        if self._optimal_bitstring is None:
            raise RuntimeError("Must call optimize() first")
        return self._optimal_bitstring, self._cost(self._optimal_bitstring)

    def sample_solutions(self, n_samples: int = 10) -> list[tuple[int, float]]:
        """
        Sample from the final QAOA state distribution.

        Returns list of (bitstring_as_int, cost) pairs sorted by cost.
        """
        if self._optimal_params is None:
            raise RuntimeError("Must call optimize() first")

        state = self._simulate_circuit(self._optimal_params)
        probs = np.abs(state) ** 2

        # Get top n_samples by probability
        indices = np.argsort(probs)[::-1][:n_samples]
        solutions = [(int(i), self._cost(int(i))) for i in indices]
        return sorted(solutions, key=lambda x: x[1])

    @property
    def optimal_energy(self) -> Optional[float]:
        """Optimal energy found (may be None before optimization)."""
        return self._optimal_value

    @property
    def optimal_bitstring(self) -> Optional[int]:
        """Optimal bitstring found as integer (may be None before optimization)."""
        return self._optimal_bitstring

    @staticmethod
    def maxcut_cost_matrix(adjacency_matrix: np.ndarray) -> np.ndarray:
        """
        Convert a MaxCut adjacency matrix to QAOA cost matrix.

        MaxCut: maximize sum_{(i,j) in E} (1 - s_i*s_j)/2
        Equivalent to: minimize sum_{(i,j) in E} s_i*s_j
        So cost_matrix[i][j] = 1 for each edge.
        """
        return adjacency_matrix

    def __repr__(self) -> str:
        return f"QAOA(n_qubits={self.n_qubits}, p={self.p})"
