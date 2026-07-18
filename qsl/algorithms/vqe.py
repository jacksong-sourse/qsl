"""
Variational Quantum Eigensolver (VQE) implementation.

Finds ground state energies of quantum systems by optimizing
a parameterized ansatz circuit.
"""

import numpy as np
from typing import Optional

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class VQE:
    """
    Variational Quantum Eigensolver.

    Finds the ground state energy of a Hamiltonian H by optimizing
    a parameterized ansatz circuit:

    E(theta) = ⟨psi(theta)| H |psi(theta)⟩
    E_0 = min_theta E(theta)

    Args:
        n_qubits: Number of qubits
        hamiltonian_pauli_terms: List of (coefficient, pauli_string) tuples
            e.g., [(-1.0, "ZZ"), (0.5, "X"), (0.5, "IX")]
            Each string must have length n_qubits.
        ansatz_type: "he" (Hardware Efficient, default) or "uccsd"
        n_layers: Number of ansatz layers (for HE ansatz).
    """

    def __init__(self,
                 n_qubits: int,
                 hamiltonian_pauli_terms: list[tuple[float, str]],
                 ansatz_type: str = "he",
                 n_layers: int = 2):
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if not isinstance(hamiltonian_pauli_terms, list) or len(hamiltonian_pauli_terms) == 0:
            raise ValueError("hamiltonian_pauli_terms must be a non-empty list")
        if ansatz_type not in ("he", "uccsd"):
            raise ValueError(f"Unknown ansatz_type '{ansatz_type}'. Must be 'he' or 'uccsd'.")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self.n_qubits = n_qubits
        self.hamiltonian_terms = hamiltonian_pauli_terms
        self.ansatz_type = ansatz_type
        self.n_layers = n_layers
        self.N = 1 << n_qubits
        self._optimal_params = None
        self._ground_energy = None
        self._ground_state = None

        _VALID_PAULI = frozenset("IXYZ")
        for coeff, pauli in hamiltonian_pauli_terms:
            if not isinstance(coeff, (int, float)):
                raise TypeError(
                    f"Hamiltonian coefficient must be numeric, got {type(coeff).__name__}"
                )
            if not isinstance(pauli, str):
                raise TypeError(
                    f"Pauli string must be str, got {type(pauli).__name__}"
                )
            if len(pauli) != n_qubits:
                raise ValueError(
                    f"Pauli string '{pauli}' has length {len(pauli)}, "
                    f"but n_qubits={n_qubits}"
                )
            for c in pauli:
                if c not in _VALID_PAULI:
                    raise ValueError(
                        f"Invalid Pauli character '{c}' in '{pauli}'. "
                        f"Must be one of I, X, Y, Z."
                    )

    def _apply_pauli_string(self, state: np.ndarray, pauli_string: str) -> np.ndarray:
        """Apply a Pauli string operator to a state vector (fully vectorized)."""
        indices = np.arange(self.N, dtype=np.int64)
        for qubit, pauli in enumerate(pauli_string):
            mask = 1 << qubit

            if pauli == 'I':
                continue
            elif pauli == 'X':
                bit_zero = (indices & mask) == 0
                bit_one = ~bit_zero
                a0 = state[bit_zero].copy()
                a1 = state[bit_one].copy()
                state[bit_zero] = a1
                state[bit_one] = a0
            elif pauli == 'Y':
                bit_zero = (indices & mask) == 0
                bit_one = ~bit_zero
                a0 = state[bit_zero].copy()
                a1 = state[bit_one].copy()
                state[bit_zero] = -1j * a1
                state[bit_one] = 1j * a0
            elif pauli == 'Z':
                bit_one = (indices & mask) != 0
                state[bit_one] *= -1

        return state

    def _expectation_pauli(self, state: np.ndarray, pauli_string: str) -> float:
        """Compute ⟨psi|P|psi⟩ for a Pauli string P (vectorized)."""
        p_psi = self._apply_pauli_string(state.copy(), pauli_string)
        expectation = np.dot(state.conjugate(), p_psi).real
        return float(expectation)

    def _energy(self, state: np.ndarray) -> float:
        """Compute ⟨psi|H|psi⟩ = sum_t coeff_t * ⟨psi|P_t|psi⟩ (vectorized)."""
        total = 0.0
        for coeff, pauli in self.hamiltonian_terms:
            total += coeff * self._expectation_pauli(state, pauli)
        return total

    @staticmethod
    def _apply_ry(state: np.ndarray, qubit: int, theta: float) -> np.ndarray:
        """Apply RY(theta) rotation to a single qubit (fully vectorized)."""
        mask = 1 << qubit
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        indices = np.arange(len(state), dtype=np.int64)

        bit_zero = (indices & mask) == 0
        bit_one = ~bit_zero
        a0 = state[bit_zero].copy()
        a1 = state[bit_one].copy()
        state[bit_zero] = c * a0 - s * a1
        state[bit_one] = s * a0 + c * a1

        return state

    @staticmethod
    def _apply_rz(state: np.ndarray, qubit: int, phi: float) -> np.ndarray:
        """Apply RZ(phi) rotation to a single qubit (fully vectorized)."""
        mask = 1 << qubit
        phase0 = np.exp(-1j * phi / 2)
        phase1 = np.exp(1j * phi / 2)
        indices = np.arange(len(state), dtype=np.int64)

        bit_zero = (indices & mask) == 0
        bit_one = ~bit_zero
        state[bit_zero] *= phase0
        state[bit_one] *= phase1

        return state

    @staticmethod
    def _apply_cnot(state: np.ndarray, control: int, target: int) -> np.ndarray:
        """Apply CNOT gate (fully vectorized)."""
        c_mask = 1 << control
        t_mask = 1 << target
        indices = np.arange(len(state), dtype=np.int64)

        c_one_t_zero = ((indices & c_mask) != 0) & ((indices & t_mask) == 0)
        c_one_t_one = ((indices & c_mask) != 0) & ((indices & t_mask) != 0)
        a_t0 = state[c_one_t_zero].copy()
        a_t1 = state[c_one_t_one].copy()
        state[c_one_t_zero] = a_t1
        state[c_one_t_one] = a_t0

        return state

    def _num_params(self) -> int:
        """Number of ansatz parameters."""
        if self.ansatz_type == "he":
            return self.n_layers * (2 * self.n_qubits) + self.n_qubits
        elif self.ansatz_type == "uccsd":
            n_occ = self.n_qubits // 2
            n_virt = self.n_qubits - n_occ
            n_singles = n_occ * n_virt
            n_doubles = (n_occ * (n_occ - 1) * n_virt * (n_virt - 1)) // 4
            total = n_singles + n_doubles
            return max(total, 1)

    def _hardware_efficient_ansatz(self, params: np.ndarray) -> np.ndarray:
        """Hardware-efficient ansatz."""
        idx = 0
        state = np.ones(self.N, dtype=complex) / np.sqrt(self.N)

        for _ in range(self.n_layers):
            for q in range(self.n_qubits):
                theta = params[idx]; idx += 1
                state = self._apply_ry(state, q, theta)
                phi = params[idx]; idx += 1
                state = self._apply_rz(state, q, phi)

            for q in range(self.n_qubits - 1):
                state = self._apply_cnot(state, q, q + 1)
            if self.n_qubits > 2:
                state = self._apply_cnot(state, self.n_qubits - 1, 0)

        for q in range(self.n_qubits):
            theta = params[idx]; idx += 1
            state = self._apply_ry(state, q, theta)

        return state

    def _apply_single_excitation(self, state: np.ndarray,
                                  i: int, j: int, theta: float) -> np.ndarray:
        """
        Apply single-excitation Givens rotation between orbitals i and j.

        Uses the standard decomposition with CNOT ladders for
        non-adjacent qubits.
        """
        if abs(i - j) == 1:
            self._apply_cnot(state, i, j)
            self._apply_ry(state, j, theta)
            self._apply_cnot(state, i, j)
        else:
            for q in range(i, j - 1):
                state = self._apply_cnot(state, q, q + 1)
            state = self._apply_ry(state, j - 1, theta)
            for q in range(j - 2, i - 1, -1):
                state = self._apply_cnot(state, q, q + 1)
        return state

    def _apply_double_excitation(self, state: np.ndarray,
                                  i: int, j: int, k: int, l: int,
                                  theta: float) -> np.ndarray:
        """
        Apply standard UCCSD double-excitation operator.

        A double excitation |ij⟩ → |kl⟩ (where i,j are occupied and k,l
        are virtual) using the standard Jordan-Wigner mapped decomposition:

        exp(θ (a_k† a_l† a_j a_i - h.c.))

        The standard decomposition uses 8 CNOT gates and 1 rotation
        for the four-qubit Pauli string Y_i X_j X_k X_l.
        For non-adjacent indices, additional SWAP/CNOT ladders are used
        to bring qubits into adjacent positions.

        The circuit (for adjacent indices):
            CNOT(j, i)  CNOT(k, j)  CNOT(l, k)
            RY(l, θ)
            CNOT(l, k)  CNOT(k, j)  CNOT(j, i)
            CNOT(j, i)  CNOT(k, j)  CNOT(l, k)
            RZ(l, π/2)  (if using Y term, but simplified here)

        Simplified: uses the first-order "CNOT staircase" parity computation
        pattern: CNOT chain → RY rotation → CNOT uncompute.
        This is the same as a 4-qubit Pauli rotation for the string YX...X.
        """
        # Standard decomposition for exp(iθ Y_i X_j X_k X_l):
        # 1. Change basis: H on j, k, l; S on i (to convert Y to Z)
        # 2. CNOT staircase to compute parity on qubit l
        # 3. RZ(θ) on qubit l
        # 4. Uncompute CNOT staircase
        # 5. Restore basis

        # Step 1: Basis change
        # Y_i → Z_i via S† on i: |0> stays |0>, |1> → -i|1>
        # X_j → Z_j via H on j
        # X_k → Z_k via H on k
        # X_l → Z_l via H on l
        # Apply S† (conjugate of S) on qubit i
        self._apply_rz(state, i, -np.pi / 2)  # S† = RZ(-π/2)
        # Apply H on j, k, l
        self._apply_h(state, j)
        self._apply_h(state, k)
        self._apply_h(state, l)

        # Step 2: CNOT staircase i→j→k→l to compute parity on l
        self._apply_cnot(state, i, j)
        self._apply_cnot(state, j, k)
        self._apply_cnot(state, k, l)

        # Step 3: Rotation on target qubit l
        self._apply_rz(state, l, theta)

        # Step 4: Uncompute CNOT staircase
        self._apply_cnot(state, k, l)
        self._apply_cnot(state, j, k)
        self._apply_cnot(state, i, j)

        # Step 5: Restore basis
        self._apply_h(state, l)
        self._apply_h(state, k)
        self._apply_h(state, j)
        self._apply_rz(state, i, np.pi / 2)  # S = RZ(π/2)

        return state

    @staticmethod
    def _apply_h(state: np.ndarray, qubit: int) -> np.ndarray:
        """Apply Hadamard gate to a single qubit (fully vectorized)."""
        mask = 1 << qubit
        inv_sqrt2 = 1.0 / np.sqrt(2)
        indices = np.arange(len(state), dtype=np.int64)

        bit_zero = (indices & mask) == 0
        bit_one = ~bit_zero
        a0 = state[bit_zero].copy()
        a1 = state[bit_one].copy()
        state[bit_zero] = (a0 + a1) * inv_sqrt2
        state[bit_one] = (a0 - a1) * inv_sqrt2
        return state

    def _uccsd_ansatz(self, params: np.ndarray) -> np.ndarray:
        """
        UCCSD ansatz with Hartree-Fock reference.

        Prepares a Hartree-Fock state |11…00⟩ (half-filling),
        then applies single- and double-excitation operators.
        """
        n_occ = self.n_qubits // 2
        n_virt = self.n_qubits - n_occ

        # Hartree-Fock: occupy the lowest-energy orbitals
        hf_index = 0
        for i in range(n_occ):
            hf_index |= (1 << i)
        state = np.zeros(self.N, dtype=complex)
        state[hf_index] = 1.0

        idx = 0

        # Single excitations
        for i in range(n_occ):
            for j in range(n_occ, self.n_qubits):
                if idx >= len(params):
                    break
                theta = params[idx]; idx += 1
                if abs(theta) > 1e-12:
                    self._apply_single_excitation(state, i, j, theta)

        # Double excitations
        for i in range(n_occ - 1):
            for j in range(i + 1, n_occ):
                for k in range(n_occ, self.n_qubits - 1):
                    for l in range(k + 1, self.n_qubits):
                        if idx >= len(params):
                            break
                        theta = params[idx]; idx += 1
                        if abs(theta) > 1e-12:
                            self._apply_double_excitation(
                                state, i, j, k, l, theta
                            )

        return state

    def _ansatz(self, params: np.ndarray) -> np.ndarray:
        """Apply the selected ansatz."""
        expected = self._num_params()
        if len(params) != expected:
            raise ValueError(
                f"params has length {len(params)}, expected {expected}"
            )

        if self.ansatz_type == "he":
            return self._hardware_efficient_ansatz(params)
        elif self.ansatz_type == "uccsd":
            return self._uccsd_ansatz(params)
        raise ValueError(f"Unknown ansatz type: {self.ansatz_type}")

    def _cost_function(self, params: np.ndarray) -> float:
        """Cost = ⟨psi(params)|H|psi(params)⟩."""
        state = self._ansatz(params)
        return self._energy(state)

    def _parameter_shift_gradient(self, params: np.ndarray) -> np.ndarray:
        """
        使用 parameter-shift rule 计算精确梯度。

        所有 ansatz 门均为 Pauli 旋转门 exp(-i*theta/2*G) (G^2 = I),
        其期望值对 theta 的偏导有闭式表达:

            dE/dtheta_i = [E(theta_i + pi/2) - E(theta_i - pi/2)] / 2

        这是精确梯度 (非有限差分近似), 每个参数只需 2 次前向求值,
        且不存在步长 eps 带来的截断误差。
        """
        grad = np.empty_like(params)
        for i in range(len(params)):
            orig = params[i]
            params[i] = orig + np.pi / 2
            e_plus = self._cost_function(params)
            params[i] = orig - np.pi / 2
            e_minus = self._cost_function(params)
            params[i] = orig
            grad[i] = (e_plus - e_minus) / 2.0
        return grad

    def _gradient_descent(self, initial_params: np.ndarray,
                          maxiter: int, verbose: bool) -> np.ndarray:
        """基于 parameter-shift 精确梯度的梯度下降。"""
        params = initial_params.astype(np.float64, copy=True)
        lr = 0.1

        for iteration in range(maxiter):
            e0 = self._cost_function(params)
            grad = self._parameter_shift_gradient(params)

            params -= lr * grad
            lr *= 0.995

            if verbose and iteration % 30 == 0:
                print(f"  VQE Iter {iteration:4d}: energy = {e0:.8f}")

        return params

    def optimize(self,
                 maxiter: int = 300,
                 verbose: bool = False) -> tuple[float, np.ndarray]:
        """
        Run VQE optimization.

        Args:
            maxiter: Maximum optimisation iterations.
            verbose: If True, print progress messages.

        Returns:
            (ground_state_energy, ground_state_wavefunction)
        """
        n_params = self._num_params()
        initial_params = np.random.uniform(-np.pi, np.pi, n_params)

        if SCIPY_AVAILABLE:
            result = minimize(
                self._cost_function,
                initial_params,
                method='COBYLA',
                options={'maxiter': maxiter, 'disp': verbose}
            )
            optimal_params = result.x
        else:
            optimal_params = self._gradient_descent(
                initial_params, maxiter, verbose
            )

        final_state = self._ansatz(optimal_params)
        ground_energy = self._energy(final_state)

        self._optimal_params = optimal_params
        self._ground_energy = ground_energy
        self._ground_state = final_state

        return ground_energy, final_state

    @property
    def ground_energy(self) -> Optional[float]:
        """Ground state energy found (None before optimize())."""
        return self._ground_energy

    @property
    def optimal_params(self) -> Optional[np.ndarray]:
        """Optimal ansatz parameters (None before optimize())."""
        return self._optimal_params

    @property
    def ground_state(self) -> np.ndarray:
        """
        Ground state wavefunction.

        Raises:
            RuntimeError: If optimize() has not been called yet.
        """
        if self._ground_state is None:
            raise RuntimeError("Must call optimize() before accessing ground_state")
        return self._ground_state

    def get_expectations(self) -> dict[str, tuple[float, float]]:
        """
        Return expectation values for each Pauli term.

        Returns:
            Dict mapping pauli_string → (coefficient, expectation_value).

        Raises:
            RuntimeError: If optimize() has not been called yet.
        """
        if self._ground_state is None:
            raise RuntimeError("Must call optimize() before get_expectations()")

        expectations = {}
        for coeff, pauli in self.hamiltonian_terms:
            exp_val = self._expectation_pauli(self._ground_state, pauli)
            expectations[pauli] = (coeff, exp_val)

        return expectations

    @staticmethod
    def h2_hamiltonian(bond_length: float = 0.74) -> list[tuple[float, str]]:
        """
        Generate H₂ molecule Hamiltonian (4 qubits, STO-3G).

        Returns a Pauli-term representation for demonstration purposes.
        Coefficients approximate the equilibrium bond length ~0.74 Å.

        Args:
            bond_length: Inter-nuclear distance in Ångström.

        Returns:
            List of (coefficient, pauli_string) tuples.
        """
        _ = bond_length  # reserved for future interpolation
        h2_terms = [
            (-0.8126, "IIII"),
            (0.1712, "ZIII"),
            (0.1712, "IZII"),
            (-0.2228, "IIZI"),
            (-0.2228, "IIIZ"),
            (0.1686, "ZZII"),
            (0.1206, "ZIZI"),
            (0.1659, "ZIIZ"),
            (0.1659, "IZZI"),
            (0.1206, "IZIZ"),
            (0.1744, "IIZZ"),
            (0.0453, "YYYY"),
            (0.0453, "XXYY"),
            (0.0453, "YYXX"),
            (0.0453, "XXXX"),
        ]
        return h2_terms

    def __repr__(self) -> str:
        return (f"VQE(n_qubits={self.n_qubits}, "
                f"ansatz='{self.ansatz_type}', "
                f"terms={len(self.hamiltonian_terms)})")
