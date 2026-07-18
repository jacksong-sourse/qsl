"""Quantum Fourier Transform implementation."""

import numpy as np


class QuantumFourierTransform:
    """
    Quantum Fourier Transform on n_qubits.

    QFT|j> = 1/sqrt(N) * sum_{k=0}^{N-1} exp(2*pi*i*j*k/N) |k>
    where N = 2^n_qubits

    Circuit: H gates + controlled phase rotations R_k
    Complexity: O(n^2) gates, O(n) depth
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.N = 1 << n_qubits

    def build_circuit(self) -> list:
        """
        Build QFT circuit as a list of gate operations.

        Returns list of gate operations forming the full QFT circuit.
        """
        circuit = []
        for i in range(self.n_qubits):
            circuit.append({'gate': 'H', 'targets': [i]})
            for j in range(i + 1, self.n_qubits):
                k = j - i + 1
                angle = 2 * np.pi / (1 << k)
                circuit.append({
                    'gate': 'CPHASE',
                    'params': {'angle': angle},
                    'control': j,
                    'target': i
                })
        return circuit

    def get_matrix(self) -> np.ndarray:
        """
        Compute the full QFT unitary matrix of size N x N.
        F_{jk} = 1/sqrt(N) * exp(2*pi*i*j*k/N)
        """
        N = self.N
        F = np.zeros((N, N), dtype=complex)
        for j in range(N):
            for k in range(N):
                F[j, k] = np.exp(2j * np.pi * j * k / N) / np.sqrt(N)
        return F

    def inverse(self) -> list:
        """Return the inverse QFT circuit (IQFT)."""
        circuit = []
        for i in range(self.n_qubits - 1, -1, -1):
            for j in range(self.n_qubits - 1, i, -1):
                k = j - i + 1
                angle = -2 * np.pi / (1 << k)
                circuit.append({
                    'gate': 'CPHASE',
                    'params': {'angle': angle},
                    'control': j,
                    'target': i
                })
            circuit.append({'gate': 'H', 'targets': [i]})
        return circuit

    def apply(self, state_vector: np.ndarray) -> np.ndarray:
        """
        Apply QFT to a state vector using circuit-based operations.

        Uses H gates and controlled phase rotations for O(n^2) gate
        operations instead of O(N^2) matrix multiplication. Each gate
        is applied via direct vector updates that are O(N) per gate,
        giving total complexity O(n^2 * 2^n) = O(N log^2 N).

        Args:
            state_vector: Complex numpy array of shape (N,)

        Returns:
            Transformed state vector
        """
        import math
        import cmath

        state = state_vector.copy()
        n = self.n_qubits
        N = self.N

        for i in range(n):
            mask_i = 1 << i

            # H gate on qubit i
            inv_sqrt2 = 1.0 / math.sqrt(2)
            for k in range(N):
                if (k & mask_i) == 0:
                    j = k | mask_i
                    a_k = state[k]
                    a_j = state[j]
                    state[k] = (a_k + a_j) * inv_sqrt2
                    state[j] = (a_k - a_j) * inv_sqrt2

            for j in range(i + 1, n):
                mask_j = 1 << j
                angle = 2 * math.pi / (1 << (j - i + 1))
                phase = cmath.exp(1j * angle)
                both_mask = mask_i | mask_j

                for k in range(N):
                    if (k & both_mask) == both_mask:
                        state[k] *= phase

        return state
