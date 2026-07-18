"""Quantum Fourier Transform implementation."""

import numpy as np


class QuantumFourierTransform:
    """
    Quantum Fourier Transform on n_qubits.

    QFT|j> = 1/sqrt(N) * sum_{k=0}^{N-1} exp(2*pi*i*j*k/N) |k>
    where N = 2^n_qubits

    Circuit (qubit 0 = LSB convention, consistent with get_matrix):
        for target i = n-1 down to 0:
            H(i)
            for control j = i-1 down to 0:
                CPHASE(2*pi/2^(i-j+1), control=j, target=i)
        finally: bit-reversal SWAP network

    Complexity: O(n^2) gates
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.N = 1 << n_qubits

    def build_circuit(self) -> list:
        """
        Build QFT circuit as a list of gate operations.

        Uses the MSB-first processing order plus a final bit-reversal
        SWAP network, so the circuit implements exactly the unitary
        returned by get_matrix().

        Returns list of gate operations forming the full QFT circuit.
        """
        circuit = []
        for i in range(self.n_qubits - 1, -1, -1):
            circuit.append({'gate': 'H', 'targets': [i]})
            for j in range(i - 1, -1, -1):
                k = i - j + 1
                angle = 2 * np.pi / (1 << k)
                circuit.append({
                    'gate': 'CPHASE',
                    'params': {'angle': angle},
                    'control': j,
                    'target': i
                })
        # 位反转 SWAP 网络
        for i in range(self.n_qubits // 2):
            circuit.append({'gate': 'SWAP', 'targets': [i, self.n_qubits - 1 - i]})
        return circuit

    def get_matrix(self) -> np.ndarray:
        """
        Compute the full QFT unitary matrix of size N x N.
        F_{jk} = 1/sqrt(N) * exp(2*pi*i*j*k/N)
        """
        N = self.N
        j = np.arange(N).reshape(-1, 1)
        k = np.arange(N).reshape(1, -1)
        return np.exp(2j * np.pi * j * k / N) / np.sqrt(N)

    def inverse(self) -> list:
        """Return the inverse QFT circuit (IQFT)."""
        circuit = []
        # 先撤销位反转 (SWAP 自逆)
        for i in range(self.n_qubits // 2):
            circuit.append({'gate': 'SWAP', 'targets': [i, self.n_qubits - 1 - i]})
        # 再以相反顺序施加负角度的受控相位和 H
        for i in range(self.n_qubits):
            for j in range(i):
                k = i - j + 1
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

        Uses the same MSB-first circuit as build_circuit() (H gates +
        controlled phase rotations + final bit reversal), so the result
        is identical to get_matrix() @ state_vector up to floating-point
        precision. Each gate is applied via direct vector updates that
        are O(N) per gate, giving total complexity O(n^2 * 2^n).

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
        inv_sqrt2 = 1.0 / math.sqrt(2)

        for i in range(n - 1, -1, -1):
            mask_i = 1 << i

            # H gate on qubit i
            for k in range(N):
                if (k & mask_i) == 0:
                    j = k | mask_i
                    a_k = state[k]
                    a_j = state[j]
                    state[k] = (a_k + a_j) * inv_sqrt2
                    state[j] = (a_k - a_j) * inv_sqrt2

            # Controlled phase rotations: control j < i, target i.
            # CPHASE 仅当控制位和目标位同时为 |1> 时附加相位
            # (受控相位门的定义), 角度为 2*pi/2^(i-j+1)。
            for j in range(i - 1, -1, -1):
                mask_j = 1 << j
                angle = 2 * math.pi / (1 << (i - j + 1))
                phase = cmath.exp(1j * angle)
                both_mask = mask_i | mask_j

                for k in range(N):
                    if (k & both_mask) == both_mask:
                        state[k] *= phase

        # 位反转 (等价于 build_circuit 末尾的 SWAP 网络)
        indices = np.arange(N)
        reversed_indices = np.zeros(N, dtype=np.int64)
        for b in range(n):
            reversed_indices |= ((indices >> b) & 1) << (n - 1 - b)
        return state[reversed_indices]
