"""Quantum kernel methods for machine learning."""

import numpy as np
from typing import Callable


def _angle_encoding_state(x: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Encode a feature vector into a quantum state using angle encoding.

    Applies RY(arctan(x_i)) rotation on each qubit i, starting from |0...0>.
    Each rotation transforms the state in-place.

    Args:
        x: Feature vector
        n_qubits: Number of qubits (must be >= len(x))

    Returns:
        Normalized state vector of shape (2^n_qubits,)
    """
    N = 1 << n_qubits
    state = np.zeros(N, dtype=complex)
    state[0] = 1.0

    for i in range(min(n_qubits, len(x))):
        angle = np.arctan(x[i])
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)

        mask = 1 << i

        for k in range(N):
            if (k & mask) == 0:
                j = k | mask
                old_k = state[k].copy()
                old_j = state[j].copy()
                # RY: |0> -> cos|0> + sin|1>,  |1> -> -sin|0> + cos|1>
                state[k] = cos_half * old_k - sin_half * old_j
                state[j] = sin_half * old_k + cos_half * old_j

    return state


def quantum_kernel(X1: np.ndarray,
                   X2: np.ndarray,
                   feature_map: Callable = None,
                   n_qubits: int = None) -> np.ndarray:
    """
    Compute the quantum kernel (Gram) matrix.

    K(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|²

    This is the fidelity between feature-mapped quantum states.

    Args:
        X1: First set of samples of shape (n1, n_features)
        X2: Second set of samples of shape (n2, n_features)
        feature_map: Function mapping x -> quantum state (default: angle encoding)
        n_qubits: Number of qubits (default: n_features)

    Returns:
        Kernel matrix of shape (n1, n2)
    """
    if n_qubits is None:
        n_qubits = X1.shape[1]

    if feature_map is None:
        def feature_map(x):
            return _angle_encoding_state(x, n_qubits)

    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = np.zeros((n1, n2))

    phi1 = np.array([feature_map(X1[i]) for i in range(n1)])

    for j in range(n2):
        phi2 = feature_map(X2[j])
        for i in range(n1):
            inner = np.abs(np.dot(phi1[i].conj(), phi2)) ** 2
            K[i, j] = float(inner)

    return K


def rbf_quantum_kernel(X1: np.ndarray,
                       X2: np.ndarray,
                       gamma: float = 1.0,
                       n_qubits: int = None) -> np.ndarray:
    """
    RBF-inspired quantum kernel using angle encoding.

    K(x, y) = exp(-gamma * (1 - fidelity(x, y)))
    approximates exp(-gamma * ||x - y||^2)

    Args:
        X1, X2: Input data
        gamma: Kernel width parameter
        n_qubits: Number of qubits

    Returns:
        Kernel matrix
    """
    base_kernel = quantum_kernel(X1, X2, n_qubits=n_qubits)
    K = np.exp(-gamma * (1 - base_kernel))
    return K
