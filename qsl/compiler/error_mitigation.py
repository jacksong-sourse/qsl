"""
Error mitigation techniques for noisy quantum computers.

Implements:
- Zero-Noise Extrapolation (ZNE)
- Readout error matrix correction
"""

import numpy as np
from typing import List, Dict, Callable, Optional

try:
    from scipy.optimize import curve_fit
    _HAS_SCIPY = True
except ImportError:
    curve_fit = None  # type: ignore
    _HAS_SCIPY = False


def zne(expectation_fn: Callable[[float], float],
        noise_scales: List[float] = None,
        extrapolation: str = "linear",
        base_noise: float = 1.0) -> float:
    """
    Zero-Noise Extrapolation (ZNE).
    
    Estimates the zero-noise expectation value by running the circuit
    at multiple amplified noise levels and extrapolating to zero noise.
    
    Args:
        expectation_fn: Function(noise_scale) -> expectation_value
        noise_scales: List of noise scale factors (default: [1, 2, 3])
        extrapolation: "linear" or "exponential"
        base_noise: Base noise level (default 1.0 = actual device noise)
        
    Returns:
        Extrapolated zero-noise expectation value
    """
    if noise_scales is None:
        noise_scales = [1.0, 2.0, 3.0]
    
    if len(noise_scales) < 2:
        raise ValueError("Need at least 2 noise scale factors for extrapolation")
    
    # Measure at each noise level
    values = []
    for scale in noise_scales:
        val = expectation_fn(scale)
        values.append(val)
    
    values = np.array(values)
    noise_scales = np.array(noise_scales)
    
    if extrapolation == "linear":
        # Linear fit: E(λ) = E(0) + a*λ
        coeffs = np.polyfit(noise_scales, values, 1)
        zero_noise_value = coeffs[1]  # Intercept
        return float(zero_noise_value)
    
    elif extrapolation == "exponential":
        # Exponential fit: E(λ) = E(0) * exp(-a*λ)
        # Linear in log space: log(E(λ)) = log(E(0)) - a*λ
        valid_mask = values > 0
        if valid_mask.sum() < 2:
            # Fallback to linear if values are not strictly positive
            coeffs = np.polyfit(noise_scales, values, 1)
            return float(coeffs[1])
        
        log_values = np.log(values[valid_mask])
        valid_scales = noise_scales[valid_mask]
        coeffs = np.polyfit(valid_scales, log_values, 1)
        zero_noise_value = np.exp(coeffs[1])
        return float(zero_noise_value)
    
    else:
        raise ValueError(f"Unknown extrapolation method: {extrapolation}")


def readout_error_correction(measurement_counts: Dict[str, int],
                              confusion_matrix: np.ndarray,
                              bit_labels: List[str] = None) -> Dict[str, int]:
    """
    Correct readout errors using the confusion matrix.
    
    Given the measurement confusion matrix P(out|true) and observed counts,
    estimate the true distribution via: n_true = P^{-1} * n_observed
    
    Args:
        measurement_counts: Dict mapping bitstring -> count
        confusion_matrix: P(out|true) matrix of shape (2, 2) for single qubit
                          or (2^n, 2^n) for multi-qubit
        bit_labels: Ordered list of bitstrings for the columns (auto-generated if None)
        
    Returns:
        Corrected measurement counts
    """
    if not measurement_counts:
        return {}
    
    if bit_labels is None:
        bit_labels = sorted(measurement_counts.keys())
    
    n_states = len(bit_labels)
    
    if confusion_matrix.shape == (2, 2):
        # Single-qubit case: extend to multi-qubit via tensor product
        n_qubits = len(next(iter(measurement_counts.keys())))
        full_matrix = confusion_matrix.copy()
        for _ in range(n_qubits - 1):
            full_matrix = np.kron(full_matrix, confusion_matrix)
        confusion_matrix = full_matrix
    
    # Build observed vector
    observed = np.zeros(n_states)
    for i, label in enumerate(bit_labels):
        observed[i] = measurement_counts.get(label, 0)
    
    # Pseudo-inverse for correction (handle singular matrices)
    try:
        if confusion_matrix.shape[0] < n_states:
            raise ValueError(
                f"confusion_matrix has {confusion_matrix.shape[0]} rows, "
                f"but {n_states} bit_labels provided"
            )
        corrected = np.linalg.solve(confusion_matrix[:n_states, :n_states], observed)
    except np.linalg.LinAlgError:
        # Fallback to least squares
        corrected, _, _, _ = np.linalg.lstsq(
            confusion_matrix[:n_states, :n_states], observed, rcond=None
        )
    
    # Ensure non-negative, renormalize
    corrected = np.maximum(corrected, 0)
    total = corrected.sum()
    if total > 0:
        corrected = corrected * (observed.sum() / total)
        corrected = np.round(corrected).astype(int)
    
    return {label: int(count) for label, count in zip(bit_labels, corrected) if count > 0}


def build_confusion_matrix(readout_error_p: float = 0.05) -> np.ndarray:
    """
    Build a single-qubit readout confusion matrix.
    
    P(out|true):
        P(0|0) = 1 - p,  P(0|1) = p
        P(1|0) = p,      P(1|1) = 1 - p
    
    Args:
        readout_error_p: Readout error probability per qubit
        
    Returns:
        2x2 confusion matrix
    """
    p = readout_error_p
    return np.array([[1 - p, p], [p, 1 - p]])


def richardson_extrapolate(values: List[float], 
                           noise_scales: List[float]) -> float:
    """
    Richardson extrapolation for ZNE.
    
    Args:
        values: Measured expectation values at each noise scale
        noise_scales: Corresponding noise scale factors
        
    Returns:
        Extrapolated zero-noise value
    """
    if len(values) < 2:
        return values[0] if values else 0.0
    
    values = np.array(values)
    noise_scales = np.array(noise_scales)
    
    # Polynomial fit of order len(values)-1
    coeffs = np.polyfit(noise_scales, values, len(values) - 1)
    return float(coeffs[-1])  # Constant term = zero-noise extrapolation
