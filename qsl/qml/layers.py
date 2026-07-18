import numpy as np
import torch
import torch.nn as nn
from typing import Literal, Optional


class QuantumLayer(nn.Module):
    """
    Quantum layer as a PyTorch nn.Module.

    Supports three encoding types:
        - "angle": Encode each feature as RY(arctan(x_i)) rotation
        - "amplitude": Encode feature vector as amplitudes (requires normalization)
        - "dense_angle": Encode features through dense + angle rotation

    The layer outputs expectation values of PauliZ on each qubit.
    Gradient is computed via parameter-shift rule.

    Args:
        n_qubits: Number of qubits
        n_features: Number of input features
        encoding: Encoding type ("angle", "amplitude", "dense_angle")
        n_layers: Number of variational layers
    """

    def __init__(
        self,
        n_qubits: int,
        n_features: int,
        encoding: str = "angle",
        n_layers: int = 2,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.encoding = encoding
        self.n_layers = n_layers
        self.N = 1 << n_qubits

        # Trainable parameters: [n_layers, n_qubits, 2] for RY+RZ per qubit per layer
        if encoding == "dense_angle":
            # Dense layer weights for encoding
            self.dense = nn.Linear(n_features, n_qubits)
        else:
            self.dense = None

        # Variational parameters
        n_params = n_layers * n_qubits * 2
        self.trainable_params = nn.Parameter(torch.randn(n_params) * 0.1)

    def _encode_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode classical data into a quantum state vector.
        Returns state vector as complex tensor of shape (..., N).
        """
        batch_shape = x.shape[:-1]
        batch_size = int(torch.prod(torch.tensor(batch_shape))) if batch_shape else 1
        x = x.reshape(batch_size, -1)

        if self.encoding == "angle":
            # Initialize |0...0>
            state_real = torch.zeros(batch_size, self.N, dtype=torch.float32)
            state_imag = torch.zeros(batch_size, self.N, dtype=torch.float32)
            state_real[:, 0] = 1.0

            # Apply RY(pi * sigmoid(x_i)) on each qubit
            # Normalize features to [0, pi] to use full RY range
            for i in range(min(self.n_qubits, x.shape[1])):
                # Map to [0, pi] using pi * sigmoid(x)
                angles = np.pi * torch.sigmoid(x[:, i])
                cos_half = torch.cos(angles / 2)
                sin_half = torch.sin(angles / 2)

                # Apply RY rotation: simplified version
                # RY|0> = cos(θ/2)|0> + sin(θ/2)|1>
                mask = 1 << i
                for k in range(self.N):
                    if (k & mask) == 0:
                        j = k | mask
                        old_real_k = state_real[:, k].clone()
                        old_imag_k = state_imag[:, k].clone()
                        old_real_j = state_real[:, j].clone()
                        old_imag_j = state_imag[:, j].clone()

                        # RY rotation
                        state_real[:, k] = (
                            cos_half * old_real_k - sin_half * old_real_j
                        )
                        state_imag[:, k] = (
                            cos_half * old_imag_k - sin_half * old_imag_j
                        )
                        state_real[:, j] = (
                            sin_half * old_real_k + cos_half * old_real_j
                        )
                        state_imag[:, j] = (
                            sin_half * old_imag_k + cos_half * old_imag_j
                        )

            return torch.complex(state_real, state_imag).reshape(
                *batch_shape, self.N
            )

        elif self.encoding == "amplitude":
            # Encode normalized feature vector as amplitudes
            norm = torch.norm(x, dim=1, keepdim=True)
            x_norm = x / (norm + 1e-10)

            # Pad to power of 2
            padded = torch.zeros(batch_size, self.N)
            padded[:, : min(self.N, x_norm.shape[1])] = x_norm[
                :, : min(self.N, x_norm.shape[1])
            ]
            padded = padded / torch.norm(padded, dim=1, keepdim=True)

            return torch.complex(padded, torch.zeros_like(padded)).reshape(
                *batch_shape, self.N
            )

        elif self.encoding == "dense_angle":
            # Dense layer first, then angle encoding with atan
            x = self.dense(x)
            x = torch.atan(x)  # Apply atan for bounded mapping, consistent with angle path
            return self._encode_state_angle(x, batch_shape)

        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

    def _encode_state_angle(
        self, x: torch.Tensor, batch_shape: tuple
    ) -> torch.Tensor:
        """Helper for angle encoding with pre-computed dense output."""
        state_real = torch.zeros(x.shape[0], self.N, dtype=torch.float32)
        state_imag = torch.zeros(x.shape[0], self.N, dtype=torch.float32)
        state_real[:, 0] = 1.0

        for i in range(min(self.n_qubits, x.shape[1])):
            angles = x[:, i]
            cos_half = torch.cos(angles / 2)
            sin_half = torch.sin(angles / 2)

            mask = 1 << i
            for k in range(self.N):
                if (k & mask) == 0:
                    j = k | mask
                    old_rk = state_real[:, k].clone()
                    old_ik = state_imag[:, k].clone()
                    old_rj = state_real[:, j].clone()
                    old_ij = state_imag[:, j].clone()
                    state_real[:, k] = cos_half * old_rk - sin_half * old_rj
                    state_imag[:, k] = cos_half * old_ik - sin_half * old_ij
                    state_real[:, j] = sin_half * old_rk + cos_half * old_rj
                    state_imag[:, j] = sin_half * old_ik + cos_half * old_ij

        return torch.complex(state_real, state_imag).reshape(*batch_shape, self.N)

    def _apply_variational_circuit(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply variational layers to the quantum state.
        Uses parameter-shift rule compatible operations.
        Minimize clone/copy operations for performance.
        """
        params = self.trainable_params.view(self.n_layers, self.n_qubits, 2)

        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                theta = params[layer, q, 0]
                phi = params[layer, q, 1]

                # RY(theta) rotation
                cos_half = torch.cos(theta / 2)
                sin_half = torch.sin(theta / 2)

                mask = 1 << q
                for k in range(self.N):
                    if (k & mask) == 0:
                        j = k | mask
                        # Use index_select to avoid .clone() where possible
                        old_k_r = state[..., k].real
                        old_k_i = state[..., k].imag
                        old_j_r = state[..., j].real
                        old_j_i = state[..., j].imag
                        new_k_r = cos_half * old_k_r - sin_half * old_j_r
                        new_k_i = cos_half * old_k_i - sin_half * old_j_i
                        new_j_r = sin_half * old_k_r + cos_half * old_j_r
                        new_j_i = sin_half * old_k_i + cos_half * old_j_i
                        state[..., k] = torch.complex(new_k_r, new_k_i)
                        state[..., j] = torch.complex(new_j_r, new_j_i)

                # RZ(phi) rotation
                phase0 = torch.exp(-1j * phi / 2)
                phase1 = torch.exp(1j * phi / 2)
                for k in range(self.N):
                    if k & mask:
                        state[..., k] = state[..., k] * phase1
                    else:
                        state[..., k] = state[..., k] * phase0

            # Entangling: CNOT chain
            for q in range(self.n_qubits - 1):
                c_mask = 1 << q
                t_mask = 1 << (q + 1)
                for k in range(self.N):
                    if (k & c_mask) and (k & t_mask) == 0:
                        j = k ^ t_mask
                        sk = state[..., k].clone()
                        sj = state[..., j].clone()
                        state[..., k] = sj
                        state[..., j] = sk

        return state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode input, apply variational circuit,
        return PauliZ expectation values.

        Args:
            x: Input tensor of shape (..., n_features)

        Returns:
            Tensor of shape (..., n_qubits) with expectation values in [-1, 1]
        """
        # Encode
        state = self._encode_state(x)

        # Apply variational circuit
        state = self._apply_variational_circuit(state)

        # Compute PauliZ expectation for each qubit
        # ⟨Z_i⟩ = sum_j (-1)^(bit_i of j) * |α_j|^2
        probs = torch.abs(state) ** 2

        expectations = torch.zeros(
            *state.shape[:-1],
            self.n_qubits,
            dtype=torch.float32,
            device=state.device,
        )

        for q in range(self.n_qubits):
            mask = 1 << q
            # Sum probabilities with sign (+1 for |0>, -1 for |1>)
            z_plus = torch.zeros(
                *state.shape[:-1], dtype=torch.float32, device=state.device
            )
            z_minus = torch.zeros(
                *state.shape[:-1], dtype=torch.float32, device=state.device
            )

            for k in range(self.N):
                if k & mask:
                    z_minus = z_minus + probs[..., k]
                else:
                    z_plus = z_plus + probs[..., k]

            expectations[..., q] = z_plus - z_minus

        return expectations
