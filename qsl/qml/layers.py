import numpy as np
import torch
import torch.nn as nn
from typing import Literal, Optional


class QuantumLayer(nn.Module):
    """
    Quantum layer as a PyTorch nn.Module.

    Supports three encoding types:
        - "angle": Encode each feature as RY(pi * sigmoid(x_i)) rotation
        - "amplitude": Encode feature vector as amplitudes (requires normalization)
        - "dense_angle": Encode features through dense + angle rotation

    The layer outputs expectation values of PauliZ on each qubit.

    All gate operations are implemented as batched tensor operations
    (reshape / stack / gather / broadcast) over the full state vector —
    there are NO Python loops over the 2^n basis states, and every
    operation is differentiable (no in-place modification of tensors
    that participate in the autograd graph).

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

        # ---- 预计算的索引/符号缓冲 (全部向量化使用) ----
        indices = torch.arange(self.N)
        # PauliZ 符号矩阵: signs[q, k] = +1 (bit q 为 0) / -1 (bit q 为 1)
        signs = torch.ones(n_qubits, self.N)
        for q in range(n_qubits):
            signs[q, (indices & (1 << q)) != 0] = -1.0
        self.register_buffer("_z_signs", signs)

        # CNOT 链 (0->1, 1->2, ..., n-2->n-1) 的基态置换 (单次 gather)。
        # 第 k 个 CNOT 将基态 j 映射到 σ_k(j); 链的总效果为
        # final[j] = initial[σ_0(σ_1(...σ_last(j)))], 需按序复合。
        perm = torch.arange(self.N)
        for q in range(n_qubits - 1):
            c_mask = 1 << q
            t_mask = 1 << (q + 1)
            sigma = torch.where((indices & c_mask) != 0,
                                indices ^ t_mask, indices)
            perm = perm[sigma]
        self.register_buffer("_cnot_perm", perm)

        # Trainable parameters: [n_layers, n_qubits, 2] for RY+RZ per qubit per layer
        if encoding == "dense_angle":
            # Dense layer weights for encoding
            self.dense = nn.Linear(n_features, n_qubits)
        else:
            self.dense = None

        # Variational parameters
        n_params = n_layers * n_qubits * 2
        self.trainable_params = nn.Parameter(torch.randn(n_params) * 0.1)

    # ----------------------------------------------------------------
    # 向量化单量子比特门 (作用于 reshape 后的 (batch, 2, ..., 2) 态)
    # ----------------------------------------------------------------

    @staticmethod
    def _apply_ry_batched(state: torch.Tensor, q: int, n_qubits: int,
                          cos_half: torch.Tensor,
                          sin_half: torch.Tensor) -> torch.Tensor:
        """
        对第 q 个量子比特施加 RY 旋转 (批量, 可微)。

        state: (..., 2, 2, ..., 2) 复数张量 (最后 n_qubits 个轴为量子比特轴,
        其中量子比特 q 对应轴 -(q+1), 即量子比特 0 是最后一个轴/LSB)
        cos_half/sin_half: 标量或 (...) 形状张量 (逐样本角度)
        """
        # 将第 q 个量子比特轴移到最后
        state = torch.movedim(state, -(q + 1), -1)
        a0 = state[..., 0]
        a1 = state[..., 1]
        # 广播角度: cos/sin 形状为 (...) 或标量
        new0 = cos_half * a0 - sin_half * a1
        new1 = sin_half * a0 + cos_half * a1
        state = torch.stack([new0, new1], dim=-1)
        return torch.movedim(state, -1, -(q + 1))

    @staticmethod
    def _apply_rz_batched(state: torch.Tensor, q: int, n_qubits: int,
                          phi: torch.Tensor) -> torch.Tensor:
        """对第 q 个量子比特施加 RZ 旋转 (批量, 可微)。"""
        phase0 = torch.exp(-0.5j * phi)
        phase1 = torch.exp(0.5j * phi)
        state = torch.movedim(state, -(q + 1), -1)
        state = torch.stack([state[..., 0] * phase0,
                             state[..., 1] * phase1], dim=-1)
        return torch.movedim(state, -1, -(q + 1))

    # ----------------------------------------------------------------
    # 编码
    # ----------------------------------------------------------------

    def _encode_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode classical data into a quantum state vector.
        Returns state vector as complex tensor of shape (..., N).
        """
        batch_shape = x.shape[:-1]
        batch_size = int(np.prod(batch_shape)) if batch_shape else 1
        x = x.reshape(batch_size, -1)

        if self.encoding == "amplitude":
            # Encode normalized feature vector as amplitudes
            norm = torch.norm(x, dim=1, keepdim=True)
            x_norm = x / (norm + 1e-10)

            # Pad to power of 2
            padded = torch.zeros(batch_size, self.N, dtype=x.dtype,
                                 device=x.device)
            width = min(self.N, x_norm.shape[1])
            padded = torch.cat([x_norm[:, :width],
                                padded[:, width:]], dim=1)
            padded = padded / torch.norm(padded, dim=1, keepdim=True)

            return padded.to(torch.complex64).reshape(*batch_shape, self.N)

        if self.encoding == "dense_angle":
            # Dense layer first, then angle encoding with atan
            angles = torch.atan(self.dense(x))
        elif self.encoding == "angle":
            # Map features to [0, pi] using pi * sigmoid(x)
            width = min(self.n_qubits, x.shape[1])
            angles = torch.zeros(batch_size, self.n_qubits,
                                 dtype=x.dtype, device=x.device)
            angles = torch.cat(
                [np.pi * torch.sigmoid(x[:, :width]),
                 angles[:, width:]], dim=1)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

        # 角度编码: 从 |0...0> 出发逐量子比特施加 RY (向量化)
        state = torch.zeros(batch_size, *([2] * self.n_qubits),
                            dtype=torch.complex64, device=x.device)
        state[(slice(None),) + (0,) * self.n_qubits] = 1.0

        n_angles = min(self.n_qubits, angles.shape[1])
        for q in range(n_angles):
            # 角度广播到 (batch, 1, ..., 1): 与移除一个量子比特轴后的
            # 态切片 (batch, 2, ..., 2) 对齐 (n_qubits - 1 个单例轴)
            theta = angles[:, q].reshape(batch_size,
                                         *([1] * (self.n_qubits - 1)))
            cos_half = torch.cos(theta / 2)
            sin_half = torch.sin(theta / 2)
            state = self._apply_ry_batched(
                state, q, self.n_qubits, cos_half, sin_half)

        return state.reshape(*batch_shape, self.N)

    # ----------------------------------------------------------------
    # 变分电路
    # ----------------------------------------------------------------

    def _apply_variational_circuit(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply variational layers to the quantum state.

        Fully batched and differentiable: RY/RZ via stack/broadcast,
        CNOT chain via a single precomputed gather permutation.
        """
        batch_shape = state.shape[:-1]
        batch_size = int(np.prod(batch_shape)) if batch_shape else 1
        params = self.trainable_params.view(self.n_layers, self.n_qubits, 2)

        state = state.reshape(batch_size, *([2] * self.n_qubits))

        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                theta = params[layer, q, 0]
                phi = params[layer, q, 1]
                state = self._apply_ry_batched(
                    state, q, self.n_qubits,
                    torch.cos(theta / 2), torch.sin(theta / 2))
                state = self._apply_rz_batched(
                    state, q, self.n_qubits, phi)

            # Entangling: CNOT 链, 单次 gather 完成 (修正了原实现中
            # (k & c_mask) and (k & t_mask) == 0 的运算符优先级 bug)
            state = state.reshape(batch_size, self.N)[..., self._cnot_perm]
            state = state.reshape(batch_size, *([2] * self.n_qubits))

        return state.reshape(*batch_shape, self.N)

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

        # Compute PauliZ expectation for each qubit (单次矩阵乘):
        # ⟨Z_q⟩ = sum_k signs[q, k] * |α_k|^2
        probs = torch.abs(state) ** 2
        expectations = probs @ self._z_signs.T

        return expectations
