"""Quantum Generative Adversarial Network (QGAN) with real quantum generator."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from .layers import QuantumLayer


def _bernoulli_straight_through(probs: torch.Tensor) -> torch.Tensor:
    """
    可微的 Bernoulli 采样 (Straight-Through Estimator)。

    前向传播: 返回硬二值采样结果 (与 torch.bernoulli 相同分布);
    反向传播: 梯度恒等传递到 probs (梯度估计的无偏一阶近似),
    避免 torch.bernoulli 不可微导致的生成器梯度链断裂。
    """
    hard = torch.bernoulli(probs)
    return probs + (hard - probs).detach()


class QuantumGenerator(nn.Module):
    """
    True quantum generator using parameterized quantum circuit.

    Uses QuantumLayer to encode latent noise into a quantum state,
    apply a variational circuit, and measure expectation values.
    The outputs are probabilities derived from quantum measurements,
    allowing gradient-based optimization via parameter-shift rule.

    Args:
        latent_dim: Dimension of latent noise vector
        data_dim: Dimension of generated data
        n_qubits: Number of qubits in the quantum circuit
        n_layers: Number of variational layers in the quantum circuit
        device: Device to run on ('cpu' or 'cuda')
    """

    def __init__(self, latent_dim: int, data_dim: int, n_qubits: int = None,
                 n_layers: int = 2, device: str = 'cpu'):
        super().__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.n_qubits = n_qubits or max(latent_dim, data_dim)
        self.n_layers = n_layers
        self.device = device
        self.N = 1 << self.n_qubits

        self.quantum_layer = QuantumLayer(
            n_qubits=self.n_qubits,
            n_features=latent_dim,
            encoding="angle",
            n_layers=n_layers
        ).to(device)

        self.classical_head = nn.Sequential(
            nn.Linear(self.n_qubits, self.n_qubits),
            nn.ReLU(),
            nn.Linear(self.n_qubits, data_dim),
            nn.Sigmoid()
        ).to(device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate samples from latent noise using quantum circuit."""
        quantum_out = self.quantum_layer(z)
        probs = self.classical_head(quantum_out)
        samples = _bernoulli_straight_through(probs)
        return samples

    def sample(self, n_samples: int, device: str = 'cpu') -> np.ndarray:
        """Generate n_samples from the generator."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(device)
            return self(z).cpu().numpy()


class ClassicalAngularGenerator(nn.Module):
    """
    Classical neural network generator with angular output activation.

    Note: This is a classical neural network generator that emulates
    a parameterized quantum circuit via sin/cos transformations on
    learned angles. It is NOT a true quantum circuit.

    The generator maps latent noise through a classical neural network
    to produce circuit-like angle parameters, then applies a sinusoidal
    transformation (emulating RX rotation) to produce probabilities,
    from which binary samples are drawn via Bernoulli sampling.
    """

    def __init__(self, latent_dim: int, data_dim: int, n_qubits: int = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.n_qubits = n_qubits or max(latent_dim, data_dim)
        self.N = 1 << self.n_qubits

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_qubits * 3),
        )

        # 当 n_qubits < data_dim 时, 概率向量只有 n_qubits 列,
        # 不足以喂给判别器 (期望 data_dim 列)。增加一个线性投影
        # 把 n_qubits 维提升到 data_dim 维, 保证输出维度恒为 data_dim。
        if self.n_qubits < data_dim:
            self.proj = nn.Linear(self.n_qubits, data_dim)
        else:
            self.proj = None

    def _circuit_to_probabilities(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Convert "circuit" angles to measurement probabilities.

        Uses sin^2(theta/2) which corresponds to the probability of
        measuring |1> after an RX(theta) rotation on |0>.
        """
        batch_size = angles.shape[0]
        angles = angles.view(batch_size, self.n_qubits, 3)
        prob_one = torch.sin(angles[:, :, 0] / 2) ** 2
        return prob_one

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate samples from latent noise."""
        angles = self.fc(z)
        probs = self._circuit_to_probabilities(angles)
        if self.proj is not None:
            # n_qubits < data_dim: 先投影到 data_dim 维 (输出已是 [0,1] 概率空间)
            probs = torch.sigmoid(self.proj(probs))
        else:
            probs = probs[:, :self.data_dim]
        samples = _bernoulli_straight_through(probs)
        return samples

    def sample(self, n_samples: int, device: str = 'cpu') -> np.ndarray:
        """Generate n_samples from the generator."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(device)
            return self(z).cpu().numpy()


class ClassicalGenerator(nn.Module):
    """
    Purely classical generator for comparison.

    Uses standard FC layers with no quantum-inspired transformations.
    This serves as a baseline to measure any benefit from the
    quantum generator.
    """

    def __init__(self, latent_dim: int, data_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, data_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return _bernoulli_straight_through(self.fc(z))

    def sample(self, n_samples: int, device: str = 'cpu') -> np.ndarray:
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(device)
            return self(z).cpu().numpy()


class ClassicalDiscriminator(nn.Module):
    """Discriminator network for QGAN."""

    def __init__(self, data_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QGAN:
    """
    Quantum Generative Adversarial Network.

    Uses a true quantum generator (parameterized quantum circuit)
    and a classical discriminator trained adversarially.

    The quantum generator uses QuantumLayer to encode latent noise
    into quantum states and apply variational circuits, with gradients
    computed via parameter-shift rule.

    Args:
        latent_dim: Dimension of latent noise vector
        data_dim: Dimension of generated data
        n_qubits: Number of qubits in the quantum circuit
        device: Device to run on ('cpu' or 'cuda')
        use_classical: If True, use ClassicalGenerator instead of
                       the quantum generator
        use_angular: If True, use ClassicalAngularGenerator (quantum-inspired)
        use_quantum: If True, use QuantumGenerator with parameterized quantum circuit
        n_layers: Number of variational layers in the quantum circuit
    """

    def __init__(self,
                 latent_dim: int,
                 data_dim: int,
                 n_qubits: int = None,
                 device: str = 'cpu',
                 use_classical: bool = False,
                 use_angular: bool = False,
                 use_quantum: bool = False,
                 n_layers: int = 2):
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.n_qubits = n_qubits or max(latent_dim, data_dim)
        self.device = device
        self.n_layers = n_layers

        if use_classical:
            self.generator = ClassicalGenerator(
                latent_dim, data_dim
            ).to(device)
        elif use_angular:
            self.generator = ClassicalAngularGenerator(
                latent_dim, data_dim, self.n_qubits
            ).to(device)
        elif use_quantum:
            self.generator = QuantumGenerator(
                latent_dim, data_dim, self.n_qubits, n_layers, device
            ).to(device)
        else:
            self.generator = ClassicalAngularGenerator(
                latent_dim, data_dim, self.n_qubits
            ).to(device)

        self.discriminator = ClassicalDiscriminator(data_dim).to(device)

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.001, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

    def train(self,
              real_data: np.ndarray,
              epochs: int = 200,
              batch_size: int = 64,
              d_steps: int = 1,
              g_steps: int = 1,
              verbose: bool = True) -> dict[str, list]:
        """
        Train the QGAN on real data.

        Args:
            real_data: Training data of shape (n_samples, data_dim)
            epochs: Number of training epochs
            batch_size: Batch size
            d_steps: Discriminator updates per epoch
            g_steps: Generator updates per epoch
            verbose: Print progress

        Returns:
            Dict with 'g_loss' and 'd_loss' history
        """
        real_data = torch.tensor(real_data, dtype=torch.float32).to(self.device)
        n_samples = real_data.shape[0]

        g_losses = []
        d_losses = []

        for epoch in range(epochs):
            for _ in range(d_steps):
                self.d_optimizer.zero_grad()

                idx = np.random.choice(n_samples, batch_size, replace=True)
                real_batch = real_data[idx]
                real_labels = torch.ones(batch_size, 1, device=self.device) * 0.9

                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_batch = self.generator(z)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)

                real_loss = self.criterion(self.discriminator(real_batch), real_labels)
                fake_loss = self.criterion(self.discriminator(fake_batch.detach()), fake_labels)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.d_optimizer.step()

            for _ in range(g_steps):
                self.g_optimizer.zero_grad()

                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_batch = self.generator(z)
                real_labels = torch.ones(batch_size, 1, device=self.device)

                g_loss = self.criterion(self.discriminator(fake_batch), real_labels)
                g_loss.backward()
                self.g_optimizer.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}: D loss = {d_loss.item():.4f}, G loss = {g_loss.item():.4f}")

        return {'g_loss': g_losses, 'd_loss': d_losses}

    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples from the trained generator."""
        return self.generator.sample(n_samples, self.device)

    def generate(self, n_samples: int = 100) -> np.ndarray:
        """Alias for sample()."""
        return self.sample(n_samples)