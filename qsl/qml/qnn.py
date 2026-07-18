"""Quantum Neural Network (QNN) - hybrid quantum-classical model."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset
from .layers import QuantumLayer


class QNN(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network.
    
    Architecture: Input -> QuantumLayer -> Linear -> Output
    
    Args:
        n_qubits: Number of qubits in the quantum layer
        n_features: Number of input features
        n_outputs: Number of output classes/values
        quantum_encoding: Encoding type for QuantumLayer
        n_quantum_layers: Number of variational layers in QuantumLayer
        hidden_dim: Dimension of hidden classical layer (None = no hidden layer)
    """
    
    def __init__(self,
                 n_qubits: int,
                 n_features: int,
                 n_outputs: int = 2,
                 quantum_encoding: str = "angle",
                 n_quantum_layers: int = 2,
                 hidden_dim: int = 16):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.n_outputs = n_outputs
        
        self.quantum_layer = QuantumLayer(
            n_qubits=n_qubits,
            n_features=n_features,
            encoding=quantum_encoding,
            n_layers=n_quantum_layers
        )
        
        # Classical post-processing
        if hidden_dim and hidden_dim > 0:
            self.classical = nn.Sequential(
                nn.Linear(n_qubits, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_outputs)
            )
        else:
            self.classical = nn.Linear(n_qubits, n_outputs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid network.
        
        Args:
            x: Input tensor of shape (batch, n_features)
            
        Returns:
            Output tensor of shape (batch, n_outputs)
        """
        quantum_out = self.quantum_layer(x)
        return self.classical(quantum_out)
    
    def fit(self, 
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 50,
            lr: float = 0.01,
            batch_size: int = 32,
            verbose: bool = True) -> list[float]:
        """
        Train the QNN model.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,) or (n_samples, n_outputs)
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size
            verbose: Print progress
            
        Returns:
            List of loss values per epoch
        """
        # Prepare data
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if y.ndim == 1:
            y_tensor = torch.tensor(y, dtype=torch.long)
            criterion = nn.CrossEntropyLoss()
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32)
            criterion = nn.MSELoss()
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                output = self(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predictions
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            output = self(X_tensor)
            
            if self.n_outputs > 1:
                # Classification: return class labels
                return output.argmax(dim=1).numpy()
            else:
                # Regression: return values
                return output.squeeze(-1).numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Probability array
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            output = self(X_tensor)
            return torch.softmax(output, dim=1).numpy()
