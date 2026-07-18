"""QSL Quantum Machine Learning (QML) package."""

try:
    from .layers import QuantumLayer
    from .qnn import QNN
    from .kernels import quantum_kernel
    from .qsvm import QuantumSVM
    from .qgan import QGAN
except ImportError as e:
    raise ImportError(
        "QML module requires torch, numpy, scipy, scikit-learn. "
        "Install: pip install torch scipy scikit-learn"
    ) from e

__all__ = ["QuantumLayer", "QNN", "quantum_kernel", "QuantumSVM", "QGAN"]
