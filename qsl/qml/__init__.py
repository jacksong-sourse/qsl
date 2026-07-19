"""QSL Quantum Machine Learning (QML) package."""

import warnings

_IMPORT_ERROR_MESSAGE = (
    "QML module requires torch, numpy, scipy, scikit-learn. "
    "Install: pip install qsl-quantum[qml] or pip install qsl-quantum[full]"
)

try:
    from .layers import QuantumLayer
    from .qnn import QNN
    from .kernels import quantum_kernel
    from .qsvm import QuantumSVM, QSVM
    from .qgan import QGAN
except ImportError as e:
    warnings.warn(
        f"QML modules not available: {e}. Install with: pip install qsl-quantum[qml]",
        ImportWarning
    )
    QuantumLayer = None
    QNN = None
    quantum_kernel = None
    QuantumSVM = None
    QSVM = None
    QGAN = None

__all__ = ["QuantumLayer", "QNN", "quantum_kernel", "QuantumSVM", "QSVM", "QGAN"]
