"""
QSL Algorithms Package
======================

Quantum algorithm implementations including:
- QuantumFourierTransform: Quantum Fourier Transform
- ShorSolver: Shor's factoring algorithm (requires .shor module)
- QAOA: Quantum Approximate Optimization Algorithm (optional: scipy)
- VQE: Variational Quantum Eigensolver (optional: scipy)
- solve_sat: SAT solver based on Grover's search
"""

try:
    from .qft import QuantumFourierTransform
except (ImportError, SyntaxError):
    QuantumFourierTransform = None

# Shor's algorithm
try:
    from .shor import ShorSolver
except ImportError:
    ShorSolver = None

# QAOA (requires scipy)
try:
    from .qaoa import QAOA
except ImportError:
    QAOA = None

# VQE
try:
    from .vqe import VQE
except ImportError:
    VQE = None

# Grover-based SAT solver
try:
    from ..core.grover import solve_sat
except ImportError:
    solve_sat = None


__all__ = [
    "QuantumFourierTransform",
    "ShorSolver",
    "QAOA",
    "VQE",
    "solve_sat",
]
