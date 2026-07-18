"""QSL Meta - Self-evolving quantum AI system."""

try:
    from .algorithm_search import AlgorithmSearcher
except ImportError:
    AlgorithmSearcher = None

try:
    from .quantum_compiler_ai import QuantumCompilerAI
except ImportError:
    QuantumCompilerAI = None

try:
    from .theory_generator import QuantumTheoremProver
except ImportError:
    QuantumTheoremProver = None

__all__ = ["AlgorithmSearcher", "QuantumCompilerAI", "QuantumTheoremProver"]
