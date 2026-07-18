"""
QSL - Quantum Search Language
=============================

Quantum computing framework supporting:
    - Declarative quantum search (Grover)
    - Quantum algorithms (QFT, Shor, QAOA, VQE)
    - Quantum machine learning (QNN, QSVM, QGAN)
    - Hardware backends (IBM, AWS Braket, Simulator)
    - Circuit compiler optimization
    - AI-powered quantum scientist
    - Application pipelines (drug discovery, crypto analysis, portfolio)

Quick start:
    >>> from qsl import QSLProgram, QSLCompiler
    >>> program = QSLProgram(name="test", n_qubits=3, premises=["x0 & x1"])
    >>> compiler = QSLCompiler()
    >>> result = compiler.compile_and_run(program)
"""

__version__ = "0.5.0"
__author__ = "Song Ziming"
__email__ = "15011462616@163.com"

# ----------------------------------------------------------------
# Core API (always available, zero deps beyond numpy)
# ----------------------------------------------------------------
from .compiler.program import QSLProgram
from .compiler.compiler import QSLCompiler, compile_and_run, analyze
from .compiler.dsl import parse_qsl
from .core.parser import parse_bool
from .core.state import QuantumState, DensityMatrix
from .core.grover import GroverSearch, GroverResult, solve_sat
from .backends.registry import list_backends, get_backend
from .backends.simulator import SimulatorBackend
from .backends.auto_selector import AutoBackend

# ----------------------------------------------------------------
# Quantum Gates (requires numpy)
# ----------------------------------------------------------------
from .quantum_gates import (
    I, X, Y, Z, H, S, T,
    rx, ry, rz, u3,
    swap, cswap, mcx,
    kronecker_prod, kron, controlled_gate,
)

# ----------------------------------------------------------------
# 真机后端 SDK 可用性检查 (导入时给出清晰警告)
# ----------------------------------------------------------------
import warnings as _warnings
from importlib.util import find_spec as _find_spec

_MISSING_BACKEND_SDK = []
if _find_spec("qiskit") is None or _find_spec("qiskit_ibm_runtime") is None:
    _MISSING_BACKEND_SDK.append(
        "IBM Quantum (pip install qiskit qiskit-aer qiskit-ibm-runtime)")
if _find_spec("braket") is None:
    _MISSING_BACKEND_SDK.append(
        "AWS Braket (pip install boto3 amazon-braket-sdk)")
if _MISSING_BACKEND_SDK:
    _warnings.warn(
        "QSL: 以下真机后端 SDK 未安装, 对应后端当前不可用: "
        + "; ".join(_MISSING_BACKEND_SDK)
        + "。本地模拟器后端不受影响。",
        RuntimeWarning, stacklevel=2,
    )
del _warnings, _find_spec, _MISSING_BACKEND_SDK

# ----------------------------------------------------------------
# Quantum Algorithms (requires numpy, scipy optional)
# ----------------------------------------------------------------
from .algorithms import (
    QuantumFourierTransform,
    ShorSolver,
    QAOA,
    VQE,
)

# ----------------------------------------------------------------
# Compiler Optimizations
# ----------------------------------------------------------------
from .compiler.optimizer import gate_fusion, commutation_optimization, depth_reduction
from .compiler.transpiler import layout_mapping, swap_insertion, get_coupling_graph
from .compiler.error_mitigation import (
    zne,
    readout_error_correction,
    build_confusion_matrix,
    richardson_extrapolate,
)

# ----------------------------------------------------------------
# Lazy imports for heavy/optional dependencies
# ----------------------------------------------------------------

_LAZY_IMPORTS = {
    "QuantumLayer": ("qsl.qml.layers", "QuantumLayer"),
    "QNN": ("qsl.qml.qnn", "QNN"),
    "quantum_kernel": ("qsl.qml.kernels", "quantum_kernel"),
    "QuantumSVM": ("qsl.qml.qsvm", "QuantumSVM"),
    "QGAN": ("qsl.qml.qgan", "QGAN"),
    "IBMBackend": ("qsl.backends.ibm", "IBMBackend"),
    "AWSBraketBackend": ("qsl.backends.aws_braket", "AWSBraketBackend"),
    "ProblemTranslator": ("qsl.ai.translator", "ProblemTranslator"),
    "QuantumAgent": ("qsl.ai.agent", "QuantumAgent"),
    "HypothesisTester": ("qsl.ai.hypotheses", "HypothesisTester"),
    "DiscoveryPipeline": ("qsl.ai.discovery", "DiscoveryPipeline"),
    "ResultExplainer": ("qsl.ai.explainer", "ResultExplainer"),
    "DrugDiscoveryPipeline": ("qsl.pipelines.drug_discovery", "DrugDiscoveryPipeline"),
    "CryptoAnalysisPipeline": ("qsl.pipelines.crypto_analysis", "CryptoAnalysisPipeline"),
    "PortfolioOptimizer": ("qsl.pipelines.portfolio", "PortfolioOptimizer"),
    "AlgorithmSearcher": ("qsl.meta.algorithm_search", "AlgorithmSearcher"),
    "QuantumCompilerAI": ("qsl.meta.quantum_compiler_ai", "QuantumCompilerAI"),
    "QuantumTheoremProver": ("qsl.meta.theory_generator", "QuantumTheoremProver"),
    "DistributedNode": ("qsl.network.distributed_node", "DistributedNode"),
    "QuantumCluster": ("qsl.network.distributed_node", "QuantumCluster"),
    "QuantumBlockchain": ("qsl.network.quantum_blockchain", "QuantumBlockchain"),
    "QuantumBlock": ("qsl.network.quantum_blockchain", "QuantumBlock"),
}


def __getattr__(name):
    """Lazy import for heavy/optional dependencies."""
    import importlib

    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        try:
            module = importlib.import_module(module_path)
            attr = getattr(module, attr_name)
            # Cache in module globals
            globals()[name] = attr
            return attr
        except ImportError as e:
            raise ImportError(
                f"Cannot import '{name}' because a required dependency is missing. "
                f"Run: pip install qsl-quantum[full] to install all optional dependencies. "
                f"Original error: {e}"
            ) from e

    raise AttributeError(f"module 'qsl' has no attribute '{name}'")


# ----------------------------------------------------------------
# Export list
# ----------------------------------------------------------------
__all__ = [
    # Core
    "QSLProgram", "QSLCompiler", "compile_and_run", "analyze",
    "parse_qsl", "parse_bool",
    "QuantumState", "DensityMatrix", "GroverSearch", "GroverResult", "solve_sat",
    "SimulatorBackend", "list_backends", "get_backend", "AutoBackend",
    # Gates
    "I", "X", "Y", "Z", "H", "S", "T",
    "rx", "ry", "rz", "u3",
    "swap", "cswap", "mcx",
    "kronecker_prod", "kron", "controlled_gate",
    # Algorithms
    "QuantumFourierTransform", "ShorSolver", "QAOA", "VQE",
    # Compiler
    "gate_fusion", "commutation_optimization", "depth_reduction",
    "layout_mapping", "swap_insertion", "get_coupling_graph",
    "zne", "readout_error_correction", "build_confusion_matrix",
    "richardson_extrapolate",
    # Lazy
    "QuantumLayer", "QNN", "quantum_kernel", "QuantumSVM", "QGAN",
    "IBMBackend", "AWSBraketBackend",
    "ProblemTranslator", "QuantumAgent", "HypothesisTester",
    "DiscoveryPipeline", "ResultExplainer",
    "DrugDiscoveryPipeline", "CryptoAnalysisPipeline", "PortfolioOptimizer",
    "AlgorithmSearcher", "QuantumCompilerAI", "QuantumTheoremProver",
    "DistributedNode", "QuantumCluster", "QuantumBlockchain", "QuantumBlock",
    # Version
    "__version__",
]
