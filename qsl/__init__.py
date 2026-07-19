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

__version__ = "0.6.2"
__author__ = "Song Ziming"
__email__ = "15011462616@163.com"

# ----------------------------------------------------------------
# Core API (always available, zero deps beyond numpy)
# ----------------------------------------------------------------
from .compiler.program import QSLProgram
from .compiler.compiler import QSLCompiler, compile_and_run, analyze
from .compiler.dsl import parse_qsl
from .core.parser import parse_bool
from .core.state import QuantumState, DensityMatrix, NoiseModel
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
# Circuit API (Qiskit 对标的电路层, 仅依赖 numpy)
# ----------------------------------------------------------------
from .circuit import (
    QuantumCircuit, Gate, Instruction, Parameter, ParameterExpression,
    ExecutionResult, library,
)
from .circuit.qasm import dumps_qasm2, loads_qasm2, dumps_qasm3, QASMParseError
from .circuit.converters import to_qiskit, from_qiskit, to_cirq

# ----------------------------------------------------------------
# 真机后端 SDK 可用性检查
#
# 设计说明 (修复"import 时全局 RuntimeWarning 不可抑制"的问题):
#   - 不再在模块顶层无条件 warn, 而是仅当用户首次访问真机后端
#     (IBMBackend / AWSBraketBackend) 时才通过 __getattr__ 提示。
#   - 可用环境变量 QSL_SILENT_BACKEND_CHECK=1 完全静默该检查。
#   - 这样 `import qsl` 保持干净, 用户可在 import 前用
#     warnings.filterwarnings('ignore') 正常控制输出。
# ----------------------------------------------------------------
import os as _os
from importlib.util import find_spec as _find_spec

_BACKEND_WARNING_EMITTED = False


def _missing_backend_sdks():
    """返回未安装的真机后端 SDK 描述列表。"""
    missing = []
    if _find_spec("qiskit") is None or _find_spec("qiskit_ibm_runtime") is None:
        missing.append(
            "IBM Quantum (pip install qiskit qiskit-aer qiskit-ibm-runtime)")
    if _find_spec("braket") is None:
        missing.append(
            "AWS Braket (pip install boto3 amazon-braket-sdk)")
    return missing


def _warn_backend_once():
    """仅在访问真机后端时提示一次缺失的 SDK。"""
    global _BACKEND_WARNING_EMITTED
    if _BACKEND_WARNING_EMITTED:
        return
    if _os.environ.get("QSL_SILENT_BACKEND_CHECK"):
        return
    missing = _missing_backend_sdks()
    if missing:
        import warnings as _w
        _w.warn(
            "QSL: 以下真机后端 SDK 未安装, 对应后端当前不可用: "
            + "; ".join(missing)
            + "。本地模拟器后端不受影响。",
            RuntimeWarning, stacklevel=3,
        )
    _BACKEND_WARNING_EMITTED = True

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
    "QSVM": ("qsl.qml.qsvm", "QSVM"),
    "QGAN": ("qsl.qml.qgan", "QGAN"),
    "IBMBackend": ("qsl.backends.ibm", "IBMBackend"),
    "AWSBraketBackend": ("qsl.backends.aws_braket", "AWSBraketBackend"),
    "ProblemTranslator": ("qsl.ai.translator", "ProblemTranslator"),
    "QuantumAgent": ("qsl.ai.agent", "QuantumAgent"),
    "HypothesisTester": ("qsl.ai.hypotheses", "HypothesisTester"),
    "DiscoveryPipeline": ("qsl.ai.discovery", "DiscoveryPipeline"),
    "ResultExplainer": ("qsl.ai.explainer", "ResultExplainer"),
    # AI: LLM 抽象层 / 自动验证 / 结构化报告 / 中文演示
    "LLMProvider": ("qsl.ai.llm_provider", "LLMProvider"),
    "create_provider": ("qsl.ai.llm_provider", "create_provider"),
    "set_default_provider": ("qsl.ai.llm_provider", "set_default_provider"),
    "get_default_provider": ("qsl.ai.llm_provider", "get_default_provider"),
    "VerificationResult": ("qsl.ai.verifier", "VerificationResult"),
    "verify": ("qsl.ai.verifier", "verify"),
    "AgentReport": ("qsl.ai.report", "AgentReport"),
    "run_demo": ("qsl.ai.demos", "run_demo"),
    "list_demos": ("qsl.ai.demos", "list_demos"),
    # 可视化 (需 matplotlib)
    "draw_circuit_mpl": ("qsl.viz.circuit_drawer", "draw_circuit_mpl"),
    "plot_bloch_sphere": ("qsl.viz.state_viz", "plot_bloch_sphere"),
    "plot_state_city": ("qsl.viz.state_viz", "plot_state_city"),
    "plot_amplitudes": ("qsl.viz.state_viz", "plot_amplitudes"),
    "plot_qsphere": ("qsl.viz.state_viz", "plot_qsphere"),
    "plot_histogram": ("qsl.viz.state_viz", "plot_histogram"),
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

    # 首次访问真机后端时才提示缺失的 SDK (可静默, 见模块顶部说明)
    if name in ("IBMBackend", "AWSBraketBackend"):
        _warn_backend_once()

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
    "QuantumState", "DensityMatrix", "NoiseModel",
    "GroverSearch", "GroverResult", "solve_sat",
    "SimulatorBackend", "list_backends", "get_backend", "AutoBackend",
    # Gates
    "I", "X", "Y", "Z", "H", "S", "T",
    "rx", "ry", "rz", "u3",
    "swap", "cswap", "mcx",
    "kronecker_prod", "kron", "controlled_gate",
    # Circuit API
    "QuantumCircuit", "Gate", "Instruction", "Parameter",
    "ParameterExpression", "ExecutionResult", "library",
    "dumps_qasm2", "loads_qasm2", "dumps_qasm3", "QASMParseError",
    "to_qiskit", "from_qiskit", "to_cirq",
    # Algorithms
    "QuantumFourierTransform", "ShorSolver", "QAOA", "VQE",
    # Compiler
    "gate_fusion", "commutation_optimization", "depth_reduction",
    "layout_mapping", "swap_insertion", "get_coupling_graph",
    "zne", "readout_error_correction", "build_confusion_matrix",
    "richardson_extrapolate",
    # Lazy
    "QuantumLayer", "QNN", "quantum_kernel", "QuantumSVM", "QSVM", "QGAN",
    "IBMBackend", "AWSBraketBackend",
    "ProblemTranslator", "QuantumAgent", "HypothesisTester",
    "DiscoveryPipeline", "ResultExplainer",
    "LLMProvider", "create_provider", "set_default_provider",
    "get_default_provider", "VerificationResult", "verify",
    "AgentReport", "run_demo", "list_demos",
    "draw_circuit_mpl", "plot_bloch_sphere", "plot_state_city",
    "plot_amplitudes", "plot_qsphere", "plot_histogram",
    "DrugDiscoveryPipeline", "CryptoAnalysisPipeline", "PortfolioOptimizer",
    "AlgorithmSearcher", "QuantumCompilerAI", "QuantumTheoremProver",
    "DistributedNode", "QuantumCluster", "QuantumBlockchain", "QuantumBlock",
    # Version
    "__version__",
]
