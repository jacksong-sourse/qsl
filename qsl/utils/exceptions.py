"""
QSL 自定义异常体系。

每个异常类都有明确的语义，便于调用方精确捕获和处理不同类型的错误。
"""


class QSLError(Exception):
    """QSL 库所有异常的基类。"""
    pass


# --- 量子态相关异常 ---

class QuantumStateError(QSLError):
    """量子态操作错误基类。"""
    pass


class InvalidQubitCountError(QuantumStateError):
    """量子比特数不合法（< 1 或超过模拟器上限）。"""

    def __init__(self, n_qubits: int, max_qubits: int = 20):
        self.n_qubits = n_qubits
        self.max_qubits = max_qubits
        super().__init__(
            f"量子比特数必须 >= 1 且 <= {max_qubits}，当前值: {n_qubits}"
        )


class QubitIndexError(QuantumStateError):
    """量子比特索引越界。"""

    def __init__(self, index: int, n_qubits: int):
        self.index = index
        self.n_qubits = n_qubits
        super().__init__(
            f"量子比特索引 {index} 越界，有效范围: [0, {n_qubits - 1}]"
        )


class DuplicateQubitError(QuantumStateError):
    """门操作中重复指定同一个量子比特。"""

    def __init__(self, qubits: list):
        self.qubits = qubits
        super().__init__(
            f"门操作不允许有重复的量子比特索引: {qubits}"
        )


class StateNormalizationError(QuantumStateError):
    """量子态归一化检查失败（概率和不等于 1）。"""

    def __init__(self, total_prob: float):
        self.total_prob = total_prob
        super().__init__(
            f"量子态归一化失败：总概率 = {total_prob:.10f}，期望 ≈ 1.0"
        )


# --- 解析器相关异常 ---

class ParseError(QSLError):
    """表达式或 DSL 解析错误基类。"""

    def __init__(self, message: str, source: str = "", position: int = -1):
        self.source = source
        self.position = position
        if position >= 0 and position < len(source):
            prefix = source[:position]
            suffix = source[position:]
            msg = f"{message}\n  {prefix}>>>{suffix}"
        else:
            msg = message
        super().__init__(msg)


class BooleanParseError(ParseError):
    """布尔表达式语法错误。"""
    pass


class DSLParseError(ParseError):
    """QSL DSL 语法错误。"""
    pass


# --- 编译器相关异常 ---

class CompilerError(QSLError):
    """编译器错误基类。"""
    pass


class NoSolutionError(CompilerError):
    """搜索空间中没有解（前提条件矛盾）。"""

    def __init__(self, premises: list, n_qubits: int):
        self.premises = premises
        self.n_qubits = n_qubits
        N = 1 << n_qubits
        super().__init__(
            f"在 {N} 个状态中没有找到满足所有前提的解。"
            f"前提: {premises}"
        )


class ProgramValidationError(CompilerError):
    """QSL 程序定义不合法。"""

    def __init__(self, field: str, value, reason: str):
        self.field = field
        self.value = value
        self.reason = reason
        super().__init__(
            f"程序字段 '{field}' 无效: {reason} (当前值: {value})"
        )


# --- 后端相关异常 ---

class BackendError(QSLError):
    """后端错误基类。"""
    pass


class BackendConnectionError(BackendError):
    """无法连接到量子后端。"""

    def __init__(self, backend_name: str, details: str = ""):
        self.backend_name = backend_name
        self.details = details
        msg = f"无法连接到后端 '{backend_name}'"
        if details:
            msg += f": {details}"
        super().__init__(msg)


class BackendJobError(BackendError):
    """量子作业执行失败。"""

    def __init__(self, job_id: str, backend_name: str, details: str = ""):
        self.job_id = job_id
        self.backend_name = backend_name
        self.details = details
        msg = f"作业 '{job_id}' 在后端 '{backend_name}' 上执行失败"
        if details:
            msg += f": {details}"
        super().__init__(msg)


class BackendNotAvailableError(BackendError):
    """指定的量子后端不可用。"""

    def __init__(self, backend_name: str):
        self.backend_name = backend_name
        super().__init__(
            f"后端 '{backend_name}' 不可用。"
            f"请检查后端名称或使用 list_backends() 查看可用后端。"
        )


class DependencyNotInstalledError(BackendError):
    """所需依赖未安装。"""

    def __init__(self, package_name: str, install_hint: str):
        self.package_name = package_name
        self.install_hint = install_hint
        super().__init__(
            f"缺少依赖 '{package_name}'。"
            f"请运行: {install_hint}"
        )


# --- 配置相关异常 ---

class ConfigurationError(QSLError):
    """配置错误。"""
    pass
