"""
经典模拟器后端。

在经典计算机上使用状态向量方法模拟量子计算。
这是 QSL 的默认后端，无需任何外部量子硬件或库。

实现原理:
    - 使用 QuantumState 类进行完整的状态向量模拟
    - 直接构建 GroverSearch 实例并执行
    - 所有计算都在本地 CPU/内存完成

适用范围:
    - 教学和原型验证
    - 小规模搜索 (n <= 20)
    - 需要完全透明的量子态内省

失败模式分析:
    1. n_qubits > 20: 2^20 = 1M 个复数振幅 ≈ 16MB, 更大则内存不足
    2. n_qubits > 15: 某些低内存设备上可能已经开始变慢
    3. 大量 shots: 每次测量需要采样，但相比状态操作开销可忽略
    4. 数值精度: 多次门操作后振幅可能漂移，归一化检查可选
"""

from typing import Callable, Optional

from .base import AbstractBackend
from ..core.state import QuantumState, MAX_QUBITS
from ..core.grover import GroverSearch, GroverResult


class SimulatorBackend(AbstractBackend):
    """
    经典量子模拟器后端。

    使用纯 Python 实现的状态向量模拟，零外部依赖。
    支持最多 20 个量子比特的模拟。

    使用:
        >>> backend = SimulatorBackend()
        >>> result = backend.run_grover_search(
        ...     n_qubits=3,
        ...     oracle=lambda x: x == 5,
        ...     num_solutions=1,
        ...     shots=10
        ... )
    """

    def __init__(self, name: str = "simulator", **options):
        """
        初始化模拟器后端。

        参数:
            name: 后端标识
            **options:
                - normalize_after_gates: 每次门操作后归一化 (默认 False)
                - check_normalization: 最后检查归一化 (默认 True)
        """
        super().__init__(name=name, **options)
        self._normalize_after_gates = options.get(
            "normalize_after_gates", False
        )
        self._check_normalization = options.get(
            "check_normalization", True
        )

    @property
    def max_qubits(self) -> int:
        return MAX_QUBITS

    def run_grover_search(self,
                           n_qubits: int,
                           oracle: Callable[[int], bool],
                           num_solutions: int,
                           shots: int,
                           verbose: bool = False,
                           **run_options) -> GroverResult:
        """
        在本地模拟器上运行 Grover 搜索。

        参数:
            n_qubits: 量子比特数
            oracle: Boolean oracle 函数
            num_solutions: 已知解的数量
            shots: 测量次数
            verbose: 是否输出详细过程
            **run_options: 额外选项

        返回:
            GroverResult

        失败模式:
            - n_qubits > max_qubits: validate_request 中抛异常
            - 搜索结果零解: 由 GroverSearch 内部处理
        """
        # 验证请求
        self.validate_request(n_qubits, shots)

        # 创建 GroverSearch 实例并执行
        search = GroverSearch(n_qubits, verbose=verbose)
        result = search.search(
            condition=oracle,
            num_solutions=num_solutions,
            shots=shots,
        )

        return result

    def run_custom_circuit(self,
                            n_qubits: int,
                            circuit_fn: Callable[[QuantumState], None],
                            shots: int = 1) -> list:
        """
        运行自定义量子电路 (不限于 Grover 搜索)。

        参数:
            n_qubits: 量子比特数
            circuit_fn: 接收 QuantumState 并对其应用门操作的函数
            shots: 测量次数

        返回:
            [(测量结果, 概率), ...]

        失败模式:
            - n_qubits 越界: validate_request
            - circuit_fn 抛出异常: 直接传递
        """
        self.validate_request(n_qubits, shots)

        state = QuantumState(n_qubits)
        circuit_fn(state)

        if self._check_normalization:
            if not state.check_normalization():
                # 可选自动修复
                if self._normalize_after_gates:
                    state.normalize()

        return state.sample(shots)
