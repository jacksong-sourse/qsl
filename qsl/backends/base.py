"""
量子后端抽象基类。

所有后端必须实现的统一接口。该接口使得上层编译器无需关心
具体是在模拟器还是真实量子计算机上运行。

设计原则:
    - 所有后端实现相同的 run_grover_search() 方法
    - 每个后端负责自己的初始化、验证和错误处理
    - 结果统一返回 GroverResult (由 core.grover 定义)

失败模式分析 (接口层):
    1. n_qubits 超出后端能力: 每个后端有独立的量子比特上限
    2. Oracle 函数无法直接传输到真实硬件: 需要电路编译
    3. shots 超出后端限制: 某些后端有最大测量次数限制
    4. 连接超时: 真实硬件可能有队列等待
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

from ..core.grover import GroverResult


class AbstractBackend(ABC):
    """
    量子计算后端抽象基类。

    子类必须实现:
        - run_grover_search()
        - max_qubits 属性
        - backend_name 属性
    """

    def __init__(self, name: str = "abstract", **options):
        """
        初始化后端。

        参数:
            name: 后端名称标识
            **options: 后端特定选项
        """
        self._name = name
        self._options = options

    @property
    def name(self) -> str:
        """后端名称。"""
        return self._name

    @property
    @abstractmethod
    def max_qubits(self) -> int:
        """该后端支持的最大量子比特数。"""
        ...

    @abstractmethod
    def run_grover_search(self,
                           n_qubits: int,
                           oracle: Callable[[int], bool],
                           num_solutions: Optional[int],
                           shots: int,
                           verbose: bool = False,
                           **run_options) -> GroverResult:
        """
        在后端执行 Grover 搜索。

        参数:
            n_qubits: 量子比特数
            oracle: Boolean oracle 函数 f(x) -> bool
            num_solutions: 解的数量 M (None 则使用 BBHT 指数搜索)
            shots: 测量次数
            verbose: 是否输出过程信息
            **run_options: 后端特定选项 (如 oracle_expressions)

        返回:
            GroverResult 包含搜索完整结果

        失败模式 (由子类具体处理):
            - 量子比特数超出上限
            - Oracle 无法编译到目标硬件
            - 连接 / 认证失败
            - 作业超时
            - 测量结果解析失败
        """
        ...

    def validate_request(self, n_qubits: int, shots: int):
        """
        验证搜索请求的合法性。

        子类应调用此方法作为 run_grover_search 的第一步。

        失败模式:
            - n_qubits > max_qubits: 抛出异常
            - n_qubits < 1: 抛出异常
            - shots < 1: 抛出异常
        """
        from ..utils.validation import validate_n_qubits, validate_shots

        validate_n_qubits(n_qubits, self.max_qubits)
        validate_shots(shots)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self._name}>"
