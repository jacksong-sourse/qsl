"""
QSL 程序数据结构。

一个 QSLProgram 描述一个完整的量子搜索问题，包含:
    - name: 程序名称
    - n_qubits: 量子比特数
    - premises: 布尔约束表达式列表
    - tools: 工具声明
    - question: 问题描述
    - main: 主算法配置

失败模式分析:
    1. n_qubits <= 0: 无物理意义
    2. premises 为空: 相当于搜索整个空间 (合法但意义不大)
    3. name 为空: 无标识的程序
    4. shots <= 0: 至少需要一次测量
    5. 不支持除 grover 外的算法: 保留扩展空间
"""

from dataclasses import dataclass, field
from typing import List, Optional

from ..core.state import MAX_QUBITS
from ..utils.validation import (
    validate_n_qubits,
    validate_shots,
    validate_premises,
)


@dataclass
class QSLProgram:
    """
    QSL 程序的完整定义。

    使用示例:
        >>> program = QSLProgram(
        ...     name="3-SAT 求解",
        ...     n_qubits=3,
        ...     premises=["x0 | ~x1", "x1 | x2", "~x0 | ~x2"],
        ...     shots=10
        ... )
    """

    name: str
    n_qubits: int
    premises: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    question: str = ""
    main_algorithm: str = "grover"
    shots: int = 1
    backend: str = "simulator"  # "simulator" 或 IBM 后端名称
    backend_options: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        自动验证字段。

        失败模式:
            - 任意字段不合法: 抛出对应异常

        注意: n_qubits=0 时跳过验证 (允许 DSL 解析器延迟设置 qubits)
        """
        if self.n_qubits != 0:
            self.validate()

    def validate(self):
        """
        完整验证程序定义。

        失败模式:
            - n_qubits 不合法
            - premises 不合法
            - shots 不合法
            - name 为空
            - main_algorithm 不在支持列表中
        """
        validate_n_qubits(self.n_qubits, MAX_QUBITS)
        validate_shots(self.shots)

        if not self.name or not self.name.strip():
            raise ValueError("程序名称不能为空")

        if self.premises:
            validate_premises(self.premises, self.n_qubits)

        SUPPORTED_ALGORITHMS = {"grover", "shor", "qaoa", "vqe"}
        if self.main_algorithm.lower() not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"不支持的算法: '{self.main_algorithm}'。"
                f"可用算法: {SUPPORTED_ALGORITHMS}"
            )

        if self.backend_options is None:
            self.backend_options = {}

    def to_dict(self) -> dict:
        """转换为字典 (用于序列化)。"""
        return {
            "name": self.name,
            "n_qubits": self.n_qubits,
            "premises": self.premises.copy(),
            "tools": self.tools.copy(),
            "question": self.question,
            "main_algorithm": self.main_algorithm,
            "shots": self.shots,
            "backend": self.backend,
            "backend_options": self.backend_options.copy(),
        }

    def copy_with(self, **updates) -> 'QSLProgram':
        """
        创建修改了部分字段的副本。

        参数:
            **updates: 要修改的字段和值

        返回:
            新的 QSLProgram 实例
        """
        data = self.to_dict()
        data.update(updates)
        return QSLProgram(**data)

    def __repr__(self) -> str:
        return (
            f"QSLProgram(name='{self.name}', "
            f"n_qubits={self.n_qubits}, "
            f"premises={len(self.premises)}, "
            f"shots={self.shots}, "
            f"backend='{self.backend}')"
        )

    def __str__(self) -> str:
        lines = [
            f"QSLProgram: {self.name}",
            f"  量子比特数: {self.n_qubits} (搜索空间: 2^{self.n_qubits} = {1 << self.n_qubits})",
            f"  前提 ({len(self.premises)}): {self.premises}",
            f"  算法: {self.main_algorithm}",
            f"  测量次数: {self.shots}",
            f"  后端: {self.backend}",
        ]
        return '\n'.join(lines)
