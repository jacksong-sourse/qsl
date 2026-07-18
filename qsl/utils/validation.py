"""
输入验证工具函数。

所有公共 API 的输入都经过此模块的验证，确保在错误发生时
能够给出清晰、可操作的错误信息，而非模糊的运行时异常。
"""

from typing import List, Optional
from .exceptions import (
    InvalidQubitCountError,
    QubitIndexError,
    DuplicateQubitError,
    ProgramValidationError,
)


# --- 量子比特数验证 ---

def validate_n_qubits(n_qubits: int, max_qubits: int = 20) -> int:
    """
    验证量子比特数。

    参数:
        n_qubits: 量子比特数
        max_qubits: 模拟器支持的最大量子比特数

    返回:
        验证通过的量子比特数

    失败模式:
        - n_qubits < 1: 没有量子比特无法计算
        - n_qubits > max_qubits: 2^n 个复数振幅超出经典内存
    """
    if not isinstance(n_qubits, int):
        raise TypeError(f"量子比特数必须是整数，当前类型: {type(n_qubits).__name__}")
    if n_qubits < 1:
        raise InvalidQubitCountError(n_qubits, max_qubits)
    if n_qubits > max_qubits:
        raise InvalidQubitCountError(n_qubits, max_qubits)
    return n_qubits


# --- 量子比特索引验证 ---

def validate_qubit_index(index: int, n_qubits: int) -> int:
    """
    验证单个量子比特索引。

    失败模式:
        - index < 0: 负数索引无意义
        - index >= n_qubits: 索引越界
    """
    if not isinstance(index, int):
        raise TypeError(
            f"量子比特索引必须是整数，当前类型: {type(index).__name__}"
        )
    if index < 0 or index >= n_qubits:
        raise QubitIndexError(index, n_qubits)
    return index


def validate_qubit_indices(indices: List[int], n_qubits: int,
                            allow_empty: bool = False) -> List[int]:
    """
    验证量子比特索引列表。

    失败模式:
        - 列表中某个索引越界
        - 列表中有重复索引（对多数门操作而言这是配置错误）
        - 列表为空且 allow_empty=False
    """
    if not allow_empty and len(indices) == 0:
        raise ValueError("量子比特索引列表不能为空")
    for idx in indices:
        validate_qubit_index(idx, n_qubits)
    # 检查重复
    if len(indices) != len(set(indices)):
        duplicates = [idx for idx in indices if indices.count(idx) > 1]
        raise DuplicateQubitError(duplicates)
    return indices


# --- 程序定义验证 ---

def validate_program_field(field_name: str, value, min_val=None, max_val=None):
    """
    验证 QSLProgram 的字段值。

    失败模式:
        - value 为 None
        - value 超出 [min_val, max_val] 范围
    """
    if value is None:
        raise ProgramValidationError(field_name, value, "不能为 None")
    if min_val is not None and value < min_val:
        raise ProgramValidationError(
            field_name, value, f"不能小于 {min_val}"
        )
    if max_val is not None and value > max_val:
        raise ProgramValidationError(
            field_name, value, f"不能大于 {max_val}"
        )


def validate_premises(premises: List[str], n_qubits: int) -> List[str]:
    """
    验证前提表达式列表。

    失败模式:
        - premises 为 None 或空列表
        - 表达式引用的变量索引超出 n_qubits 范围
        - 表达式语法错误（由解析器检测）
    """
    if premises is None:
        raise ProgramValidationError("premises", None, "前提列表不能为 None")
    if not isinstance(premises, list):
        raise ProgramValidationError(
            "premises", premises,
            f"前提必须是字符串列表，当前类型: {type(premises).__name__}"
        )
    if len(premises) == 0:
        return premises  # 空前提允许 (搜索整个空间，合法但意义不大)
    for i, p in enumerate(premises):
        if not isinstance(p, str):
            raise ProgramValidationError(
                f"premises[{i}]", p,
                f"前提必须是字符串，当前类型: {type(p).__name__}"
            )
        if not p.strip():
            raise ProgramValidationError(
                f"premises[{i}]", p, "前提表达式不能为空字符串"
            )
    return premises


def validate_shots(shots: int) -> int:
    """
    验证测量次数。

    失败模式:
        - shots < 1: 至少需要一次测量
        - shots > 100000: 防止无意义的超大量测量（会 OOM）
    """
    if not isinstance(shots, int):
        raise ProgramValidationError(
            "shots", shots,
            f"测量次数必须是整数，当前类型: {type(shots).__name__}"
        )
    if shots < 1:
        raise ProgramValidationError("shots", shots, "测量次数必须 >= 1")
    if shots > 100000:
        raise ProgramValidationError("shots", shots, "测量次数必须 <= 100000")
    return shots


# --- 概率值验证 ---

def validate_probability(prob: float, name: str = "概率") -> float:
    """
    验证概率值在 [0, 1] 范围内。

    失败模式:
        - prob < 0 或 prob > 1: 概率值不合法
        - prob 为 NaN 或 inf
    """
    import math
    if math.isnan(prob) or math.isinf(prob):
        raise ValueError(f"{name} 不是合法数值: {prob}")
    if prob < -1e-12 or prob > 1.0 + 1e-12:
        raise ValueError(
            f"{name} 超出 [0, 1] 范围: {prob:.10f}"
        )
    return max(0.0, min(1.0, prob))  # 钳位到 [0, 1]
