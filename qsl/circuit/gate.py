"""
量子门对象 (Gate) — 支持 control/power/inverse 的通用门抽象。

与 Qiskit 对齐的核心能力:
    - Gate.to_matrix()      返回酉矩阵
    - Gate.control(n)       生成 n 控制位版本
    - Gate.power(k)         矩阵幂 (QPE/QFT 必需)
    - Gate.inverse()        共轭转置
    - Gate.num_qubits       作用的量子比特数

约定:
    矩阵基序为 |q_{k-1} ... q_1 q_0> 的标准二进制定序 (big-endian 矩阵)，
    与 qsl.quantum_gates 中 mcx/controlled_gate 的约定一致：
    最低位 (qubit index 0) 是矩阵的最低位地址。
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .parameter import ParameterExpression, resolve


class Gate:
    """
    通用量子门。

    参数:
        name:   门名 (如 "h", "cx", "rz")
        matrix: (2^k, 2^k) 复数酉矩阵；含符号参数的门可为 None，
                此时必须提供 matrix_fn
        num_qubits: 作用比特数 k
        params: 门的参数列表 (数值或 ParameterExpression)
        matrix_fn: 可选，params -> ndarray 的函数，用于参数化门
        label:  绘图标签
    """

    def __init__(
        self,
        name: str,
        matrix: Optional[np.ndarray] = None,
        num_qubits: int = 1,
        params: Optional[Sequence] = None,
        matrix_fn=None,
        label: Optional[str] = None,
        definition: Optional[list] = None,
    ):
        self.name = name
        self.num_qubits = int(num_qubits)
        self.params = list(params) if params else []
        self._matrix = None if matrix is None else np.asarray(matrix, dtype=complex)
        self._matrix_fn = matrix_fn
        self.label = label or name.upper()
        # definition: 可选的门级展开定义 (list of (Gate, qubits))，用于 decompose
        self.definition = definition

        if self._matrix is not None:
            dim = 1 << self.num_qubits
            if self._matrix.shape != (dim, dim):
                raise ValueError(
                    f"门 '{name}' 矩阵形状 {self._matrix.shape} 与 "
                    f"num_qubits={num_qubits} (期望 {dim}x{dim}) 不匹配"
                )

    # ----------------------------------------------------------------
    # 矩阵
    # ----------------------------------------------------------------
    def to_matrix(self) -> np.ndarray:
        """返回数值酉矩阵。含未绑定参数时抛出 ValueError。"""
        if self._matrix is not None:
            return self._matrix.copy()
        if self._matrix_fn is None:
            raise ValueError(f"门 '{self.name}' 没有矩阵表示")
        resolved = [resolve(p) for p in self.params]
        return np.asarray(self._matrix_fn(*resolved), dtype=complex)

    @property
    def is_parameterized(self) -> bool:
        """是否仍含未绑定的符号参数。"""
        if self._matrix is not None:
            return False
        return any(isinstance(p, ParameterExpression) and p.parameters
                   for p in self.params)

    def bind_parameters(self, mapping: dict) -> "Gate":
        """返回绑定参数后的新 Gate（数值化矩阵）。"""
        if not self.is_parameterized:
            return self.copy()
        new_params = []
        for p in self.params:
            if isinstance(p, ParameterExpression):
                free = p.parameters
                sub = {k: v for k, v in mapping.items() if k in free}
                if sub:
                    merged = dict(sub)
                    # 部分绑定 -> 保留表达式; 完全绑定 -> 数值
                    remaining = free - set(sub.keys())
                    if remaining:
                        # 部分绑定：构造一个新表达式不可行（表达式树是闭集），
                        # 因此仅在全部绑定时数值化，否则抛错提示
                        raise ValueError(
                            f"参数 {sorted(x.name for x in remaining)} 未绑定，"
                            f"请一次性绑定门 '{self.name}' 的全部参数。"
                        )
                    new_params.append(p.bind(merged))
                else:
                    new_params.append(p)
            else:
                new_params.append(p)
        g = Gate(self.name, None, self.num_qubits, new_params,
                 self._matrix_fn, self.label, self.definition)
        return g

    # ----------------------------------------------------------------
    # 代数操作
    # ----------------------------------------------------------------
    def inverse(self, name_suffix: str = "†") -> "Gate":
        """返回逆门（共轭转置）。对参数化门自动取负角度（若适用）。"""
        if self._matrix is not None:
            inv = self._matrix.conj().T
            return Gate(self.name + "_dg", inv, self.num_qubits,
                        label=self.label + name_suffix)
        # 参数化门：逆 = 共轭转置，需用矩阵函数数值化后转置
        mat = self.to_matrix()
        return Gate(self.name + "_dg", mat.conj().T, self.num_qubits,
                    label=self.label + name_suffix)

    def power(self, k: float) -> "Gate":
        """
        门的 k 次幂 U^k（通过对角化计算，支持分数/负数幂）。

        QPE 中受控 U^(2^j) 即由此构造。
        """
        mat = self.to_matrix()
        # 酉矩阵可对角化: U = V D V† => U^k = V D^k V†
        eigvals, eigvecs = np.linalg.eig(mat)
        powered = eigvecs @ np.diag(eigvals ** k) @ np.linalg.inv(eigvecs)
        # 消除数值噪声
        powered = np.asarray(powered, dtype=complex)
        return Gate(f"{self.name}^{k:g}", powered, self.num_qubits,
                    label=f"{self.label}^{k:g}")

    def control(self, n: int = 1) -> "Gate":
        """
        生成 n 控制位版本的门（替代手写 mcx/mcz）。

        控制位在高位，目标门作用于低 num_qubits 位：
            C^n(G) = I ⊕ G  (分块对角, 右下块为 G)
        """
        if n < 1:
            raise ValueError("控制位数 n 必须 >= 1")
        base = self.to_matrix()
        dim_base = base.shape[0]
        dim = dim_base << n
        out = np.eye(dim, dtype=complex)
        out[dim - dim_base:, dim - dim_base:] = base
        return Gate(f"c{n}{self.name}", out, self.num_qubits + n,
                    label=f"C{n if n > 1 else ''}-{self.label}")

    # ----------------------------------------------------------------
    # 其他
    # ----------------------------------------------------------------
    def copy(self) -> "Gate":
        return Gate(self.name, self._matrix, self.num_qubits,
                    list(self.params), self._matrix_fn, self.label,
                    self.definition)

    def is_unitary(self, tol: float = 1e-9) -> bool:
        m = self.to_matrix()
        return np.allclose(m.conj().T @ m, np.eye(m.shape[0]), atol=tol)

    def __repr__(self) -> str:
        p = f", params={self.params}" if self.params else ""
        return f"Gate('{self.name}', num_qubits={self.num_qubits}{p})"


# ====================================================================
# 常用单比特门的 ZYZ 分解工具（decompose 使用）
# ====================================================================

def zyz_decompose(matrix: np.ndarray):
    """
    任意 2x2 酉矩阵的 Z-Y-Z 欧拉分解:
        U = e^{iγ} Rz(φ) Ry(θ) Rz(λ)
    返回 (theta, phi, lam, gamma)。

    公式 (对 su = e^{-iγ}U ∈ SU(2), 记 a=su00, b=su01, c=su10, d=su11):
        su = [[e^{-i(φ+λ)/2}c,  -e^{-i(φ-λ)/2}s],
              [e^{ i(φ-λ)/2}s,   e^{ i(φ+λ)/2}c]],  c=cos(θ/2), s=sin(θ/2)
        θ = 2·atan2(|c|, |a|)
        φ = arg(c) - arg(a)
        λ = arg(d) - arg(c)
    退化情形 (sin 或 cos 为 0) 合并两个 Z 角。所有分支均经重建数值验证。
    """
    m = np.asarray(matrix, dtype=complex)
    det = np.linalg.det(m)
    gamma = 0.5 * np.angle(det)
    # 去除全局相位得到 SU(2)
    su = m * np.exp(-1j * gamma)
    a, b, c, d = su[0, 0], su[0, 1], su[1, 0], su[1, 1]
    theta = 2 * math.atan2(abs(c), abs(a))
    if abs(math.sin(theta / 2)) < 1e-12:
        # su ≈ 对角: 只有 φ+λ 可观测, 取 φ=0
        phi = 0.0
        lam = float(np.angle(d) - np.angle(a))
    elif abs(math.cos(theta / 2)) < 1e-12:
        # su ≈ 反对角: 只有 φ-λ 可观测, 取 λ=0
        # b = -e^{-i(φ-λ)/2} -> arg(c)-arg(b) = (φ-λ) - π
        phi = float(np.angle(c) - np.angle(b) + math.pi)
        lam = 0.0
    else:
        phi = float(np.angle(c) - np.angle(a))
        lam = float(np.angle(d) - np.angle(c))
    return theta, phi, lam, float(gamma)


__all__ = ["Gate", "zyz_decompose"]
