"""
布尔表达式相位 Oracle 量子电路编译器。

将 BooleanExpr AST 直接编译为可逆量子电路（X / CNOT / Toffoli / Z 门），
完全不经过 O(2^n) 的经典解空间枚举：

    1. 自底向上递归地为每个子表达式分配 ancilla 量子比特，
       用 Toffoli/CNOT/X 门以可逆方式计算其子表达式的值；
    2. 在保存整个表达式结果的量子比特上施加 Z 门
       （|x>|f(x)> -> (-1)^{f(x)} |x>|f(x)>）；
    3. 逆计算（uncompute）将所有 ancilla 恢复到 |0>。

Oracle 电路规模为 O(表达式长度)，每次 Grover 迭代查询一次 Oracle，
保持了 Grover 算法 O(√N) 的量子查询复杂度。

生成的门列表是后端无关的，模拟器直接作用于 QuantumState，
IBM 后端将其翻译为对应的 Qiskit 门。
"""

from typing import List, Tuple

from .parser import BooleanExpr, VarExpr, NotExpr, AndExpr, OrExpr, XorExpr

# 门元组: (名称, 量子比特索引)
#   ("X", (q,))            - Pauli-X
#   ("Z", (q,))            - Pauli-Z (相位翻转)
#   ("CNOT", (c, t))       - 受控非
#   ("TOFFOLI", (c1, c2, t)) - 双控制非
GateTuple = Tuple[str, Tuple[int, ...]]


class OracleCircuit:
    """
    编译得到的相位 Oracle 电路。

    属性:
        gates: 门元组列表，按应用顺序排列
        n_ancilla: 需要的 ancilla 量子比特数（位于主寄存器之后，
                   索引为 n_qubits .. n_qubits + n_ancilla - 1）
    """

    __slots__ = ("gates", "n_ancilla")

    def __init__(self, gates: List[GateTuple], n_ancilla: int):
        self.gates = gates
        self.n_ancilla = n_ancilla

    def __len__(self) -> int:
        return len(self.gates)

    def __repr__(self) -> str:
        return f"OracleCircuit(gates={len(self.gates)}, n_ancilla={self.n_ancilla})"


class BooleanOracleCompiler:
    """
    将布尔表达式编译为 ancilla 辅助的相位 Oracle 电路。

    参数:
        n_qubits: 主寄存器量子比特数（变量 x0..x_{n-1}）
    """

    def __init__(self, n_qubits: int):
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        self.n = n_qubits
        self._n_ancilla = 0

    # ----------------------------------------------------------------
    # ancilla 管理
    # ----------------------------------------------------------------

    def _alloc(self) -> int:
        """分配一个新的 ancilla 量子比特。

        注意: ancilla 不做垃圾复用 —— 可逆计算中保存中间值的
        ancilla 是"脏"的, 直接复用会导致新值与旧垃圾异或而损坏
        计算结果。所有 ancilla 在电路末尾的逆计算中统一恢复为 |0>。
        """
        q = self.n + self._n_ancilla
        self._n_ancilla += 1
        return q

    def _release(self, q: int):
        """标记子表达式结果已消费 (仅语义标记, 不做物理复用)。"""
        pass

    # ----------------------------------------------------------------
    # 编译入口
    # ----------------------------------------------------------------

    def compile(self, expr: BooleanExpr) -> OracleCircuit:
        """
        编译表达式为相位 Oracle 电路。

        电路效果: |x>|0...0> -> (-1)^{f(x)} |x>|0...0>

        参数:
            expr: BooleanExpr AST（变量索引必须 < n_qubits）

        返回:
            OracleCircuit

        失败模式:
            - 变量索引超出主寄存器: 抛出 ValueError
        """
        self._validate_indices(expr)
        compute: List[GateTuple] = []
        out = self._emit(expr, compute)
        # 在结果比特上施加 Z，然后逆计算恢复 ancilla
        gates = list(compute)
        gates.append(("Z", (out,)))
        gates.extend(reversed(compute))  # 所有门均自逆
        return OracleCircuit(gates, self._n_ancilla)

    def _validate_indices(self, expr: BooleanExpr):
        """确保表达式引用的变量索引都在主寄存器范围内。"""
        stack = [expr]
        while stack:
            node = stack.pop()
            if isinstance(node, VarExpr):
                if node.index < 0 or node.index >= self.n:
                    raise ValueError(
                        f"变量 x{node.index} 超出主寄存器范围 "
                        f"[0, {self.n - 1}]"
                    )
            elif isinstance(node, NotExpr):
                stack.append(node.expr)
            elif isinstance(node, (AndExpr, OrExpr, XorExpr)):
                stack.append(node.left)
                stack.append(node.right)

    # ----------------------------------------------------------------
    # 递归门生成
    # ----------------------------------------------------------------

    def _emit(self, expr: BooleanExpr, gates: List[GateTuple]) -> int:
        """
        为子表达式生成计算门，返回保存其值的量子比特索引。

        若返回的是主寄存器比特（变量本身），调用方不得修改它；
        若返回的是 ancilla，调用方使用完毕后应通过 _release 释放。
        """
        if isinstance(expr, VarExpr):
            # 变量的值就在主寄存器中，无需任何门
            return expr.index

        if isinstance(expr, NotExpr):
            q = self._emit(expr.expr, gates)
            if q < self.n:
                # 子表达式是变量：复制到 ancilla 再取反，避免改动主寄存器
                a = self._alloc()
                gates.append(("CNOT", (q, a)))
                gates.append(("X", (a,)))
                return a
            # 子表达式已在 ancilla 中：直接原地取反
            gates.append(("X", (q,)))
            return q

        if isinstance(expr, AndExpr):
            ql = self._emit(expr.left, gates)
            qr = self._emit(expr.right, gates)
            a = self._alloc()
            if ql == qr:
                # a & a = a: 直接复制，避免控制位重复的非法 Toffoli
                gates.append(("CNOT", (ql, a)))
            else:
                gates.append(("TOFFOLI", (ql, qr, a)))
            self._release(ql)
            self._release(qr)
            return a

        if isinstance(expr, OrExpr):
            # De Morgan: a | b = ~(~a & ~b)
            ql = self._emit(expr.left, gates)
            qr = self._emit(expr.right, gates)
            a = self._alloc()
            if ql == qr:
                # a | a = a: 直接复制
                gates.append(("CNOT", (ql, a)))
            else:
                gates.append(("X", (ql,)))
                gates.append(("X", (qr,)))
                gates.append(("TOFFOLI", (ql, qr, a)))
                gates.append(("X", (ql,)))
                gates.append(("X", (qr,)))
                gates.append(("X", (a,)))
            self._release(ql)
            self._release(qr)
            return a

        if isinstance(expr, XorExpr):
            ql = self._emit(expr.left, gates)
            qr = self._emit(expr.right, gates)
            a = self._alloc()
            gates.append(("CNOT", (ql, a)))
            gates.append(("CNOT", (qr, a)))
            self._release(ql)
            self._release(qr)
            return a

        raise TypeError(
            f"不支持的表达式节点类型: {type(expr).__name__}"
        )


def compile_phase_oracle(expressions: List[BooleanExpr],
                         n_qubits: int) -> OracleCircuit:
    """
    将多个布尔表达式（逻辑与关系）编译为单个相位 Oracle 电路。

    所有表达式必须同时为真时才施加 -1 相位:
        Oracle|x> = (-1)^{f_1(x) AND ... AND f_k(x)} |x>

    参数:
        expressions: BooleanExpr 列表（至少一个）
        n_qubits: 主寄存器量子比特数

    返回:
        OracleCircuit
    """
    if not expressions:
        raise ValueError("expressions 不能为空")
    if len(expressions) == 1:
        combined = expressions[0]
    else:
        combined = expressions[0]
        for e in expressions[1:]:
            combined = AndExpr(combined, e)
    return BooleanOracleCompiler(n_qubits).compile(combined)
