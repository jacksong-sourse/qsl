"""
QuantumCircuit — 对标 Qiskit 的量子电路对象模型。

核心能力:
    - 逐门追加 append / 便捷方法 h(0), cx(0,1), ...
    - 插入 insert / 删除 remove / 清空 clear
    - 逆置 inverse() (circuit.inverse())
    - 拼接 compose(other)
    - 参数化 Parameter + bind_parameters()
    - 等价变换 decompose() / transpile()
    - 通用受控 gate.control(n) / 幂次 gate.power(k) / 逆 gate.inverse()
    - 执行 execute() -> counts, 态向量 statevector()
    - 序列化 to_json()/from_json()

比特序约定 (与 qsl 全系一致, 与 Qiskit statevector 一致):
    - 态向量索引 i 的第 q 位 = 量子比特 q 的值 (qubit 0 = LSB)
    - k 比特门矩阵作用在 qubits=(t0,...,t_{k-1}) 上时,
      矩阵行/列索引 = Σ_j bit(t_j) · 2^{k-1-j} (t0 为最高位)
    - 即量子门矩阵的基序为 |t0 t1 ... t_{k-1}> (big-endian 列出的比特)
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .. import quantum_gates as G
from .gate import Gate, zyz_decompose
from .parameter import Parameter, ParameterExpression, resolve


class Instruction:
    """一条电路指令: 一个门 + 作用的量子比特元组。"""

    __slots__ = ("gate", "qubits")

    def __init__(self, gate: Gate, qubits: Sequence[int]):
        self.gate = gate
        self.qubits = tuple(int(q) for q in qubits)

    def copy(self) -> "Instruction":
        return Instruction(self.gate.copy(), self.qubits)

    def __repr__(self) -> str:
        return f"Instruction({self.gate.name}, {self.qubits})"


class QuantumCircuit:
    """
    量子电路对象。

    参数:
        num_qubits: 量子比特数
        name: 电路名
        global_phase: 全局相位 (弧度), 数值上与 Qiskit 对齐

    示例:
        >>> qc = QuantumCircuit(2)
        >>> qc.h(0); qc.cx(0, 1)
        >>> print(qc.draw())
        >>> counts = qc.execute(shots=1000)
    """

    def __init__(self, num_qubits: int, name: str = "", global_phase: float = 0.0):
        if num_qubits < 1:
            raise ValueError("num_qubits 必须 >= 1")
        self.num_qubits = int(num_qubits)
        self.name = name
        self.global_phase = float(global_phase)
        self.data: List[Instruction] = []

    # ================================================================
    # 内部校验
    # ================================================================
    def _check_qubits(self, qubits: Sequence[int]):
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"量子比特重复: {qubits}")
        for q in qubits:
            if not (0 <= q < self.num_qubits):
                raise ValueError(
                    f"量子比特 {q} 越界 (电路只有 {self.num_qubits} 比特, "
                    f"合法范围 0..{self.num_qubits - 1})"
                )

    # ================================================================
    # 基础操作: append / insert / remove / clear
    # ================================================================
    def append(self, gate: Gate, qubits: Sequence[int]) -> "QuantumCircuit":
        """在电路末尾追加一个门。返回 self 支持链式调用。"""
        qubits = tuple(qubits)
        self._check_qubits(qubits)
        if len(qubits) != gate.num_qubits:
            raise ValueError(
                f"门 '{gate.name}' 是 {gate.num_qubits} 比特门, "
                f"但给了 {len(qubits)} 个比特 {qubits}"
            )
        self.data.append(Instruction(gate, qubits))
        return self

    def insert(self, index: int, gate: Gate, qubits: Sequence[int]) -> "QuantumCircuit":
        """在 index 位置插入一个门。"""
        qubits = tuple(qubits)
        self._check_qubits(qubits)
        if len(qubits) != gate.num_qubits:
            raise ValueError(
                f"门 '{gate.name}' 是 {gate.num_qubits} 比特门, "
                f"但给了 {len(qubits)} 个比特"
            )
        self.data.insert(index, Instruction(gate, qubits))
        return self

    def remove(self, index: int) -> Instruction:
        """删除并返回 index 位置的指令。"""
        return self.data.pop(index)

    def clear(self):
        """清空所有指令(保留比特数和名称)。"""
        self.data.clear()

    def copy(self) -> "QuantumCircuit":
        qc = QuantumCircuit(self.num_qubits, self.name, self.global_phase)
        qc.data = [inst.copy() for inst in self.data]
        return qc

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    # ================================================================
    # 统计
    # ================================================================
    def size(self) -> int:
        """指令总数。"""
        return len(self.data)

    def depth(self) -> int:
        """电路深度 (贪心层调度)。"""
        last_layer = [0] * self.num_qubits
        for inst in self.data:
            layer = max(last_layer[q] for q in inst.qubits) + 1
            for q in inst.qubits:
                last_layer[q] = layer
        return max(last_layer) if self.data else 0

    def count_ops(self) -> Dict[str, int]:
        """按门名统计数量。"""
        out: Dict[str, int] = {}
        for inst in self.data:
            out[inst.gate.name] = out.get(inst.gate.name, 0) + 1
        return out

    def width(self) -> int:
        """电路宽度 (量子比特数, Qiskit 兼容)。"""
        return self.num_qubits

    @property
    def n_qubits(self) -> int:
        """量子比特数 (num_qubits 的别名, 接口统一)。"""
        return self.num_qubits

    def num_nonlocal_gates(self) -> int:
        """非局部门数 (作用于 ≥2 比特的门, barrier 不计)。"""
        return sum(1 for inst in self.data
                   if inst.gate.num_qubits >= 2 and inst.gate.name != "barrier")

    def num_tensor_factors(self) -> int:
        """张量因子数 (同 width, Qiskit 兼容)。"""
        return self.num_qubits

    @property
    def qubits(self) -> List[int]:
        """虚拟量子比特索引列表 (Qiskit 兼容 API)。"""
        return list(range(self.num_qubits))

    @property
    def clbits(self) -> List[int]:
        """经典比特 (QSL 当前无经典比特, 返回空列表以兼容)。"""
        return []

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    @property
    def parameters(self) -> set:
        out = set()
        for inst in self.data:
            for p in inst.gate.params:
                if isinstance(p, ParameterExpression):
                    out |= p.parameters
        return out

    def summary(self) -> str:
        """电路统计摘要字符串。"""
        ops = self.count_ops()
        ops_str = ", ".join(f"{k}:{v}" for k, v in sorted(ops.items()))
        return (
            f"QuantumCircuit('{self.name}', qubits={self.num_qubits}, "
            f"gates={self.size()}, depth={self.depth()}, "
            f"params={self.num_parameters}, ops={{{ops_str}}})"
        )

    # ================================================================
    # 参数化
    # ================================================================
    def bind_parameters(self, mapping: Dict[Parameter, float],
                        inplace: bool = False) -> "QuantumCircuit":
        """
        将符号参数绑定为数值, 返回绑定后的新电路 (或就地)。

        参数:
            mapping: {Parameter: value} 或 {参数名: value}
        """
        # 允许用字符串名做 key
        norm: Dict[Parameter, float] = {}
        for k, v in mapping.items():
            if isinstance(k, Parameter):
                norm[k] = v
            else:
                norm[Parameter(str(k))] = v

        qc = self if inplace else self.copy()
        for inst in qc.data:
            if inst.gate.is_parameterized:
                inst.gate = inst.gate.bind_parameters(norm)
        return qc

    def assign_parameters(self, mapping, inplace: bool = False) -> "QuantumCircuit":
        """bind_parameters 的别名 (Qiskit 兼容)。"""
        return self.bind_parameters(mapping, inplace=inplace)

    def assign(self, mapping, inplace: bool = False) -> "QuantumCircuit":
        """assign_parameters 的简写。"""
        return self.bind_parameters(mapping, inplace=inplace)

    # ================================================================
    # 逆置与拼接
    # ================================================================
    def inverse(self, inplace: bool = False) -> "QuantumCircuit":
        """逆电路: 逆序 + 每门取逆。"""
        qc = self if inplace else self.copy()
        qc.data = [Instruction(inst.gate.inverse(), inst.qubits)
                   for inst in reversed(qc.data)]
        qc.global_phase = -qc.global_phase
        return qc

    def compose(self, other: "QuantumCircuit",
                qubits: Optional[Sequence[int]] = None,
                inplace: bool = False,
                front: bool = False) -> "QuantumCircuit":
        """
        拼接另一个电路。

        参数:
            other: 要拼接的电路
            qubits: other 的比特映射到本电路的比特 (默认恒等映射)。
                    qubits[i] 表示 other 的第 i 个比特放到本电路的哪个比特。
            inplace: 是否就地修改
            front: True 时把 other 插到本电路之前
        """
        if qubits is None:
            if other.num_qubits > self.num_qubits:
                raise ValueError(
                    f"被拼接电路有 {other.num_qubits} 比特, "
                    f"超过本电路 {self.num_qubits} 比特; 请用 qubits= 指定映射"
                )
            qubits = list(range(other.num_qubits))
        if len(qubits) != other.num_qubits:
            raise ValueError("qubits 映射长度必须等于 other.num_qubits")

        qc = self if inplace else self.copy()
        mapped = [Instruction(inst.gate.copy(), tuple(qubits[q] for q in inst.qubits))
                  for inst in other.data]
        if front:
            qc.data = mapped + qc.data
        else:
            qc.data = qc.data + mapped
        return qc

    def tensor(self, other: "QuantumCircuit") -> "QuantumCircuit":
        """张量积拼接: other 的比特放在高位 (返回新电路)。"""
        qc = QuantumCircuit(self.num_qubits + other.num_qubits,
                            name=f"{self.name}⊗{other.name}")
        qc.data = [inst.copy() for inst in self.data]
        offset = self.num_qubits
        qc.data += [Instruction(inst.gate.copy(),
                                tuple(q + offset for q in inst.qubits))
                    for inst in other.data]
        return qc

    # ================================================================
    # 门方法 — 单比特
    # ================================================================
    def _add1(self, name, matrix, q, label=None, matrix_fn=None, params=None):
        gate = Gate(name, matrix, 1, params=params, matrix_fn=matrix_fn,
                    label=label)
        return self.append(gate, (q,))

    def id(self, q):
        return self._add1("id", G.I, q, "I")

    def x(self, q):
        return self._add1("x", G.X, q, "X")

    def y(self, q):
        return self._add1("y", G.Y, q, "Y")

    def z(self, q):
        return self._add1("z", G.Z, q, "Z")

    def h(self, q):
        return self._add1("h", G.H, q, "H")

    def s(self, q):
        return self._add1("s", G.S, q, "S")

    def sdg(self, q):
        return self._add1("sdg", G.Sdg, q, "S†")

    def t(self, q):
        return self._add1("t", G.T, q, "T")

    def tdg(self, q):
        return self._add1("tdg", G.Tdg, q, "T†")

    def sx(self, q):
        return self._add1("sx", G.SX, q, "√X")

    def sxdg(self, q):
        return self._add1("sxdg", G.SXdg, q, "√X†")

    def rx(self, q, theta):
        return self._add1("rx", None, q, "RX", G.rx, [theta])

    def ry(self, q, theta):
        return self._add1("ry", None, q, "RY", G.ry, [theta])

    def rz(self, q, phi):
        return self._add1("rz", None, q, "RZ", G.rz, [phi])

    def p(self, q, lam):
        return self._add1("p", None, q, "P", G.p, [lam])

    def u(self, q, theta, phi, lam):
        return self._add1("u", None, q, "U", G.u3, [theta, phi, lam])

    # ================================================================
    # 门方法 — 两比特
    # ================================================================
    def _add2(self, name, matrix, q0, q1, label=None, matrix_fn=None, params=None):
        gate = Gate(name, matrix, 2, params=params, matrix_fn=matrix_fn,
                    label=label)
        return self.append(gate, (q0, q1))

    def cx(self, control, target):
        # 矩阵基序 |control target> (control 高位)
        mat = np.eye(4, dtype=complex)
        mat[2, 2] = 0; mat[3, 3] = 0
        mat[2, 3] = 1; mat[3, 2] = 1
        return self._add2("cx", mat, control, target, "X")

    def cy(self, control, target):
        return self._add2("cy", G.controlled_gate(G.Y, 1), control, target, "Y")

    def cz(self, control, target):
        return self._add2("cz", G.controlled_gate(G.Z, 1), control, target, "Z")

    def ch(self, control, target):
        return self._add2("ch", G.ch(), control, target, "H")

    def cs(self, control, target):
        return self._add2("cs", G.cs(), control, target, "S")

    def csdg(self, control, target):
        return self._add2("csdg", G.csdg(), control, target, "S†")

    def ct(self, control, target):
        return self._add2("ct", G.ct(), control, target, "T")

    def ctdg(self, control, target):
        return self._add2("ctdg", G.ctdg(), control, target, "T†")

    def crx(self, theta, control, target):
        return self._add2("crx", None, control, target, "RX", G.crx, [theta])

    def cry(self, theta, control, target):
        return self._add2("cry", None, control, target, "RY", G.cry, [theta])

    def crz(self, theta, control, target):
        return self._add2("crz", None, control, target, "RZ", G.crz, [theta])

    def cp(self, lam, control, target):
        return self._add2("cp", None, control, target, "P", G.cp, [lam])

    def cu(self, theta, phi, lam, control, target, gamma=0.0):
        return self._add2("cu", None, control, target, "U", G.cu,
                          [theta, phi, lam, gamma])

    def swap(self, q0, q1):
        return self._add2("swap", G.swap(), q0, q1, "×")

    def iswap(self, q0, q1):
        return self._add2("iswap", G.iswap(), q0, q1, "i×")

    def dcx(self, q0, q1):
        return self._add2("dcx", G.dcx(), q0, q1, "DCX")

    def rxx(self, theta, q0, q1):
        return self._add2("rxx", None, q0, q1, "RXX", G.rxx, [theta])

    def ryy(self, theta, q0, q1):
        return self._add2("ryy", None, q0, q1, "RYY", G.ryy, [theta])

    def rzz(self, theta, q0, q1):
        return self._add2("rzz", None, q0, q1, "RZZ", G.rzz, [theta])

    def ecr(self, q0, q1):
        return self._add2("ecr", G.ecr(), q0, q1, "ECR")

    # ================================================================
    # 门方法 — 三比特及以上
    # ================================================================
    def ccx(self, c1, c2, target):
        return self.append(Gate("ccx", G.mcx(2), 3, label="X"),
                           (c1, c2, target))

    def cswap(self, control, q0, q1):
        return self.append(Gate("cswap", G.cswap(), 3, label="×"),
                           (control, q0, q1))

    def mcx(self, controls: Sequence[int], target: int):
        """多控制 X 门 (controls 任意个, 自动生成)。"""
        controls = list(controls)
        n = len(controls)
        return self.append(Gate(f"mcx{n}", G.mcx(n), n + 1, label="X"),
                           tuple(controls) + (target,))

    def mcz(self, qubits: Sequence[int]):
        """多控制 Z 门: 全部比特为 1 时相位翻转。"""
        qs = list(qubits)
        n = len(qs)
        dim = 1 << n
        mat = np.eye(dim, dtype=complex)
        mat[-1, -1] = -1
        return self.append(Gate(f"mcz{n}", mat, n, label="Z"), tuple(qs))

    def unitary(self, matrix, qubits: Sequence[int], label: str = "U"):
        """追加任意酉矩阵门。"""
        matrix = np.asarray(matrix, dtype=complex)
        k = len(qubits)
        if matrix.shape != (1 << k, 1 << k):
            raise ValueError(
                f"酉矩阵形状 {matrix.shape} 与 {k} 比特不匹配"
            )
        return self.append(Gate("unitary", matrix, k, label=label), tuple(qubits))

    def barrier(self, qubits: Optional[Sequence[int]] = None):
        """逻辑屏障 (仅可视化/编译提示, 不影响数值)。"""
        qs = tuple(qubits) if qubits is not None else tuple(range(self.num_qubits))
        mat = np.eye(1 << len(qs), dtype=complex)
        return self.append(Gate("barrier", mat, len(qs), label="|"), qs)

    def measure_all(self, qubits: Optional[Sequence[int]] = None,
                    shots: int = 1024, seed: Optional[int] = None):
        """
        便捷方法: 直接执行电路并返回测量计数 (dict[int,int])。
        不修改电路本身; 如需在电路中插入测量操作, 请使用 barrier 或其他方式。
        """
        res = self.execute(shots=shots, seed=seed)
        return res.counts

    def reverse_bits(self) -> "QuantumCircuit":
        """
        返回一个新电路, 所有比特索引按 [n-1,...,0] 反转。
        对 QFT/位序处理很有用。
        """
        n = self.num_qubits
        qc = QuantumCircuit(n, name=self.name + "(rev)" if self.name else "",
                            global_phase=self.global_phase)
        for inst in self.data:
            rev_qs = tuple(n - 1 - q for q in inst.qubits)
            qc.append(inst.gate.copy(), rev_qs)
        return qc

    def get_instructions(self, name: str) -> List[Tuple[Instruction]]:
        """按门名筛选指令。"""
        return [inst for inst in self.data if inst.gate.name == name]

    def has_register(self, name: str) -> bool:
        """兼容性占位 (QSL 当前无量子寄存器概念)。"""
        return False

    def remove_final_measurements(self) -> "QuantumCircuit":
        """Qiskit 兼容空操作 —— QSL 电路中不保存测量指令, 返回 self。"""
        return self

    # ================================================================
    # 受控/幂次通用接口
    # ================================================================
    def controlled(self, gate: Gate, controls: Sequence[int],
                   target_qubits: Sequence[int]) -> "QuantumCircuit":
        """把任意门作为受控门加入电路: gate.control(len(controls))。"""
        controls = tuple(controls)
        targets = tuple(target_qubits)
        cg = gate.control(len(controls))
        return self.append(cg, controls + targets)

    # ================================================================
    # 等价变换
    # ================================================================
    def decompose(self, basis: Optional[set] = None,
                    reps: int = 10) -> "QuantumCircuit":
        """
        把多比特门拆解到基础门集。

        默认基础门集: {cx, id, rz, ry, sx, x, z, h, s, sdg, t, tdg, p, swap}
        当前实现:
            - 单比特任意酉 -> ZYZ 分解 (rz-ry-rz, 全局相位累积到 global_phase)
            - 受控单比特门 -> 标准 2-CNOT 分解 (Barenco)
            - ccx -> 6-CNOT 分解; cswap -> ccx+cx
            - rzz -> cx+rz; rxx/ryy -> 共轭 rzz; iswap -> cz+swap+s
            - dcx -> cx+cx; ecr -> h+cx+固定单比特旋转
            - 其它 >=3 比特任意酉: 抛 NotImplementedError (提示先分解)

        参数:
            basis: 目标门名集合 (默认 None 用上述集合)
            reps: 递归分解最大轮数 (默认 10, 通常 2~3 轮即收敛)
        """
        basis = basis or {"cx", "id", "rz", "ry", "sx", "x", "z", "h",
                          "s", "sdg", "t", "tdg", "p", "swap"}
        qc = self.copy()
        for _ in range(reps):
            new_data: List[Instruction] = []
            changed = False
            for inst in qc.data:
                name = inst.gate.name
                if name in basis or name == "barrier":
                    new_data.append(inst)
                    continue
                expanded = _expand_instruction(inst, self.num_qubits)
                if expanded is None:
                    new_data.append(inst)
                else:
                    insts, phase = expanded
                    new_data.extend(insts)
                    qc.global_phase += phase
                    changed = True
            qc.data = new_data
            if not changed:
                break
        return qc

    def transpile(self, basis_gates: Optional[set] = None,
                    coupling_map: Optional[List[Tuple[int, int]]] = None,
                    optimization_level: int = 1,
                    seed: Optional[int] = None) -> "QuantumCircuit":
        """
        转译入口: 分解到基础门集 + 可选布局/SWAP 插入 + 门消减优化。

        参数:
            basis_gates: 目标门集
            coupling_map: 设备耦合边列表 [(q0,q1),...]
            optimization_level: 0=不优化, 1=消恒等/合并旋转, 2=再加对消
            seed: 随机种子 (布局初始化用)
        """
        qc = self.decompose(basis_gates)
        if coupling_map is not None:
            qc = _map_to_coupling(qc, coupling_map, seed=seed)
        if optimization_level >= 1:
            qc = _cancel_identities(qc)
            qc = _merge_rotations(qc)
        if optimization_level >= 2:
            qc = _cancel_self_inverse(qc)
        return qc

    # ================================================================
    # 执行
    # ================================================================
    def _to_state(self, initial_state: Optional[np.ndarray] = None):
        """在模拟器上运行电路, 返回最终 QuantumState。"""
        from ..core.state import QuantumState
        st = QuantumState(self.num_qubits)
        if initial_state is not None:
            iv = np.asarray(initial_state, dtype=complex).reshape(-1)
            if iv.shape[0] != st.size:
                raise ValueError(
                    f"初始态维度 {iv.shape[0]} 应为 2^{self.num_qubits}={st.size}"
                )
            st.amplitudes = iv / np.linalg.norm(iv)
        for inst in self.data:
            if inst.gate.name == "barrier":
                continue
            st.apply_gate(inst.gate.to_matrix(), *inst.qubits)
        if self.global_phase:
            st.amplitudes *= np.exp(1j * self.global_phase)
        return st

    def statevector(self, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """返回电路作用后的态向量 (numpy ndarray)。"""
        return self._to_state(initial_state).amplitudes.copy()

    def unitary_matrix(self) -> np.ndarray:
        """返回整个电路的酉矩阵 (含全局相位)。"""
        n = self.num_qubits
        dim = 1 << n
        # 逐列作用: 对每列单位向量跑一遍电路代价太高, 改为矩阵乘
        U = np.eye(dim, dtype=complex)
        from ..core.state import _embed_gate
        for inst in self.data:
            if inst.gate.name == "barrier":
                continue
            U = _embed_gate(inst.gate.to_matrix(), inst.qubits, n) @ U
        if self.global_phase:
            U = np.exp(1j * self.global_phase) * U
        return U

    def execute(self, shots: int = 1024, seed: Optional[int] = None,
                initial_state: Optional[np.ndarray] = None) -> "ExecutionResult":
        """
        运行电路并采样, 返回 ExecutionResult (含 counts 与态向量)。
        """
        st = self._to_state(initial_state)
        counts = st.sample_counts(shots, seed=seed)
        return ExecutionResult(self, st, counts, shots)

    def execute_density(self, shots: int = 1024, seed: Optional[int] = None,
                        noise=None,
                        initial_state: Optional[np.ndarray] = None
                        ) -> "ExecutionResult":
        """
        密度矩阵模拟路径 — 支持噪声信道 (退极化/振幅阻尼/相位阻尼/读出误差)。

        逐门应用 U rho U†, 每个门后按 NoiseModel 应用噪声信道;
        测量时对角线采样并按 readout_error 翻转比特。

        参数:
            shots: 采样次数
            seed: 随机种子 (可复现)
            noise: NoiseModel 实例; None 等价于理想演化 (仍走密度矩阵路径)
            initial_state: 可选初始态向量 (默认 |0...0>)

        返回:
            ExecutionResult, .state 为 DensityMatrix 实例

        注意:
            密度矩阵维度 4^n, 仅建议 n <= 12; 纯态大电路请用 execute()。
        """
        from ..core.state import DensityMatrix, NoiseModel, _embed_gate
        if noise is None:
            noise = NoiseModel()
        n = self.num_qubits
        if n > 12:
            import warnings
            warnings.warn(
                f"密度矩阵模拟内存为 16^n 字节量级 (n={n}), 建议 n <= 12")
        dm = DensityMatrix(n)
        if initial_state is not None:
            iv = np.asarray(initial_state, dtype=complex).reshape(-1)
            if iv.shape[0] != dm.dim:
                raise ValueError(
                    f"初始态维度 {iv.shape[0]} 应为 2^{n}={dm.dim}")
            iv = iv / np.linalg.norm(iv)
            dm._rho = np.outer(iv, iv.conj())
        for inst in self.data:
            if inst.gate.name == "barrier":
                continue
            full = _embed_gate(inst.gate.to_matrix(), inst.qubits, n)
            if self.global_phase:
                # 全局相位对密度矩阵无影响: e^{iφ} ρ e^{-iφ} = ρ
                pass
            dm.apply_unitary(full)
            if noise.depolarizing:
                dm.apply_depolarizing(noise.depolarizing)
            if noise.amplitude_damping:
                dm.apply_amplitude_damping(noise.amplitude_damping)
            if noise.phase_damping:
                dm.apply_phase_damping(noise.phase_damping)

        # 对角线采样 + 读出误差翻转
        rng = np.random.default_rng(seed)
        probs = np.real(np.diag(dm._rho)).copy()
        probs = np.clip(probs, 0.0, None)
        total = probs.sum()
        probs = probs / total if total > 0 else np.full(dm.dim, 1.0 / dm.dim)
        samples = rng.choice(dm.dim, size=shots, p=probs)
        if noise.readout_error:
            flips = rng.random((shots, n)) < noise.readout_error
            for b in range(n):
                samples = samples ^ (flips[:, b].astype(np.int64) << b)
        counts_arr = np.bincount(samples, minlength=dm.dim)
        counts = {i: int(counts_arr[i]) for i in range(dm.dim)
                  if counts_arr[i] > 0}
        return ExecutionResult(self, dm, counts, shots)

    def expectation(self, observable, initial_state: Optional[np.ndarray] = None) -> float:
        """
        免采样直接计算解析期望值 <ψ|O|ψ>。

        参数:
            observable: Pauli 字符串 (如 "XXZ") / Pauli 串列表及系数
                        [(coeff, "XYZ"), ...] / 稀疏矩阵 / 稠密矩阵
        """
        st = self._to_state(initial_state)
        return st.expectation(observable)

    def draw(self, output: str = "text", **kwargs):
        """
        绘制电路。

        output='text' 返回 ASCII 字符串; output='mpl' 调用
        qsl.viz.draw_circuit_mpl 返回 (matplotlib Figure, Axes)
        (需 pip install matplotlib, 支持 ax/style/fold 等关键字参数)。
        """
        if output == "text":
            from .text_drawer import draw_text
            return draw_text(self, **kwargs)
        if output in ("mpl", "matplotlib"):
            from ..viz.circuit_drawer import draw_circuit_mpl
            return draw_circuit_mpl(self, **kwargs)
        raise ValueError(f"不支持的输出格式: {output!r} (可选 'text'/'mpl')")

    # ================================================================
    # 序列化
    # ================================================================
    def to_json(self) -> str:
        """序列化为 JSON 字符串 (含符号参数)。"""
        def enc(v):
            if isinstance(v, Parameter):
                return {"__param__": v.name}
            if isinstance(v, ParameterExpression):
                return {"__expr__": str(v)}
            return v

        insts = []
        for inst in self.data:
            g = inst.gate
            entry = {
                "name": g.name,
                "qubits": list(inst.qubits),
                "params": [enc(p) for p in g.params],
            }
            if g._matrix is not None and g.name in ("unitary",) or g.name.startswith("c") and g._matrix is not None and g.name not in _KNOWN_MATRIX_GATES:
                entry["matrix_re"] = np.real(g._matrix).tolist()
                entry["matrix_im"] = np.imag(g._matrix).tolist()
            insts.append(entry)
        return json.dumps({
            "format": "qsl-circuit",
            "version": 1,
            "name": self.name,
            "num_qubits": self.num_qubits,
            "global_phase": self.global_phase,
            "instructions": insts,
        })

    @staticmethod
    def from_json(s: str) -> "QuantumCircuit":
        """从 JSON 字符串恢复电路。"""
        obj = json.loads(s)
        if obj.get("format") != "qsl-circuit":
            raise ValueError("不是 qsl-circuit 格式的 JSON")
        qc = QuantumCircuit(obj["num_qubits"], obj.get("name", ""),
                            obj.get("global_phase", 0.0))
        for entry in obj["instructions"]:
            name = entry["name"]
            qubits = tuple(entry["qubits"])
            params = []
            for p in entry.get("params", []):
                if isinstance(p, dict) and "__param__" in p:
                    params.append(Parameter(p["__param__"]))
                else:
                    params.append(p)
            if "matrix_re" in entry:
                mat = (np.array(entry["matrix_re"])
                       + 1j * np.array(entry["matrix_im"]))
                qc.append(Gate(name, mat, len(qubits)), qubits)
            else:
                qc.append(_make_named_gate(name, params), qubits)
        return qc

    # ================================================================
    # 显示
    # ================================================================
    def __repr__(self) -> str:
        return (f"QuantumCircuit(num_qubits={self.num_qubits}, "
                f"gates={self.size()}, depth={self.depth()})")

    def __str__(self) -> str:
        return self.draw("text")


# ====================================================================
# 门工厂 (from_json 使用)
# ====================================================================

_KNOWN_MATRIX_GATES = {
    "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg",
    "cx", "cy", "cz", "ch", "cs", "csdg", "ct", "ctdg", "swap",
    "iswap", "dcx", "ecr", "ccx", "cswap",
}


def _make_named_gate(name: str, params: list) -> Gate:
    """按名称重建参数化门。"""
    fn_map = {
        "rx": (G.rx, 1, "RX"), "ry": (G.ry, 1, "RY"), "rz": (G.rz, 1, "RZ"),
        "p": (G.p, 1, "P"), "u": (G.u3, 1, "U"),
        "crx": (G.crx, 2, "RX"), "cry": (G.cry, 2, "RY"),
        "crz": (G.crz, 2, "RZ"), "cp": (G.cp, 2, "P"),
        "cu": (G.cu, 2, "U"),
        "rxx": (G.rxx, 2, "RXX"), "ryy": (G.ryy, 2, "RYY"),
        "rzz": (G.rzz, 2, "RZZ"),
    }
    mat_map = {
        "id": (G.I, 1, "I"), "x": (G.X, 1, "X"), "y": (G.Y, 1, "Y"),
        "z": (G.Z, 1, "Z"), "h": (G.H, 1, "H"), "s": (G.S, 1, "S"),
        "sdg": (G.Sdg, 1, "S†"), "t": (G.T, 1, "T"), "tdg": (G.Tdg, 1, "T†"),
        "sx": (G.SX, 1, "√X"), "sxdg": (G.SXdg, 1, "√X†"),
        # cx 矩阵基序 |c,t> 即控制位高位
        "cx": (_CX_MATRIX, 2, "X"),
        "cy": (G.controlled_gate(G.Y, 1), 2, "Y"),
        "cz": (G.controlled_gate(G.Z, 1), 2, "Z"),
        "swap": (G.swap(), 2, "×"), "iswap": (G.iswap(), 2, "i×"),
        "dcx": (G.dcx(), 2, "DCX"), "ecr": (G.ecr(), 2, "ECR"),
        "ch": (G.ch(), 2, "H"), "cs": (G.cs(), 2, "S"),
        "csdg": (G.csdg(), 2, "S†"), "ct": (G.ct(), 2, "T"),
        "ctdg": (G.ctdg(), 2, "T†"),
        "ccx": (G.mcx(2), 3, "X"), "cswap": (G.cswap(), 3, "×"),
    }
    if name in fn_map:
        fn, nq, label = fn_map[name]
        return Gate(name, None, nq, params=list(params), matrix_fn=fn, label=label)
    if name in mat_map:
        mat, nq, label = mat_map[name]
        return Gate(name, mat, nq, label=label)
    if name.startswith("mcx"):
        n = int(name[3:])
        return Gate(name, G.mcx(n), n + 1, label="X")
    if name.startswith("mcz"):
        n = int(name[3:])
        dim = 1 << n
        mat = np.eye(dim, dtype=complex)
        mat[-1, -1] = -1
        return Gate(name, mat, n, label="Z")
    raise ValueError(f"未知的门名: {name!r}")


# ====================================================================
# 执行结果
# ====================================================================

class ExecutionResult:
    """电路执行结果: 计数 + 最终态。"""

    def __init__(self, circuit: QuantumCircuit, state, counts: Dict[int, int],
                 shots: int):
        self.circuit = circuit
        self.state = state
        self.counts = counts
        self.shots = shots

    def get_counts(self, binary: bool = True) -> Dict[str, int]:
        """返回 {比特串: 次数} (binary=True) 或 {整数: 次数}。"""
        if not binary:
            return dict(self.counts)
        n = self.circuit.num_qubits
        return {format(k, f"0{n}b"): v for k, v in self.counts.items()}

    def most_frequent(self, n: int = 1) -> List[Tuple[str, int]]:
        """最高频的 n 个结果 (比特串, 次数)。"""
        items = sorted(self.counts.items(), key=lambda kv: -kv[1])[:n]
        width = self.circuit.num_qubits
        return [(format(k, f"0{width}b"), v) for k, v in items]

    def probabilities(self) -> Dict[str, float]:
        """经验概率分布 {比特串: 频率}。"""
        width = self.circuit.num_qubits
        return {format(k, f"0{width}b"): v / self.shots
                for k, v in self.counts.items()}

    def statevector(self) -> np.ndarray:
        return self.state.amplitudes.copy()

    def __repr__(self) -> str:
        return f"ExecutionResult(shots={self.shots}, outcomes={len(self.counts)})"


# ====================================================================
# decompose / transpile 辅助
# ====================================================================

def _expand_instruction(inst: Instruction, num_qubits: int
                        ) -> Optional[Tuple[List[Instruction], float]]:
    """
    把一条指令展开到基础门; 无法展开返回 None。

    返回 (指令列表, 全局相位增量)。单比特 ZYZ 分解丢弃的 e^{iγ}
    通过相位增量累积回 circuit.global_phase, 保证与 Qiskit 数值逐位一致。
    """
    g = inst.gate
    qs = inst.qubits
    name = g.name
    out: List[Instruction] = []

    def add(gate, qubits):
        out.append(Instruction(gate, qubits))

    def rz_gate(a, q):
        return Instruction(Gate("rz", None, 1, [a], G.rz, "RZ"), (q,))

    # ---- ccx 标准分解 (6 CNOT) ----
    if name == "ccx":
        a, b, c = qs
        h = Gate("h", G.H, 1, label="H")
        cx = _cx_gate()
        t = Gate("t", G.T, 1, label="T")
        tdg = Gate("tdg", G.Tdg, 1, label="T†")
        for gate, qubits in [
            (h, (c,)), (cx, (b, c)), (tdg, (c,)), (cx, (a, c)),
            (t, (c,)), (cx, (b, c)), (tdg, (c,)), (cx, (a, c)),
            (t, (b,)), (t, (c,)), (h, (c,)), (cx, (a, b)),
            (t, (a,)), (tdg, (b,)), (cx, (a, b)),
        ]:
            add(gate, qubits)
        return out, 0.0

    # ---- cswap -> ccx + cx (Nielsen-Chuang Fig.4.9: CX(b,a)·CCX(c,a,b)·CX(b,a)) ----
    if name == "cswap":
        c, a, b = qs
        add(_cx_gate(), (b, a))
        add(Gate("ccx", G.mcx(2), 3, label="X"), (c, a, b))
        add(_cx_gate(), (b, a))
        return out, 0.0

    # ---- rzz(θ) = CX · Rz(θ) · CX  (CNOT 共轭 I⊗Z -> Z⊗Z) ----
    if name == "rzz":
        theta = resolve(g.params[0])
        q0, q1 = qs
        add(_cx_gate(), (q0, q1))
        add(Gate("rz", None, 1, [theta], G.rz, "RZ"), (q1,))
        add(_cx_gate(), (q0, q1))
        return out, 0.0

    # ---- rxx(θ) = (H⊗H) RZZ(θ) (H⊗H) ----
    if name == "rxx":
        theta = resolve(g.params[0])
        q0, q1 = qs
        h = Gate("h", G.H, 1, label="H")
        for q in qs:
            add(h, (q,))
        add(Gate("rzz", None, 2, [theta], G.rzz, "RZZ"), (q0, q1))
        for q in qs:
            add(h, (q,))
        return out, 0.0

    # ---- ryy(θ) = (S⊗S) RXX(θ) (S†⊗S†)  (Y = S X S†) ----
    if name == "ryy":
        theta = resolve(g.params[0])
        q0, q1 = qs
        s = Gate("s", G.S, 1, label="S")
        sdg = Gate("sdg", G.Sdg, 1, label="S†")
        for q in qs:
            add(sdg, (q,))
        add(Gate("rxx", None, 2, [theta], G.rxx, "RXX"), (q0, q1))
        for q in qs:
            add(s, (q,))
        return out, 0.0

    # ---- iswap = (S⊗S) · SWAP · CZ ----
    if name == "iswap":
        q0, q1 = qs
        s = Gate("s", G.S, 1, label="S")
        add(Gate("cz", G.controlled_gate(G.Z, 1), 2, label="Z"), (q0, q1))
        add(Gate("swap", G.swap(), 2, label="×"), (q0, q1))
        add(s, (q0,))
        add(s, (q1,))
        return out, 0.0

    # ---- dcx = CX(q1,q0) · CX(q0,q1) (时间序: 先 cx(q1,q0)) ----
    if name == "dcx":
        q0, q1 = qs
        add(_cx_gate(), (q1, q0))
        add(_cx_gate(), (q0, q1))
        return out, 0.0

    # ---- ecr = (A⊗B) · CX · (H⊗H), A/B 为固定单比特酉 (再经 ZYZ 展开) ----
    if name == "ecr":
        q0, q1 = qs
        a_mat = np.array([[-0.5 - 0.5j, 0.5 - 0.5j],
                          [-0.5 - 0.5j, -0.5 + 0.5j]])
        b_mat = np.array([[-1.0, 1.0], [1.0j, 1.0j]]) / math.sqrt(2)
        h = Gate("h", G.H, 1, label="H")
        add(h, (q0,))
        add(h, (q1,))
        add(_cx_gate(), (q0, q1))
        add(Gate("unitary", a_mat, 1, label="A"), (q0,))
        add(Gate("unitary", b_mat, 1, label="B"), (q1,))
        return out, 0.0

    # ---- 受控单比特门 -> 2-CNOT 分解 ----
    if g.num_qubits == 2 and name.startswith("c") and name not in (
            "cx", "cz", "swap", "iswap", "dcx", "ecr", "rxx", "ryy", "rzz"):
        return _decompose_controlled_1q(g, qs), 0.0

    # ---- 任意单比特酉 -> ZYZ (全局相位 e^{iγ} 通过返回值累积) ----
    if g.num_qubits == 1 and name not in (
            "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg",
            "rx", "ry", "rz", "p"):
        insts, gamma = _decompose_1q(g, qs)
        return insts, gamma

    return None


def _cx_matrix() -> np.ndarray:
    """CX 矩阵 (基序 |c,t>, 控制位高位)。"""
    mat = np.eye(4, dtype=complex)
    mat[2, 2] = 0; mat[3, 3] = 0
    mat[2, 3] = 1; mat[3, 2] = 1
    return mat


_CX_MATRIX = _cx_matrix()


def _cx_gate() -> Gate:
    return Gate("cx", _CX_MATRIX, 2, label="X")


def _decompose_1q(g: Gate, qs) -> Tuple[List[Instruction], float]:
    """
    任意单比特酉的 ZYZ 分解。

    返回 (指令列表, 全局相位 γ)。ZYZ 只能恢复 SU(2) 部分,
    剩余全局相位 e^{iγ} 必须由调用方累积, 否则受控版本会出错。
    """
    theta, phi, lam, gamma = zyz_decompose(g.to_matrix())
    out = []
    q = qs[0]
    out.append(Instruction(Gate("rz", None, 1, [lam], G.rz, "RZ"), (q,)))
    out.append(Instruction(Gate("ry", None, 1, [theta], G.ry, "RY"), (q,)))
    out.append(Instruction(Gate("rz", None, 1, [phi], G.rz, "RZ"), (q,)))
    return out, gamma


def _decompose_controlled_1q(g: Gate, qs) -> List[Instruction]:
    """
    受控单比特门 C-U 的标准分解 (Barenco et al. 1995, Lemma 5.2)。

    门矩阵为 4x4 分块对角 [[I,0],[0,U]] (控制位高位), 需先取右下 2x2 的 U。
    对 U = e^{iγ} Rz(φ) Ry(θ) Rz(λ) (ZYZ 分解), 有 U = e^{iγ} A·X·B·X·C:
        A = Rz(φ) Ry(θ/2)
        B = Ry(-θ/2) Rz(-(λ+φ)/2)
        C = Rz((λ-φ)/2)
    且 A·B·C = I。于是:
        C-U = P(γ)⊗A·CX·B·CX·C  (执行顺序: C, CX, B, CX, A, P(γ))
    """
    full = g.to_matrix()
    if full.shape != (4, 4):
        raise ValueError(f"_decompose_controlled_1q 期望 4x4 受控门矩阵, 得到 {full.shape}")
    # 校验受控结构: 左上块应为 I, 非对角块应为 0
    if not (np.allclose(full[:2, :2], np.eye(2), atol=1e-10)
            and np.allclose(full[:2, 2:], 0, atol=1e-10)
            and np.allclose(full[2:, :2], 0, atol=1e-10)):
        raise ValueError(f"门 '{g.name}' 不是标准的单控制单比特门 (分块对角 [[I,0],[0,U]])")
    u_mat = full[2:, 2:]
    theta, phi, lam, gamma = zyz_decompose(u_mat)
    c, t = qs

    def rz_gate(a, q):
        return Instruction(Gate("rz", None, 1, [a], G.rz, "RZ"), (q,))

    def ry_gate(a, q):
        return Instruction(Gate("ry", None, 1, [a], G.ry, "RY"), (q,))

    out: List[Instruction] = []
    # C = Rz((λ-φ)/2)
    out.append(rz_gate((lam - phi) / 2, t))
    out.append(Instruction(_cx_gate(), (c, t)))
    # B = Ry(-θ/2) Rz(-(λ+φ)/2)  (先 Rz 后 Ry)
    out.append(rz_gate(-(lam + phi) / 2, t))
    out.append(ry_gate(-theta / 2, t))
    out.append(Instruction(_cx_gate(), (c, t)))
    # A = Rz(φ) Ry(θ/2)  (先 Ry 后 Rz)
    out.append(ry_gate(theta / 2, t))
    out.append(rz_gate(phi, t))
    # 全局相位 -> 控制位上的相位门
    if abs(gamma) > 1e-12:
        out.append(Instruction(Gate("p", None, 1, [gamma], G.p, "P"), (c,)))
    return out


def _cancel_identities(qc: QuantumCircuit) -> QuantumCircuit:
    """消去近恒等门 (如 rx(0)) 与相邻逆门对。"""
    out = qc.copy()
    new_data = []
    for inst in out.data:
        g = inst.gate
        if g.num_qubits == 1 and g.params:
            try:
                mat = g.to_matrix()
                if np.allclose(mat, np.eye(2), atol=1e-12):
                    continue
            except Exception:
                pass
        new_data.append(inst)
    out.data = new_data
    return out


def _merge_rotations(qc: QuantumCircuit) -> QuantumCircuit:
    """合并同一比特上相邻的同种旋转门。"""
    out = qc.copy()
    merged: List[Instruction] = []
    for inst in out.data:
        if (merged and inst.gate.num_qubits == 1
                and inst.gate.name in ("rx", "ry", "rz", "p")
                and merged[-1].gate.name == inst.gate.name
                and merged[-1].qubits == inst.qubits
                and not inst.gate.is_parameterized
                and not merged[-1].gate.is_parameterized):
            prev = merged[-1].gate
            total = prev.params[0] + inst.gate.params[0]
            merged[-1] = Instruction(
                Gate(inst.gate.name, None, 1, [total],
                     getattr(G, inst.gate.name), prev.label),
                inst.qubits)
        else:
            merged.append(inst)
    out.data = merged
    return out


def _cancel_self_inverse(qc: QuantumCircuit) -> QuantumCircuit:
    """消去相邻的自逆门对 (h-h, x-x, cx-cx 等)。"""
    self_inverse = {"h", "x", "y", "z", "id", "cx", "cy", "cz", "swap", "ccx"}
    inverse_pairs = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t",
                     "sx": "sxdg", "sxdg": "sx"}
    out = qc.copy()
    stack: List[Instruction] = []
    for inst in out.data:
        if stack:
            prev = stack[-1]
            same_qubits = prev.qubits == inst.qubits
            if same_qubits and prev.gate.name == inst.gate.name \
                    and inst.gate.name in self_inverse:
                stack.pop()
                continue
            if same_qubits and inverse_pairs.get(prev.gate.name) == inst.gate.name:
                stack.pop()
                continue
        stack.append(inst)
    out.data = stack
    return out


def _map_to_coupling(qc: QuantumCircuit,
                     coupling_map: List[Tuple[int, int]],
                     seed: Optional[int] = None) -> QuantumCircuit:
    """
    把电路映射到受限耦合图: 对不邻接的双比特门插入 SWAP 路径。
    采用简单的 BFS 最短路径策略。
    """
    from collections import deque

    adj: Dict[int, List[int]] = {}
    for a, b in coupling_map:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    def shortest_path(src, dst):
        if src == dst:
            return [src]
        seen = {src}
        dq = deque([(src, [src])])
        while dq:
            node, path = dq.popleft()
            for nb in adj.get(node, []):
                if nb in seen:
                    continue
                if nb == dst:
                    return path + [nb]
                seen.add(nb)
                dq.append((nb, path + [nb]))
        return None

    rng = np.random.default_rng(seed)
    out = qc.copy()
    new_data: List[Instruction] = []
    for inst in out.data:
        if inst.gate.num_qubits != 2 or inst.gate.name == "swap":
            new_data.append(inst)
            continue
        a, b = inst.qubits
        if b in adj.get(a, []):
            new_data.append(inst)
            continue
        path = shortest_path(a, b)
        if path is None:
            raise ValueError(f"耦合图中 {a} 与 {b} 不连通")
        # 把 a 的内容经 SWAP 链移到 b 旁边
        for i in range(len(path) - 2):
            new_data.append(Instruction(Gate("swap", G.swap(), 2, label="×"),
                                        (path[i], path[i + 1])))
        new_data.append(Instruction(inst.gate, (path[-2], path[-1])))
        for i in reversed(range(len(path) - 2)):
            new_data.append(Instruction(Gate("swap", G.swap(), 2, label="×"),
                                        (path[i], path[i + 1])))
    out.data = new_data
    return out


__all__ = ["QuantumCircuit", "Instruction", "ExecutionResult"]
