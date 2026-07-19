"""
标准电路库 — 常用量子电路的工厂函数。

包含: QFT、QPE、GHZ、W 态、量子隐形传态、Grover 扩散算子、
      随机电路生成器、Bell 态、叠加态制备。
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .circuit import QuantumCircuit
from .gate import Gate


def bell_state(which: str = "phi+") -> QuantumCircuit:
    """
    Bell 态制备电路 (2 比特)。

    参数:
        which: "phi+", "phi-", "psi+", "psi-"
    """
    qc = QuantumCircuit(2, name=f"Bell-{which}")
    qc.h(0)
    qc.cx(0, 1)
    if which == "phi-":
        qc.z(0)
    elif which == "psi+":
        qc.x(1)
    elif which == "psi-":
        qc.x(0)
        qc.x(1)
    elif which != "phi+":
        raise ValueError(f"未知 Bell 态: {which!r}")
    return qc


def ghz_state(n: int) -> QuantumCircuit:
    """
    GHZ 态 (|00...0> + |11...1>)/√2 制备电路。

    参数:
        n: 量子比特数
    """
    qc = QuantumCircuit(n, name=f"GHZ-{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def w_state(n: int) -> QuantumCircuit:
    """
    W 态 (|10..0> + |01..0> + ... + |0..01>)/√n 制备电路。

    构造: 先在 q0 放一个激发, 再用受控 G(θ) 门逐级把振幅均分到后续比特,
    每步后用 CNOT 把激发位搬回, 最终得到等幅叠加。
    """
    qc = QuantumCircuit(n, name=f"W-{n}")
    qc.x(0)
    for k in range(1, n):
        # 当前激发在 q_{k-1}; 以概率 1/(n-k+1) 把它分裂到 q_k
        theta = 2 * math.acos(math.sqrt(1.0 / (n - k + 1)))
        g = _g_gate(theta)
        qc.append(g.control(1), (k - 1, k))
        qc.cx(k, k - 1)
    return qc


def _g_gate(theta: float) -> Gate:
    """W 态制备的辅助旋转门 G(θ) = exp(-iθY/2) 的 |0>→cos|0>+sin|1> 形式。"""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return Gate("g", np.array([[c, -s], [s, c]], dtype=complex), 1, label="G")


def qft(n: int, inverse: bool = False, do_swaps: bool = True,
        approximation_degree: int = 0) -> QuantumCircuit:
    """
    量子傅里叶变换电路。

    参数:
        n: 比特数
        inverse: 是否逆 QFT (IQFT)
        do_swaps: 末尾是否带 SWAP 反转比特序
        approximation_degree: 忽略角度小于 2π/2^(n-k) 的受控相位 (0=精确)
    """
    qc = QuantumCircuit(n, name="IQFT" if inverse else "QFT")
    for j in range(n):
        qc.h(j)
        num_cp = min(n - j - 1, (n - 1) - approximation_degree)
        for k in range(1, num_cp + 1):
            lam = math.pi / (2 ** k)
            qc.cp(lam, j + k, j)
    if do_swaps:
        for i in range(n // 2):
            qc.swap(i, n - 1 - i)
    return qc.inverse() if inverse else qc


def qpe(unitary: Gate, n_counting: int, eigenstate_prep=None) -> QuantumCircuit:
    """
    量子相位估计 (QPE) 电路。

    参数:
        unitary: 待估相位的单比特门 U (其本征值 e^{2πiφ})
        n_counting: 计数寄存器比特数 (决定精度)
        eigenstate_prep: 可选, 作用在本征态寄存器上的制备电路
    返回:
        总计 n_counting + unitary.num_qubits 比特的电路
    """
    n_target = unitary.num_qubits
    qc = QuantumCircuit(n_counting + n_target, name="QPE")
    counting = list(range(n_counting))
    target = list(range(n_counting, n_counting + n_target))

    if eigenstate_prep is not None:
        qc.compose(eigenstate_prep, qubits=target, inplace=True)

    # 计数寄存器置均匀叠加
    for q in counting:
        qc.h(q)
    # 受控 U^{2^j}
    for j, cq in enumerate(counting):
        powered = unitary.power(2 ** j)
        qc.controlled(powered, (cq,), tuple(target))
    # 逆 QFT
    qc.compose(qft(n_counting, inverse=True, do_swaps=True),
               qubits=counting, inplace=True)
    return qc


def grover_diffusion(n: int) -> QuantumCircuit:
    """
    Grover 扩散算子 D = H^⊗n X^⊗n MCZ X^⊗n H^⊗n (2|s><s| - I)。
    """
    qc = QuantumCircuit(n, name="Diffusion")
    for q in range(n):
        qc.h(q)
        qc.x(q)
    if n == 1:
        qc.z(0)
    elif n == 2:
        qc.cz(0, 1)
    else:
        qc.mcz(list(range(n)))
    for q in range(n):
        qc.x(q)
        qc.h(q)
    return qc


def teleportation() -> QuantumCircuit:
    """
    量子隐形传态电路 (3 比特):
        q0 = 待传送态, q1/q2 = 共享 Bell 对。
        结束后 q2 持有原 q0 的态 (经典修正已内置)。
    """
    qc = QuantumCircuit(3, name="Teleportation")
    # Bell 对
    qc.h(1)
    qc.cx(1, 2)
    # Alice
    qc.cx(0, 1)
    qc.h(0)
    # 经典修正 (以受控门形式)
    qc.cx(1, 2)
    qc.cz(0, 2)
    return qc


def random_circuit(n: int, depth: int, seed: Optional[int] = None,
                   gate_set: Sequence[str] = ("h", "x", "y", "z", "s", "t",
                                              "rx", "ry", "rz", "cx", "cz")) -> QuantumCircuit:
    """
    随机电路生成器 (用于基准测试/量子优越性演示)。

    参数:
        n: 比特数
        depth: 层数
        seed: 随机种子
        gate_set: 候选门集合
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n, name=f"random-{n}x{depth}")
    one_q = [g for g in gate_set if g not in ("cx", "cz", "swap")]
    two_q = [g for g in gate_set if g in ("cx", "cz", "swap")]
    for _ in range(depth):
        for q in range(n):
            g = one_q[rng.integers(len(one_q))]
            if g == "h":
                qc.h(q)
            elif g == "x":
                qc.x(q)
            elif g == "y":
                qc.y(q)
            elif g == "z":
                qc.z(q)
            elif g == "s":
                qc.s(q)
            elif g == "t":
                qc.t(q)
            elif g in ("rx", "ry", "rz"):
                getattr(qc, g)(q, float(rng.uniform(0, 2 * math.pi)))
        if two_q and n > 1:
            q0, q1 = rng.choice(n, size=2, replace=False)
            g = two_q[rng.integers(len(two_q))]
            getattr(qc, g)(int(q0), int(q1))
    return qc


def uniform_superposition(n: int) -> QuantumCircuit:
    """均匀叠加态 H^⊗n 制备电路。"""
    qc = QuantumCircuit(n, name=f"|s>{n}")
    for q in range(n):
        qc.h(q)
    return qc


def quantum_walk_cycle(n_positions: int) -> QuantumCircuit:
    """
    环上离散时间量子行走的一步 (额外功能)。
    位置用 ceil(log2(n_positions)) 比特编码, 1 个硬币比特。
    """
    n_pos = max(1, math.ceil(math.log2(n_positions)))
    qc = QuantumCircuit(n_pos + 1, name=f"QWalk-{n_positions}")
    coin = n_pos
    # 硬币: Hadamard
    qc.h(coin)
    # 受控移位 (简化: 硬币控制位置 +/- 1, 用增量电路)
    # 增量器 (模 2^n_pos)
    for i in range(n_pos):
        qc.controlled(Gate("x", np.array([[0, 1], [1, 0]], dtype=complex), 1, label="X"),
                      tuple([coin] + list(range(i))), (i,))
    return qc


__all__ = [
    "bell_state", "ghz_state", "w_state", "qft", "qpe", "grover_diffusion",
    "teleportation", "random_circuit", "uniform_superposition",
    "quantum_walk_cycle",
]
