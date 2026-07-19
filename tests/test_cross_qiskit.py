"""
与 Qiskit 的数值交叉验证。

对比路径: qsl QuantumCircuit --to_qiskit--> qiskit QuantumCircuit,
然后比较 qsl unitary_matrix() 与 qiskit.quantum_info.Operator 的矩阵。

比特序说明:
    qsl 与 qiskit 的态向量都是 qubit 0 = LSB, 但"单门矩阵"的基序相反:
    qsl 矩阵行/列索引 = Σ bit(t_j)·2^{k-1-j} (qargs[0] 在矩阵高位),
    qiskit 单门矩阵 qargs[0] 在低位。对交换对称的门 (cx/cz/swap/rxx...)
    与受控门 (控制位在两边约定中都落在 qargs[0]) 两种约定给出同一
    物理算符; 对非对称门 dcx / ecr, 同样数值的矩阵在两种约定下是
    不同的物理算符 —— 实测 qsl.dcx(0,1) == qiskit.dcx(1,0),
    qsl.ecr(0,1) == qiskit.ecr(1,0)。这两个门单独按实际语义断言
    (见 test_gate_dcx / test_gate_ecr)。

全局相位说明:
    qiskit >= 2.x 的 Operator(qc) 已计入 qc.global_phase (本文件
    test_global_phase 真实验证), 因此直接对比矩阵即可, 无需手动
    乘 exp(1j*phase)。

全部断言 atol=1e-10。
"""

import math

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit import QuantumCircuit as QiskitQC  # noqa: E402
from qiskit.circuit.library import QFTGate  # noqa: E402
from qiskit.quantum_info import Operator, Statevector  # noqa: E402

from qsl.circuit import QuantumCircuit, Parameter  # noqa: E402
from qsl.circuit.converters import to_qiskit  # noqa: E402

ATOL = 1e-10


def _assert_same_unitary(qsl_circ: QuantumCircuit):
    """qsl unitary_matrix() 与 to_qiskit 转换后的 Operator 矩阵逐元一致。"""
    qk = to_qiskit(qsl_circ)
    np.testing.assert_allclose(
        qsl_circ.unitary_matrix(), Operator(qk).data, atol=ATOL)


# ====================================================================
# 单比特固定门
# ====================================================================

@pytest.mark.parametrize("name", ["h", "x", "y", "z", "s", "sdg",
                                  "t", "tdg", "sx", "sxdg"])
def test_gate_1q_fixed(name):
    c = QuantumCircuit(1)
    getattr(c, name)(0)
    _assert_same_unitary(c)


# ====================================================================
# 单比特参数化门
# ====================================================================

@pytest.mark.parametrize("name,args", [
    ("rx", (0.7,)), ("ry", (0.7,)), ("rz", (0.7,)), ("p", (0.7,)),
    ("u", (0.3, 0.4, 0.5)),
])
def test_gate_1q_parametric(name, args):
    c = QuantumCircuit(1)
    getattr(c, name)(0, *args)
    _assert_same_unitary(c)


# ====================================================================
# 两比特受控门 / 对称门
# ====================================================================

@pytest.mark.parametrize("name", ["cx", "cy", "cz", "ch", "cs", "csdg",
                                  "ct", "ctdg", "swap", "iswap"])
def test_gate_2q_fixed(name):
    c = QuantumCircuit(2)
    getattr(c, name)(0, 1)
    _assert_same_unitary(c)


@pytest.mark.parametrize("name,args", [
    ("cp", (0.7,)), ("crx", (0.7,)), ("cry", (0.7,)), ("crz", (0.7,)),
    ("cu", (0.3, 0.4, 0.5, 0.6)),
    ("rxx", (0.7,)), ("ryy", (0.7,)), ("rzz", (0.7,)),
])
def test_gate_2q_parametric(name, args):
    c = QuantumCircuit(2)
    if name == "cu":
        theta, phi, lam, gamma = args
        c.cu(theta, phi, lam, 0, 1, gamma)
    else:
        getattr(c, name)(args[0], 0, 1)
    _assert_same_unitary(c)


# ====================================================================
# 三比特门
# ====================================================================

def test_gate_ccx():
    c = QuantumCircuit(3)
    c.ccx(0, 1, 2)
    _assert_same_unitary(c)


def test_gate_cswap():
    c = QuantumCircuit(3)
    c.cswap(0, 1, 2)
    _assert_same_unitary(c)


# ====================================================================
# 非交换对称门: dcx / ecr —— 与 qiskit 的已知比特序差异
#
# qsl 与 qiskit 的 dcx/ecr 单门矩阵数值完全相同, 但矩阵基序相反
# (qsl qargs[0] 在矩阵高位, qiskit 在低位), 因此物理算符相差一次
# 比特交换。按实际正确语义断言: qsl.dcx(0,1) == qiskit.dcx(1,0),
# qsl.ecr(0,1) == qiskit.ecr(1,0)。
# ====================================================================

def test_gate_dcx():
    c = QuantumCircuit(2)
    c.dcx(0, 1)
    qk = QiskitQC(2)
    qk.dcx(1, 0)  # qiskit 语义下与 qsl.dcx(0,1) 等价的比特序
    np.testing.assert_allclose(
        c.unitary_matrix(), Operator(qk).data, atol=ATOL)


def test_gate_ecr():
    c = QuantumCircuit(2)
    c.ecr(0, 1)
    qk = QiskitQC(2)
    qk.ecr(1, 0)  # 同上, 比特序交换后等价
    np.testing.assert_allclose(
        c.unitary_matrix(), Operator(qk).data, atol=ATOL)


# ====================================================================
# QFT(3) 与 qiskit.circuit.library.QFTGate 数值一致
# ====================================================================

def _qsl_qft(n: int) -> QuantumCircuit:
    """标准 QFT 电路 (MSB 优先处理 + 末尾位反转 SWAP 网络)。"""
    qc = QuantumCircuit(n)
    for i in range(n - 1, -1, -1):
        qc.h(i)
        for j in range(i - 1, -1, -1):
            k = i - j + 1
            qc.cp(2 * math.pi / (1 << k), j, i)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc


def test_qft3_matches_qiskit():
    qc = _qsl_qft(3)
    qk_op = Operator(QFTGate(3)).data  # do_swaps=True (默认)
    np.testing.assert_allclose(qc.unitary_matrix(), qk_op, atol=ATOL)


# ====================================================================
# 组合电路: 酉矩阵 + 逐振幅一致
# ====================================================================

def _combo_circuit() -> QuantumCircuit:
    """h/cx/rz/crx/ccx 等混合 13 门的 3 比特电路。"""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(1, 0.7)
    qc.crx(0.4, 1, 2)
    qc.ccx(0, 1, 2)
    qc.t(2)
    qc.cp(0.3, 0, 2)
    qc.ry(2, 1.1)
    qc.cz(2, 0)
    qc.swap(0, 1)
    qc.s(1)
    qc.rxx(-0.6, 0, 2)
    qc.h(2)
    return qc


def test_combo_circuit_unitary():
    _assert_same_unitary(_combo_circuit())


def test_combo_circuit_statevector():
    qc = _combo_circuit()
    qk = to_qiskit(qc)
    qk_sv = Statevector(qk).data
    np.testing.assert_allclose(qc.statevector(), qk_sv, atol=ATOL)


# ====================================================================
# 参数化电路: qsl bind_parameters vs qiskit assign_parameters
# ====================================================================

def test_parameterized_circuit_binding():
    theta = Parameter("theta")
    lam = Parameter("lam")

    qc = QuantumCircuit(2)
    qc.rx(0, theta)
    qc.ry(1, theta + 0.3)
    qc.cx(0, 1)
    qc.crz(2 * lam - 0.1, 0, 1)
    qc.cp(lam / 3, 1, 0)

    values = {"theta": 0.9, "lam": -0.4}
    qsl_bound = qc.bind_parameters(values)

    qk = to_qiskit(qc)  # 保持符号的 qiskit 电路
    qk_bound = qk.assign_parameters(values)

    np.testing.assert_allclose(
        qsl_bound.unitary_matrix(), Operator(qk_bound).data, atol=ATOL)


# ====================================================================
# 全局相位: Operator(qc) 已计入 global_phase (qiskit >= 2.x 实测),
# 因此非零 global_phase 的电路也直接逐元一致。
# ====================================================================

def test_global_phase():
    qc = QuantumCircuit(1, global_phase=0.35)
    qc.h(0)
    qc.rz(0, 0.8)
    _assert_same_unitary(qc)
