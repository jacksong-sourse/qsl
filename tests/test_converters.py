"""
Qiskit/Cirq 转换器测试。

- 已安装 qiskit 时: 真实 round-trip 测试 (qsl -> qiskit -> qsl),
  比较 unitary_matrix + global_phase, atol=1e-10。
- 已安装 cirq 时: to_cirq 数值一致性测试。
- 无论是否安装: monkeypatch sys.modules 模拟未安装, 验证抛出 ImportError。
"""

import math
import sys

import numpy as np
import pytest

from qsl.circuit import QuantumCircuit, Parameter
from qsl.circuit.converters import to_qiskit, from_qiskit, to_cirq

ATOL = 1e-10


def _roundtrip(circ):
    """qsl -> qiskit -> qsl, 断言酉矩阵与全局相位一致, 返回中间 qiskit 电路。"""
    qq = to_qiskit(circ)
    back = from_qiskit(qq)
    np.testing.assert_allclose(
        back.unitary_matrix(), circ.unitary_matrix(), atol=ATOL)
    # qiskit 会把 global_phase 归一化到 [0, 2π), 按模 2π 比较
    d = (back.global_phase - circ.global_phase + math.pi) % (2 * math.pi) - math.pi
    assert abs(d) <= ATOL
    return qq, back


# ====================================================================
# qiskit round-trip: 单比特固定门
# ====================================================================

@pytest.mark.parametrize("name", ["h", "x", "y", "z", "s", "sdg",
                                  "t", "tdg", "sx", "sxdg", "id"])
def test_roundtrip_1q_fixed(name):
    pytest.importorskip("qiskit")
    c = QuantumCircuit(1)
    getattr(c, name)(0)
    _roundtrip(c)


# ====================================================================
# qiskit round-trip: 单比特参数化门
# ====================================================================

@pytest.mark.parametrize("name,args", [
    ("rx", (0.37,)), ("ry", (-1.11,)), ("rz", (2.5,)), ("p", (0.42,)),
    ("u", (0.3, -0.7, 1.1)),
])
def test_roundtrip_1q_parametric(name, args):
    pytest.importorskip("qiskit")
    c = QuantumCircuit(1)
    getattr(c, name)(0, *args)
    _roundtrip(c)


# ====================================================================
# qiskit round-trip: 两比特门
# ====================================================================

@pytest.mark.parametrize("name", ["cx", "cy", "cz", "ch", "cs", "csdg",
                                  "ct", "ctdg", "swap", "iswap", "dcx",
                                  "ecr"])
def test_roundtrip_2q_fixed(name):
    pytest.importorskip("qiskit")
    c = QuantumCircuit(2)
    getattr(c, name)(0, 1)
    _roundtrip(c)


@pytest.mark.parametrize("name,args", [
    ("cp", (0.5,)), ("crx", (0.37,)), ("cry", (-1.11,)), ("crz", (2.5,)),
    ("cu", (0.3, -0.7, 1.1, 0.42)),
    ("rxx", (0.77,)), ("ryy", (-0.31,)), ("rzz", (1.23,)),
])
def test_roundtrip_2q_parametric(name, args):
    pytest.importorskip("qiskit")
    c = QuantumCircuit(2)
    if name == "cu":
        theta, phi, lam, gamma = args
        c.cu(theta, phi, lam, 0, 1, gamma)
    else:
        getattr(c, name)(args[0], 0, 1)
    _roundtrip(c)


# ====================================================================
# qiskit round-trip: 三比特及以上 / 特殊门
# ====================================================================

def test_roundtrip_ccx():
    pytest.importorskip("qiskit")
    c = QuantumCircuit(3)
    c.ccx(0, 1, 2)
    _roundtrip(c)


def test_roundtrip_cswap():
    pytest.importorskip("qiskit")
    c = QuantumCircuit(3)
    c.cswap(0, 1, 2)
    _roundtrip(c)


def test_roundtrip_mcx():
    pytest.importorskip("qiskit")
    c = QuantumCircuit(4)
    c.mcx([0, 1, 2], 3)
    _roundtrip(c)


def test_roundtrip_mcz():
    pytest.importorskip("qiskit")
    c = QuantumCircuit(3)
    c.mcz([0, 1, 2])
    _roundtrip(c)


def test_roundtrip_unitary():
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    rng = np.random.default_rng(42)
    z = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    q, r = np.linalg.qr(z)
    mat = q @ np.diag(np.diag(r) / np.abs(np.diag(r)))  # 酉化
    c = QuantumCircuit(2)
    c.unitary(mat, [0, 1], label="RND")
    qq, _ = _roundtrip(c)
    # 跨框架校验: qiskit 侧算子与 qsl 酉矩阵必须逐元一致 (比特序约定)
    np.testing.assert_allclose(Operator(qq).data, c.unitary_matrix(), atol=ATOL)


def test_roundtrip_barrier_preserved():
    pytest.importorskip("qiskit")
    c = QuantumCircuit(2)
    c.h(0)
    c.barrier()
    c.cx(0, 1)
    _, back = _roundtrip(c)
    assert back.count_ops().get("barrier", 0) == 1


def test_roundtrip_global_phase():
    pytest.importorskip("qiskit")
    c = QuantumCircuit(2, name="gp", global_phase=0.5)
    c.h(0)
    c.cz(0, 1)
    qq, back = _roundtrip(c)
    assert abs(float(qq.global_phase) - 0.5) <= ATOL
    assert back.name == "gp"


def test_roundtrip_combined_circuit():
    """多门组合电路 round-trip。"""
    pytest.importorskip("qiskit")
    c = QuantumCircuit(3, global_phase=-0.25)
    c.h(0); c.x(1); c.s(2); c.t(0)
    c.rx(0, 0.3); c.ry(1, -0.8); c.rz(2, 1.9); c.p(0, 0.44)
    c.u(1, 0.3, 0.6, -0.9)
    c.cx(0, 1); c.cy(1, 2); c.cz(2, 0); c.ch(0, 2)
    c.cp(0.7, 1, 2); c.crx(-0.4, 2, 0); c.crz(1.1, 0, 1)
    c.cu(0.2, 0.5, 0.8, 1, 2, 0.1)
    c.swap(0, 2); c.iswap(0, 1); c.dcx(1, 2); c.ecr(2, 0)
    c.rxx(0.9, 0, 1); c.ryy(-0.6, 1, 2); c.rzz(0.33, 0, 2)
    c.barrier([0, 1, 2])
    c.ccx(0, 1, 2); c.cswap(2, 0, 1)
    _roundtrip(c)


# ====================================================================
# qiskit round-trip: 符号参数
# ====================================================================

def test_roundtrip_symbolic_parameter():
    pytest.importorskip("qiskit")
    theta = Parameter("theta")
    c = QuantumCircuit(2)
    c.rx(0, theta)
    c.ry(1, 2 * theta + 0.5)
    c.cx(0, 1)

    qq = to_qiskit(c)
    assert {p.name for p in qq.parameters} == {"theta"}

    back = from_qiskit(qq)
    params = back.parameters
    assert {p.name for p in params} == {"theta"}
    # 同名参数映射为同一对象 (按名缓存)
    assert len(params) == 1

    v = 0.7
    m1 = c.bind_parameters({theta: v}).unitary_matrix()
    m2 = back.bind_parameters({Parameter("theta"): v}).unitary_matrix()
    np.testing.assert_allclose(m2, m1, atol=ATOL)


# ====================================================================
# from_qiskit: 未知门的展开与兜底
# ====================================================================

def test_from_qiskit_expands_definition():
    """带 .definition 的自定义门应被递归展开。"""
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit as _QQC

    inner = _QQC(1, name="myh")
    inner.h(0)
    custom = inner.to_gate()
    qq = _QQC(1)
    qq.append(custom, [0])

    back = from_qiskit(qq)
    ref = QuantumCircuit(1)
    ref.h(0)
    np.testing.assert_allclose(
        back.unitary_matrix(), ref.unitary_matrix(), atol=ATOL)


def test_from_qiskit_unitary_fallback():
    """无定义且无名的门应包成 qsl unitary 门。"""
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit as _QQC
    from qiskit.circuit import Gate as _QGate

    mat = np.array([[0, 1j], [1j, 0]])

    class WeirdGate(_QGate):
        """无 definition、仅提供矩阵的自定义门。"""

        def __init__(self):
            super().__init__("weird", 1, [])

        def to_matrix(self):
            return mat

    qq = _QQC(1)
    qq.append(WeirdGate(), [0])

    back = from_qiskit(qq)
    assert back.data[0].gate.name == "unitary"
    np.testing.assert_allclose(back.data[0].gate.to_matrix(), mat, atol=ATOL)


def test_from_qiskit_native_circuit():
    """直接转换原生 qiskit 电路 (含 Parameter)。"""
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit as _QQC
    from qiskit.circuit import Parameter as _QParam

    th = _QParam("phi")
    qq = _QQC(2, global_phase=0.125)
    qq.h(0)
    qq.rz(th, 1)
    qq.cx(0, 1)
    qq.barrier()

    back = from_qiskit(qq)
    assert {p.name for p in back.parameters} == {"phi"}
    assert abs(back.global_phase - 0.125) <= ATOL
    bound = back.bind_parameters({Parameter("phi"): 0.9})
    from qiskit.quantum_info import Operator
    ref = Operator(qq.assign_parameters({th: 0.9})).data
    np.testing.assert_allclose(bound.unitary_matrix(), ref, atol=ATOL)


# ====================================================================
# to_cirq (未安装则跳过)
# ====================================================================

def test_to_cirq_basic():
    cirq = pytest.importorskip("cirq")
    c = QuantumCircuit(2)
    c.h(0); c.cx(0, 1); c.rx(1, 0.3); c.rzz(-0.7, 0, 1)
    cc = to_cirq(c)
    qs = cirq.LineQubit.range(2)
    # cirq 的 qubit_order[0] 为最高位, 反转后与 qsl (q0=LSB) 对齐
    m = cc.unitary(qubit_order=list(reversed(qs)))
    np.testing.assert_allclose(m, c.unitary_matrix(), atol=ATOL)


def test_to_cirq_symbolic():
    cirq = pytest.importorskip("cirq")

    theta = Parameter("theta")
    c = QuantumCircuit(1)
    c.ry(0, theta)
    cc = to_cirq(c)
    assert cirq.parameter_names(cc) == {"theta"}
    resolved = cirq.resolve_parameters(cc, {"theta": 0.7})
    m = resolved.unitary(qubit_order=[cirq.LineQubit(0)])
    np.testing.assert_allclose(
        m, c.bind_parameters({theta: 0.7}).unitary_matrix(), atol=ATOL)


# ====================================================================
# 未安装时的 ImportError (monkeypatch 模拟, 始终运行)
# ====================================================================

def _block_import(monkeypatch, *modules):
    for mod in modules:
        monkeypatch.setitem(sys.modules, mod, None)


def test_to_qiskit_importerror(monkeypatch):
    _block_import(monkeypatch, "qiskit", "qiskit.circuit",
                  "qiskit.circuit.library")
    c = QuantumCircuit(1)
    c.h(0)
    with pytest.raises(ImportError, match="pip install qiskit"):
        to_qiskit(c)


def test_from_qiskit_importerror(monkeypatch):
    _block_import(monkeypatch, "qiskit", "qiskit.circuit")
    with pytest.raises(ImportError, match="pip install qiskit"):
        from_qiskit(object())


def test_to_cirq_importerror(monkeypatch):
    _block_import(monkeypatch, "cirq", "sympy")
    c = QuantumCircuit(1)
    c.h(0)
    with pytest.raises(ImportError, match="pip install cirq"):
        to_cirq(c)
