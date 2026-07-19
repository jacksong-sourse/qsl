"""OpenQASM 导入/导出测试。"""

import math

import numpy as np
import pytest

from qsl.circuit import (
    QuantumCircuit, Parameter,
    dumps_qasm2, loads_qasm2, dumps_qasm3, QASMParseError,
)


# ====================================================================
# 辅助
# ====================================================================

def assert_circuit_equiv(qc1: QuantumCircuit, qc2: QuantumCircuit,
                         atol: float = 1e-10):
    """
    比较两个电路的完整酉矩阵。

    unitary_matrix() 本身已乘上 e^{i·global_phase}, 即比较的是
    U_gates1 · e^{i·phase1} vs U_gates2 · e^{i·phase2}, 全局相位差异已补正。
    """
    assert qc1.num_qubits == qc2.num_qubits
    u1 = qc1.unitary_matrix()
    u2 = qc2.unitary_matrix()
    np.testing.assert_allclose(u1, u2, atol=atol)


def roundtrip(qc: QuantumCircuit, atol: float = 1e-10) -> QuantumCircuit:
    text = dumps_qasm2(qc)
    qc2 = loads_qasm2(text)
    assert_circuit_equiv(qc, qc2, atol=atol)
    return qc2


# ====================================================================
# Bell 电路 round-trip
# ====================================================================

def test_bell_roundtrip_statevector_and_unitary():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    text = dumps_qasm2(qc)
    assert text.startswith('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];')
    assert "h q[0];" in text
    assert "cx q[0],q[1];" in text

    qc2 = loads_qasm2(text)
    sv1, sv2 = qc.statevector(), qc2.statevector()
    fidelity = abs(np.vdot(sv1, sv2)) ** 2
    assert fidelity > 1 - 1e-10
    np.testing.assert_allclose(qc.unitary_matrix(), qc2.unitary_matrix(),
                               atol=1e-12)


# ====================================================================
# 单门 round-trip: unitary_matrix 一致
# ====================================================================

@pytest.mark.parametrize("name", ["h", "x", "y", "z", "s", "sdg", "t", "tdg",
                                  "sx", "sxdg"])
def test_fixed_1q_gate_roundtrip(name):
    qc = QuantumCircuit(1)
    getattr(qc, name)(0)
    roundtrip(qc)


@pytest.mark.parametrize("name,params", [
    ("rx", (0.7,)), ("ry", (0.8,)), ("rz", (0.9,)), ("p", (1.1,)),
    ("u", (0.3, 0.4, 0.5)),
])
def test_param_1q_gate_roundtrip(name, params):
    qc = QuantumCircuit(1)
    getattr(qc, name)(0, *params)
    roundtrip(qc)


@pytest.mark.parametrize("name", ["cx", "cy", "cz", "ch", "swap", "iswap",
                                  "dcx", "ecr"])
def test_fixed_2q_gate_roundtrip(name):
    qc = QuantumCircuit(2)
    getattr(qc, name)(0, 1)
    roundtrip(qc)


@pytest.mark.parametrize("name,params", [
    ("crx", (0.7,)), ("cry", (0.8,)), ("crz", (0.9,)), ("cp", (1.1,)),
    ("cu", (0.3, 0.4, 0.5, 0.0)),
    ("cu", (0.3, 0.4, 0.5, 0.6)),   # gamma 非 0 -> qelib1 cu(θ,φ,λ,γ)
    ("rxx", (0.7,)), ("ryy", (0.8,)), ("rzz", (0.9,)),
])
def test_param_2q_gate_roundtrip(name, params):
    qc = QuantumCircuit(2)
    if name == "cu":
        theta, phi, lam, gamma = params
        qc.cu(theta, phi, lam, 0, 1, gamma)
    else:
        getattr(qc, name)(params[0], 0, 1)
    roundtrip(qc)


@pytest.mark.parametrize("name", ["cs", "ct", "csdg", "ctdg"])
def test_controlled_phase_gate_roundtrip(name):
    qc = QuantumCircuit(2)
    getattr(qc, name)(0, 1)
    roundtrip(qc)


def test_ccx_roundtrip():
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    roundtrip(qc)


def test_cswap_roundtrip():
    qc = QuantumCircuit(3)
    qc.cswap(0, 1, 2)
    roundtrip(qc)


def test_mcx_roundtrip():
    qc = QuantumCircuit(4)
    qc.mcx([0, 1, 2], 3)
    roundtrip(qc)


def test_mcx_single_control_roundtrip():
    qc = QuantumCircuit(2)
    qc.mcx([0], 1)
    roundtrip(qc)


def test_mcz_roundtrip():
    qc = QuantumCircuit(3)
    qc.mcz([0, 1, 2])
    roundtrip(qc)


def test_barrier_roundtrip():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.barrier([0, 1])
    qc.cx(0, 2)
    text = dumps_qasm2(qc)
    assert "barrier q[0],q[1];" in text
    qc2 = loads_qasm2(text)
    assert qc2.count_ops().get("barrier") == 1
    assert_circuit_equiv(qc, qc2)


# ====================================================================
# 解析
# ====================================================================

def test_parameter_expression_parsing():
    text = (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "qreg q[1];\n"
        "rx(pi/2) q[0];\n"
        "ry(-pi/4) q[0];\n"
        "rz(2*pi/3 + 0.25) q[0];\n"
        "p((pi)) q[0];\n"
        "u3(pi/2, pi/4, -pi/8) q[0];\n"
    )
    qc = loads_qasm2(text)
    got = [inst.gate.params for inst in qc.data]
    assert got[0][0] == pytest.approx(math.pi / 2)
    assert got[1][0] == pytest.approx(-math.pi / 4)
    assert got[2][0] == pytest.approx(2 * math.pi / 3 + 0.25)
    assert got[3][0] == pytest.approx(math.pi)
    assert got[4] == pytest.approx([math.pi / 2, math.pi / 4, -math.pi / 8])


def test_comments_and_measure_ignored():
    text = (
        "// 文件头注释\n"
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "qreg q[1];\n"
        "creg c[1];\n"
        "h q[0]; // 行尾注释\n"
        "measure q[0] -> c[0];\n"
    )
    qc = loads_qasm2(text)
    assert qc.size() == 1
    assert qc.data[0].gate.name == "h"


def test_qelib1_alias_gates_parsing():
    """u1/u2/u3/cu1/cu3 等 qelib1 名映射到 p/u/cp/cu。"""
    text = (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "qreg q[2];\n"
        "u1(pi/3) q[0];\n"
        "u2(pi/4, pi/5) q[1];\n"
        "cu1(pi/6) q[0],q[1];\n"
        "cu3(pi/7, pi/8, pi/9) q[1],q[0];\n"
        "barrier q;\n"
    )
    qc = loads_qasm2(text)
    assert [inst.gate.name for inst in qc.data] == ["p", "u", "cp", "cu", "barrier"]
    assert qc.data[0].gate.params[0] == pytest.approx(math.pi / 3)
    assert qc.data[1].gate.params == pytest.approx(
        [math.pi / 2, math.pi / 4, math.pi / 5])
    assert qc.data[2].gate.params[0] == pytest.approx(math.pi / 6)
    assert qc.data[4].qubits == (0, 1)


def test_unbound_parameter_export_raises():
    theta = Parameter("θ")
    qc = QuantumCircuit(1)
    qc.rx(0, theta)
    with pytest.raises(ValueError, match="bind_parameters"):
        dumps_qasm2(qc)


def test_unknown_gate_parse_error_with_line():
    text = (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "qreg q[2];\n"
        "h q[0];\n"
        "foo q[1];\n"
    )
    with pytest.raises(QASMParseError) as exc_info:
        loads_qasm2(text)
    assert exc_info.value.line == 5
    assert "foo" in str(exc_info.value)
    assert "5" in str(exc_info.value)


def test_bad_expression_parse_error():
    text = (
        "OPENQASM 2.0;\n"
        "qreg q[1];\n"
        "rx(__import__('os')) q[0];\n"
    )
    with pytest.raises(QASMParseError):
        loads_qasm2(text)


# ====================================================================
# QASM 3.0 导出
# ====================================================================

def test_qasm3_export_header_and_gates():
    qc = QuantumCircuit(2, global_phase=0.25)
    qc.h(0)
    qc.cx(0, 1)
    qc.rx(0, 0.5)
    qc.sx(1)
    qc.cp(0.7, 0, 1)
    text = dumps_qasm3(qc)
    assert text.startswith('OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;')
    assert "h q[0];" in text
    assert "cx q[0],q[1];" in text
    assert "rx(0.5) q[0];" in text
    assert "sx q[1];" in text
    assert "cp(" in text
    assert "// global phase: 0.25" in text


def test_qasm3_u_and_cu_names():
    qc = QuantumCircuit(2)
    qc.u(0, 0.1, 0.2, 0.3)
    qc.cu(0.1, 0.2, 0.3, 0, 1, 0.4)
    text = dumps_qasm3(qc)
    assert "U(" in text
    assert "cu(" in text


# ====================================================================
# 边界情形
# ====================================================================

def test_empty_circuit_roundtrip():
    qc = QuantumCircuit(3)
    text = dumps_qasm2(qc)
    assert "qreg q[3];" in text
    assert "// global phase" not in text
    qc2 = loads_qasm2(text)
    assert qc2.num_qubits == 3
    assert qc2.size() == 0
    assert qc2.global_phase == 0.0


def test_global_phase_roundtrip():
    qc = QuantumCircuit(2, global_phase=0.7)
    qc.h(0)
    qc.cx(0, 1)
    text = dumps_qasm2(qc)
    assert "// global phase:" in text
    qc2 = loads_qasm2(text)
    assert qc2.global_phase == pytest.approx(0.7)
    assert_circuit_equiv(qc, qc2)


def test_global_phase_from_decomposition_roundtrip():
    """sx 的 ZYZ 分解残余相位 (π/4) 应进入 global phase 注释并往返一致。"""
    qc = QuantumCircuit(1)
    qc.sx(0)
    text = dumps_qasm2(qc)
    assert "// global phase:" in text
    qc2 = roundtrip(qc)
    assert qc2.global_phase == pytest.approx(math.pi / 4)
