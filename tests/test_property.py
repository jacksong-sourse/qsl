"""
基于 hypothesis 的性质测试 (property-based testing)。

性质:
    1. 随机单比特门序列组成的电路, unitary_matrix 是酉的 (U†U = I)。
    2. 随机 3 比特电路 (cx/h/t/rz 混合) 作用后的态向量保持归一。
    3. 任意电路 inverse().compose(原电路) 的酉矩阵 = 恒等。
    4. Parameter 表达式的 bind 结果与手动数值计算一致,
       且电路级 bind 与直接数值构造的电路酉矩阵一致。

每个性质 @settings(max_examples=30, deadline=None) 控制运行时长。
"""

import math

import numpy as np
from hypothesis import given, settings, strategies as st

from qsl.circuit import QuantumCircuit, Parameter

ATOL = 1e-8

_angle = st.floats(min_value=-2 * math.pi, max_value=2 * math.pi,
                   allow_nan=False, allow_infinity=False)
_small = st.floats(min_value=-4.0, max_value=4.0,
                   allow_nan=False, allow_infinity=False)

# (门名, 角度) — 固定门忽略角度
_1q_gate = st.tuples(st.sampled_from(["h", "x", "y", "z", "s", "t",
                                      "rx", "ry", "rz"]), _angle)


def _apply_1q(qc: QuantumCircuit, name: str, angle: float):
    if name in ("rx", "ry", "rz"):
        getattr(qc, name)(0, angle)
    else:
        getattr(qc, name)(0)


# ====================================================================
# 性质 1: 随机单比特门序列的酉矩阵是酉的
# ====================================================================

@given(st.lists(_1q_gate, min_size=1, max_size=20))
@settings(max_examples=30, deadline=None)
def test_random_1q_sequence_is_unitary(ops):
    qc = QuantumCircuit(1)
    for name, angle in ops:
        _apply_1q(qc, name, angle)
    u = qc.unitary_matrix()
    np.testing.assert_allclose(u.conj().T @ u, np.eye(2), atol=ATOL)


# ====================================================================
# 性质 2: 随机 3 比特电路 (cx/h/t/rz) 态向量保持归一
# ====================================================================

_3q_gate = st.tuples(
    st.sampled_from(["h", "t", "rz", "cx"]),
    st.integers(min_value=0, max_value=2),
    st.integers(min_value=0, max_value=2),
    _angle,
)


@given(st.lists(_3q_gate, min_size=1, max_size=15))
@settings(max_examples=30, deadline=None)
def test_random_3q_statevector_normalized(ops):
    qc = QuantumCircuit(3)
    for name, q0, q1, angle in ops:
        if name == "cx":
            if q0 == q1:
                continue
            qc.cx(q0, q1)
        elif name == "rz":
            qc.rz(q0, angle)
        else:
            getattr(qc, name)(q0)
    sv = qc.statevector()
    np.testing.assert_allclose(np.linalg.norm(sv), 1.0, atol=ATOL)


# ====================================================================
# 性质 3: inverse().compose(原电路) = 恒等
# ====================================================================

_2q_gate = st.tuples(
    st.sampled_from(["h", "x", "s", "t", "rx", "ry", "rz", "cx"]),
    st.integers(min_value=0, max_value=1),
    st.integers(min_value=0, max_value=1),
    _angle,
)


@given(st.lists(_2q_gate, min_size=1, max_size=12))
@settings(max_examples=30, deadline=None)
def test_inverse_compose_is_identity(ops):
    qc = QuantumCircuit(2)
    for name, q0, q1, angle in ops:
        if name == "cx":
            if q0 == q1:
                continue
            qc.cx(q0, q1)
        elif name in ("rx", "ry", "rz"):
            getattr(qc, name)(q0, angle)
        else:
            getattr(qc, name)(q0)
    inv_then_qc = qc.inverse().compose(qc)
    np.testing.assert_allclose(
        inv_then_qc.unitary_matrix(), np.eye(4), atol=ATOL)


# ====================================================================
# 性质 4: Parameter 表达式 bind 与手动计算一致
# ====================================================================

@given(_small, _small, _small, _small)
@settings(max_examples=30, deadline=None)
def test_parameter_expression_bind(x, y, a, b):
    px, py = Parameter("x"), Parameter("y")
    expr = (a * px - b * py) / 2 + 1
    result = expr.bind({px: x, py: y})
    expected = (a * x - b * y) / 2 + 1
    assert math.isclose(result, expected, rel_tol=1e-12, abs_tol=1e-10)


@given(_angle, _small, _small)
@settings(max_examples=30, deadline=None)
def test_circuit_bind_matches_numeric(x, a, b):
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.rx(0, a * theta + b)
    bound = qc.bind_parameters({theta: x})

    ref = QuantumCircuit(1)
    ref.rx(0, a * x + b)
    np.testing.assert_allclose(
        bound.unitary_matrix(), ref.unitary_matrix(), atol=ATOL)
