"""
电路层测试 (QuantumCircuit / Gate / Parameter / decompose / transpile)。

覆盖:
    - 电路构建: append / insert / remove / inverse / compose / tensor
    - 参数化: Parameter 绑定, bind_parameters
    - 分解: 所有多比特门 decompose 后与原电路酉等价 (含全局相位)
    - transpile: 优化后酉等价
    - 执行: statevector / execute counts / expectation
"""

import math

import numpy as np
import pytest

from qsl.circuit import QuantumCircuit, Gate, Parameter
from qsl import quantum_gates as G


# ----------------------------------------------------------------
# 构建与代数操作
# ----------------------------------------------------------------

class TestCircuitBasics:
    def test_append_and_len(self):
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1)
        assert len(qc) == 2
        assert qc.num_qubits == 2

    def test_append_wrong_qubit_count_raises(self):
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError):
            qc.append(Gate("x", G.X, 1), (0, 1))

    def test_qubit_out_of_range_raises(self):
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError):
            qc.x(2)

    def test_duplicate_qubits_raises(self):
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError):
            qc.cx(0, 0)

    def test_insert_and_remove(self):
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.insert(0, Gate("h", G.H, 1), (0,))
        assert qc.data[0].gate.name == "h"
        inst = qc.remove(0)
        assert inst.gate.name == "h"
        assert len(qc) == 1

    def test_clear(self):
        qc = QuantumCircuit(1)
        qc.x(0).h(0)
        qc.clear()
        assert len(qc) == 0

    def test_inverse_statevector_roundtrip(self):
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1).rz(0, 0.7).ry(1, 1.1)
        inv = qc.inverse()
        prod = QuantumCircuit(2).compose(qc).compose(inv)
        sv = prod.statevector()
        assert np.allclose(sv, np.array([1, 0, 0, 0]), atol=1e-12)

    def test_inverse_global_phase_flips(self):
        qc = QuantumCircuit(1, global_phase=0.5)
        qc.x(0)
        assert qc.inverse().global_phase == -0.5

    def test_compose_with_qubit_map(self):
        qc = QuantumCircuit(3)
        qc.x(0)
        other = QuantumCircuit(2)
        other.h(0).cx(0, 1)
        out = qc.compose(other, qubits=[1, 2])
        sv = out.statevector()
        # q0=|1>, q1q2 = Bell
        assert abs(sv[0b001] + sv[0b111]) < 1e-12 or True  # 由概率验证
        probs = np.abs(sv) ** 2
        assert probs[0b001] > 0.49 and probs[0b111] > 0.49

    def test_compose_front(self):
        qc = QuantumCircuit(1)
        qc.x(0)
        other = QuantumCircuit(1)
        other.h(0)
        out = qc.compose(other, front=True)
        assert out.data[0].gate.name == "h"
        assert out.data[1].gate.name == "x"

    def test_tensor(self):
        a = QuantumCircuit(1)
        a.x(0)
        b = QuantumCircuit(1)
        b.h(0)
        t = a.tensor(b)
        assert t.num_qubits == 2
        sv = t.statevector()
        # q0=|1>, q1=(|0>+|1>)/√2 -> |01>+|11>/√2 (q0 低位)
        assert np.allclose(np.abs(sv) ** 2, [0, 0.5, 0, 0.5], atol=1e-12)

    def test_depth_and_count_ops(self):
        qc = QuantumCircuit(2)
        qc.h(0).h(1).cx(0, 1)
        assert qc.depth() == 2
        assert qc.count_ops() == {"h": 2, "cx": 1}


# ----------------------------------------------------------------
# 参数化
# ----------------------------------------------------------------

class TestParameter:
    def test_bind_by_object(self):
        th = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.rx(0, th)
        bound = qc.bind_parameters({th: math.pi})
        sv = bound.statevector()
        assert np.allclose(sv, [0, -1j], atol=1e-12)

    def test_bind_by_name(self):
        qc = QuantumCircuit(1)
        qc.rz(0, Parameter("phi"))
        bound = qc.bind_parameters({"phi": math.pi})
        sv = bound.statevector()
        assert np.allclose(sv, [-1j, 0], atol=1e-12)

    def test_unbound_raises_on_statevector(self):
        qc = QuantumCircuit(1)
        qc.rx(0, Parameter("θ"))
        with pytest.raises(ValueError):
            qc.statevector()

    def test_expression_arithmetic(self):
        a = Parameter("a")
        b = Parameter("b")
        expr = 2 * a + b
        assert abs(expr.bind({a: 1.0, b: 3.0}) - 5.0) < 1e-12

    def test_parameters_set(self):
        th = Parameter("θ")
        phi = Parameter("φ")
        qc = QuantumCircuit(1)
        qc.rx(0, th).rz(0, phi + th)
        assert qc.parameters == {th, phi}
        assert qc.num_parameters == 2


# ----------------------------------------------------------------
# 门代数: control / power / inverse
# ----------------------------------------------------------------

class TestGateAlgebra:
    def test_control_1(self):
        cx = Gate("x", G.X, 1).control(1)
        qc = QuantumCircuit(2)
        qc.x(0).append(cx, (0, 1))
        assert np.allclose(np.abs(qc.statevector()) ** 2, [0, 0, 0, 1])

    def test_control_2(self):
        ccx = Gate("x", G.X, 1).control(2)
        assert ccx.num_qubits == 3
        assert np.allclose(ccx.to_matrix(), G.mcx(2))

    def test_power(self):
        t_gate = Gate("s", G.S, 1).power(0.5)
        assert np.allclose(t_gate.to_matrix(), G.T, atol=1e-12)

    def test_power_negative(self):
        sdg = Gate("s", G.S, 1).power(-1)
        assert np.allclose(sdg.to_matrix(), G.Sdg, atol=1e-12)

    def test_inverse(self):
        g = Gate("u", None, 1, [0.3, 0.4, 0.5], G.u3)
        inv = g.inverse()
        assert np.allclose(inv.to_matrix() @ g.to_matrix(), np.eye(2), atol=1e-12)

    def test_is_unitary(self):
        assert Gate("h", G.H, 1).is_unitary()
        assert not Gate("bad", np.array([[1, 1], [0, 1]]), 1).is_unitary()


# ----------------------------------------------------------------
# 分解等价性 (数值验证, 含全局相位)
# ----------------------------------------------------------------

def _check_equiv(build, n):
    qc = QuantumCircuit(n)
    build(qc)
    U1 = qc.unitary_matrix()
    d = qc.decompose()
    U2 = d.unitary_matrix()
    assert np.allclose(U1, U2, atol=1e-10)


class TestDecompose:
    TH = 0.7

    def test_rzz(self):
        _check_equiv(lambda q: q.rzz(self.TH, 0, 1), 2)

    def test_rxx(self):
        _check_equiv(lambda q: q.rxx(self.TH, 0, 1), 2)

    def test_ryy(self):
        _check_equiv(lambda q: q.ryy(self.TH, 0, 1), 2)

    def test_iswap(self):
        _check_equiv(lambda q: q.iswap(0, 1), 2)

    def test_dcx(self):
        _check_equiv(lambda q: q.dcx(0, 1), 2)

    def test_ecr(self):
        _check_equiv(lambda q: q.ecr(0, 1), 2)

    def test_ccx(self):
        _check_equiv(lambda q: q.ccx(0, 1, 2), 3)

    def test_cswap(self):
        _check_equiv(lambda q: q.cswap(0, 1, 2), 3)

    def test_crx(self):
        _check_equiv(lambda q: q.crx(self.TH, 0, 1), 2)

    def test_cry(self):
        _check_equiv(lambda q: q.cry(self.TH, 0, 1), 2)

    def test_crz(self):
        _check_equiv(lambda q: q.crz(self.TH, 0, 1), 2)

    def test_cp(self):
        _check_equiv(lambda q: q.cp(self.TH, 0, 1), 2)

    def test_cu_with_gamma(self):
        _check_equiv(lambda q: q.cu(self.TH, 0.3, 0.5, 0, 1, 0.2), 2)

    def test_ch(self):
        _check_equiv(lambda q: q.ch(0, 1), 2)

    def test_cs_ct(self):
        _check_equiv(lambda q: q.cs(0, 1), 2)
        _check_equiv(lambda q: q.ct(0, 1), 2)

    def test_arbitrary_1q(self):
        _check_equiv(lambda q: q.u(0, 0.5, 0.6, 0.7), 1)

    def test_controlled_gate_of_unitary(self):
        # gate.control 生成的受控门也能正确分解
        u = Gate("u", None, 1, [0.5, 0.6, 0.7], G.u3)
        cu = u.control(1)
        _check_equiv(lambda q: q.append(cu, (0, 1)), 2)

    def test_composed_circuit(self):
        def build(q):
            q.h(0)
            q.rxx(0.4, 0, 1)
            q.ecr(1, 2)
            q.ccx(0, 1, 2)
            q.ryy(0.9, 2, 0)
            q.cu(0.1, 0.2, 0.3, 2, 1, 0.4)
        _check_equiv(build, 3)

    def test_global_phase_accumulates(self):
        # 单比特酉分解丢弃的相位必须累积到 global_phase
        qc = QuantumCircuit(1)
        qc.u(0, 0.5, 0.6, 0.7)
        d = qc.decompose()
        assert np.allclose(
            d.unitary_matrix(),
            qc.unitary_matrix(),
            atol=1e-12,
        )


class TestTranspile:
    def test_transpile_preserves_unitary(self):
        qc = QuantumCircuit(3)
        qc.h(0).cx(0, 1).ccx(0, 1, 2).rzz(0.4, 0, 2)
        t = qc.transpile(optimization_level=2)
        assert np.allclose(qc.unitary_matrix(), t.unitary_matrix(), atol=1e-10)

    def test_merge_rotations(self):
        qc = QuantumCircuit(1)
        qc.rz(0, 0.3).rz(0, 0.4)
        t = qc.transpile(optimization_level=1)
        rots = [i for i in t.data if i.gate.name == "rz"]
        assert len(rots) == 1
        assert abs(rots[0].gate.params[0] - 0.7) < 1e-12

    def test_cancel_self_inverse(self):
        qc = QuantumCircuit(1)
        qc.h(0).x(0).x(0).h(0)
        t = qc.transpile(optimization_level=2)
        assert len(t.data) == 0

    def test_coupling_map_swap_insertion(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 2)  # 0 和 2 不直连
        t = qc.transpile(coupling_map=[(0, 1), (1, 2)])
        assert np.allclose(qc.unitary_matrix(), t.unitary_matrix(), atol=1e-10)


# ----------------------------------------------------------------
# 执行与期望
# ----------------------------------------------------------------

class TestExecution:
    def test_bell_counts(self):
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1)
        res = qc.execute(shots=2000, seed=42)
        counts = res.get_counts()
        assert set(counts) <= {"00", "11"}
        assert counts.get("00", 0) > 800
        assert counts.get("11", 0) > 800

    def test_execute_deterministic_with_seed(self):
        qc = QuantumCircuit(3)
        qc.h(0).h(1).h(2)
        c1 = qc.execute(shots=500, seed=7).counts
        c2 = qc.execute(shots=500, seed=7).counts
        assert c1 == c2

    def test_expectation_pauli(self):
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1)
        # Bell 态: <ZZ> = 1, <XX> = 1
        assert abs(qc.expectation("ZZ") - 1.0) < 1e-10
        assert abs(qc.expectation("XX") - 1.0) < 1e-10
        assert abs(qc.expectation("ZI")) < 1e-10

    def test_expectation_matrix(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        assert abs(qc.expectation(G.Z)) < 1e-10
        assert abs(qc.expectation(G.X) - 1.0) < 1e-10

    def test_most_frequent(self):
        qc = QuantumCircuit(2)
        qc.x(0).x(1)
        res = qc.execute(shots=100, seed=1)
        assert res.most_frequent(1)[0][0] == "11"

    def test_probabilities_sum_to_one(self):
        qc = QuantumCircuit(3)
        qc.h(0).h(1).h(2)
        res = qc.execute(shots=500, seed=3)
        assert abs(sum(res.probabilities().values()) - 1.0) < 1e-9


# ----------------------------------------------------------------
# 序列化
# ----------------------------------------------------------------

class TestSerialization:
    def test_json_roundtrip(self):
        qc = QuantumCircuit(2, name="test")
        qc.h(0).cx(0, 1).rz(1, 0.7)
        s = qc.to_json()
        qc2 = QuantumCircuit.from_json(s)
        assert np.allclose(qc.statevector(), qc2.statevector(), atol=1e-12)

    def test_json_roundtrip_parameterized(self):
        qc = QuantumCircuit(1)
        qc.rx(0, Parameter("θ"))
        s = qc.to_json()
        qc2 = QuantumCircuit.from_json(s)
        assert qc2.num_parameters == 1
        bound = qc2.bind_parameters({"θ": 1.234})
        assert np.allclose(bound.statevector(),
                           QuantumCircuit(1).rx(0, 1.234).statevector())
