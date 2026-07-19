"""qsl.ai.verifier 自动验证器测试: 每个 verify_* 的正例 + 反例。"""

import numpy as np
import pytest

from qsl.ai.verifier import (
    VerificationResult,
    hamiltonian_matrix,
    verify,
    verify_grover,
    verify_qaoa,
    verify_sat,
    verify_shor,
    verify_vqe,
)


# ----------------------------------------------------------------
# verify_shor
# ----------------------------------------------------------------

class TestVerifyShor:
    def test_correct_factors(self):
        r = verify_shor(15, [3, 5])
        assert r.passed
        assert r.details["product"] == 15

    def test_correct_factors_unordered(self):
        assert verify_shor(15, [5, 3]).passed

    def test_correct_multiple_factors(self):
        assert verify_shor(30, [2, 3, 5]).passed

    def test_prime_number_itself(self):
        assert verify_shor(13, [13]).passed

    def test_wrong_product(self):
        r = verify_shor(15, [3, 4])
        assert not r.passed
        assert "回乘" in r.message

    def test_factor_one(self):
        r = verify_shor(15, [1, 15])
        assert not r.passed

    def test_composite_factor(self):
        # 3*9=27 回乘正确, 但 9 不是素数
        r = verify_shor(27, [3, 9])
        assert not r.passed
        assert "素数" in r.message

    def test_failed_factoring_returns_N(self):
        r = verify_shor(15, [15])
        assert not r.passed

    def test_empty_factors(self):
        assert not verify_shor(15, []).passed

    def test_non_integer_factor(self):
        assert not verify_shor(15, [3.5, 5]).passed


# ----------------------------------------------------------------
# verify_sat
# ----------------------------------------------------------------

class TestVerifySat:
    CLAUSES_STR = ["x0 & x1", "~x1 | x2"]

    def test_string_clauses_satisfied_dict(self):
        r = verify_sat(self.CLAUSES_STR, {"x0": True, "x1": True, "x2": True})
        assert r.passed

    def test_string_clauses_unsatisfied(self):
        r = verify_sat(self.CLAUSES_STR, {"x0": True, "x1": False, "x2": False})
        assert not r.passed
        assert len(r.details["clause_results"]) == 2

    def test_string_clauses_second_clause_fails(self):
        # x1=True, x2=False -> (~x1 | x2) = False
        r = verify_sat(self.CLAUSES_STR, {"x0": True, "x1": True, "x2": False})
        assert not r.passed

    def test_bitstring_assignment(self):
        # "111" -> x0=x1=x2=1
        assert verify_sat(self.CLAUSES_STR, "111").passed
        assert not verify_sat(self.CLAUSES_STR, "001").passed

    def test_int_assignment(self):
        assert verify_sat(self.CLAUSES_STR, 0b111).passed

    def test_int_literal_clauses_satisfied(self):
        # (x1 | ~x2) & (x2), 1 基文字: 唯一解 x1=True, x2=True
        clauses = [(1, -2), (2,)]
        assert verify_sat(clauses, {"x0": True, "x1": True}).passed

    def test_int_literal_clauses_unsatisfied(self):
        clauses = [(1, -2), (2,)]
        r = verify_sat(clauses, {"x0": False, "x1": True})
        assert not r.passed

    def test_int_literal_with_one_based_dict(self):
        clauses = [(1, -2), (2,)]
        assert verify_sat(clauses, {1: True, 2: True}).passed
        assert not verify_sat(clauses, {1: False, 2: True}).passed

    def test_int_literal_with_bit_int(self):
        clauses = [(1, -2), (2,)]
        assert verify_sat(clauses, 0b11).passed
        assert not verify_sat(clauses, 0b10).passed

    def test_empty_clauses(self):
        assert not verify_sat([], {"x0": True}).passed


# ----------------------------------------------------------------
# verify_qaoa
# ----------------------------------------------------------------

# 4 节点环图 MaxCut 的 Ising 代价矩阵 (最优 cost = -4)
C4_ADJ = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
], dtype=float)


class TestVerifyQaoa:
    def test_optimal_bitstring_passes(self):
        # 交替分割 0101 / 1010 都是最优 (cost=-4)
        assert verify_qaoa(C4_ADJ, 0b0101).passed
        assert verify_qaoa(C4_ADJ, 0b1010).passed

    def test_worst_bitstring_fails(self):
        # 全部同侧 cost=+4, 劣于枚举最优 -4
        r = verify_qaoa(C4_ADJ, 0b0000)
        assert not r.passed
        assert r.details["classical_best_cost"] == -4.0
        assert r.details["quantum_cost"] == 4.0

    def test_exact_baseline(self):
        r = verify_qaoa(C4_ADJ, 0b0101, baseline="exact")
        assert r.passed
        assert r.details["baseline"] == "enumerate"

    def test_bitstring_str_input(self):
        assert verify_qaoa(C4_ADJ, "0101").passed

    def test_random_baseline_large_n(self):
        # n=17 > 16 -> 随机采样路径; 植入最优解 bitstring=0 (cost=-17)
        Q = np.diag([-1.0] * 17)
        r = verify_qaoa(Q, 0, baseline="random")
        assert r.passed
        assert r.details["baseline"] == "random1000"
        assert r.details["quantum_cost"] == -17.0

    def test_qubo_encoding(self):
        # QUBO: E(0)=0, E(1)=2, E(2)=10, E(3)=11 -> 最优 bitstring 0
        Q = np.array([[2.0, -1.0], [-1.0, 10.0]])
        assert verify_qaoa(Q, 0, encoding="qubo").passed
        assert not verify_qaoa(Q, 2, encoding="qubo").passed


# ----------------------------------------------------------------
# verify_grover
# ----------------------------------------------------------------

class TestVerifyGrover:
    def test_hit(self):
        r = verify_grover([6, 11], {6: 48, 11: 46, 3: 3, 0: 3}, n_qubits=4)
        assert r.passed
        assert r.details["success_rate"] == 94 / 100
        assert r.details["classical_random_rate"] == 2 / 16

    def test_top_k_miss(self):
        r = verify_grover([6, 11], {6: 5, 1: 50, 2: 45}, n_qubits=4)
        assert not r.passed
        assert "标记集合" in r.message

    def test_success_rate_not_better_than_random(self):
        # top-1 命中 (并列时标记态排前) 但成功率 1/16 不超过经典随机 1/16
        counts = {6: 1}
        counts.update({i: 1 for i in range(16) if i != 6})
        r = verify_grover([6], counts, n_qubits=4)
        assert not r.passed
        assert "经典随机" in r.message

    def test_empty_measured(self):
        assert not verify_grover([6], {}).passed

    def test_empty_marked(self):
        assert not verify_grover([], {6: 10}).passed

    def test_infer_n_qubits(self):
        r = verify_grover([6, 11], {6: 48, 11: 46})
        assert r.passed
        assert r.details["n_qubits"] == 4


# ----------------------------------------------------------------
# verify_vqe
# ----------------------------------------------------------------

class TestVerifyVqe:
    def test_simple_z_hamiltonian(self):
        # H = Z, 基态能量 -1 (|1>)
        assert verify_vqe(-1.0, [(1.0, "Z")]).passed

    def test_energy_below_ground_fails(self):
        r = verify_vqe(-1.5, [(1.0, "Z")])
        assert not r.passed
        assert "下界" in r.message

    def test_h2_reasonable_energy(self):
        from qsl.algorithms.vqe import VQE
        h2 = VQE.h2_hamiltonian()
        ground = float(np.linalg.eigvalsh(hamiltonian_matrix(h2))[0].real)
        # 略高于基态: 通过; 低于基态: 违反物理下界, 不通过
        assert verify_vqe(ground + 0.01, h2).passed
        assert not verify_vqe(ground - 0.01, h2).passed

    def test_h2_energy_too_low(self):
        from qsl.algorithms.vqe import VQE
        h2 = VQE.h2_hamiltonian()
        r = verify_vqe(-2.0, h2)
        assert not r.passed

    def test_hf_upper_bound(self):
        terms = [(1.0, "Z")]
        assert verify_vqe(-1.0, terms, hf_energy=-0.5).passed
        assert not verify_vqe(-0.2, terms, hf_energy=-0.5).passed

    def test_ansatz_state_cross_check(self):
        # |00> 对 H = ZZ 的能量为 +1
        state = np.zeros(4, dtype=complex)
        state[0] = 1.0
        assert verify_vqe(1.0, [(1.0, "ZZ")], ansatz_state=state).passed
        r = verify_vqe(-1.0, [(1.0, "ZZ")], ansatz_state=state)
        assert not r.passed

    def test_nan_energy(self):
        assert not verify_vqe(float("nan"), [(1.0, "Z")]).passed

    def test_hamiltonian_matrix_pauli_convention(self):
        # XI: X 作用在 qubit0 (LSB); 最小本征值 -1
        H = hamiltonian_matrix([(2.0, "XI")])
        assert np.isclose(np.linalg.eigvalsh(H)[0].real, -2.0)


# ----------------------------------------------------------------
# verify 分发器
# ----------------------------------------------------------------

class TestVerifyDispatcher:
    def test_shor(self):
        r = verify("shor", {"factors": [3, 5]}, {"N": 15})
        assert r.passed
        r = verify("shor", {"factors": [3, 4]}, {"N": 15})
        assert not r.passed

    def test_sat(self):
        r = verify("sat", {"assignment": {"x0": True, "x1": True}},
                   {"clauses": ["x0 & x1"]})
        assert r.passed

    def test_qaoa(self):
        r = verify("qaoa", {"bitstring": 0b0101}, {"cost_matrix": C4_ADJ})
        assert r.passed

    def test_grover(self):
        r = verify("grover", {"measured": {6: 90, 11: 8, 0: 2}},
                   {"marked_states": [6, 11], "n_qubits": 4})
        assert r.passed

    def test_vqe(self):
        r = verify("vqe", {"energy": -1.0},
                   {"hamiltonian_terms": [(1.0, "Z")]})
        assert r.passed

    def test_unknown_algorithm(self):
        r = verify("teleportation", {}, {})
        assert not r.passed
        assert "未知算法" in r.message
