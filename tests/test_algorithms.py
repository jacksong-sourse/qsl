"""
测试 qsl.algorithms 中的所有量子算法。

覆盖:
    - QuantumFourierTransform (QFT)
    - ShorSolver (Shor因子分解)
    - QAOA (量子近似优化)
    - VQE (变分量子特征求解器)
"""

import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from qsl.algorithms.qft import QuantumFourierTransform
from qsl.algorithms.shor import ShorSolver
from qsl.algorithms.qaoa import QAOA
from qsl.algorithms.vqe import VQE


# ============================================================================
#  TestQFT
# ============================================================================

class TestQFT:
    """测试 QuantumFourierTransform 类。"""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.n_qubits = 3
        self.qft = QuantumFourierTransform(self.n_qubits)

    # -- 矩阵正确性 -------------------------------------------------------

    def test_matrix_is_square(self):
        """QFT 矩阵应为 N×N 方阵。"""
        F = self.qft.get_matrix()
        N = 1 << self.n_qubits
        assert F.shape == (N, N)

    def test_matrix_is_unitary(self):
        """QFT 矩阵应为酉矩阵：F @ F† = I。"""
        F = self.qft.get_matrix()
        identity = np.eye(F.shape[0], dtype=complex)
        np.testing.assert_allclose(F @ F.conj().T, identity, atol=1e-12)

    def test_matrix_first_element_value(self):
        """F[0, 0] 应等于 1/√N。"""
        F = self.qft.get_matrix()
        N = 1 << self.n_qubits
        expected = 1.0 / np.sqrt(N)
        assert abs(F[0, 0] - expected) < 1e-12

    def test_matrix_symmetry(self):
        """F[j, k] 应等于 F[k, j]（QFT 矩阵对称）。"""
        F = self.qft.get_matrix()
        np.testing.assert_allclose(F, F.T, atol=1e-12)

    def test_matrix_rows_unit_norm(self):
        """每一行的模应为 1（归一化）。"""
        F = self.qft.get_matrix()
        for row in F:
            norm = np.sqrt(np.sum(np.abs(row) ** 2))
            assert abs(norm - 1.0) < 1e-12

    # -- build_circuit ---------------------------------------------------

    def test_build_circuit_returns_list(self):
        """build_circuit 应返回列表。"""
        circuit = self.qft.build_circuit()
        assert isinstance(circuit, list)

    def test_build_circuit_non_empty(self):
        """build_circuit 不应返回空列表。"""
        circuit = self.qft.build_circuit()
        assert len(circuit) > 0

    def test_build_circuit_entries_are_dicts(self):
        """每个电路条目应为字典。"""
        circuit = self.qft.build_circuit()
        for gate_op in circuit:
            assert isinstance(gate_op, dict)

    def test_build_circuit_first_gate_is_hadamard(self):
        """电路第一个门应为 H 门。"""
        circuit = self.qft.build_circuit()
        assert circuit[0]['gate'] == 'H'

    def test_build_circuit_contains_cphase(self):
        """电路应包含 CPHASE 门。"""
        circuit = self.qft.build_circuit()
        gates = {op['gate'] for op in circuit}
        assert 'CPHASE' in gates

    # -- inverse ---------------------------------------------------------

    def test_inverse_returns_list(self):
        """inverse 应返回列表。"""
        inv = self.qft.inverse()
        assert isinstance(inv, list)

    def test_inverse_same_length_as_circuit(self):
        """逆电路长度应与正向电路相同。"""
        circuit = self.qft.build_circuit()
        inv = self.qft.inverse()
        assert len(inv) == len(circuit)

    def test_inverse_gates_same_set(self):
        """逆电路应包含相同的门类型集合。"""
        circuit = self.qft.build_circuit()
        inv = self.qft.inverse()
        gates_fwd = {op['gate'] for op in circuit}
        gates_inv = {op['gate'] for op in inv}
        assert gates_fwd == gates_inv

    def test_inverse_negates_phases(self):
        """逆电路的 CPHASE 角度应为负值。"""
        inv = self.qft.inverse()
        for op in inv:
            if op['gate'] == 'CPHASE':
                assert op['params']['angle'] < 0

    def test_inverse_then_forward_identity(self):
        """先应用正向 QFT 再应用逆 QFT 应恢复原状态。"""
        circuit = self.qft.build_circuit()
        F = self.qft.get_matrix()
        F_inv = F.conj().T
        identity = F_inv @ F
        np.testing.assert_allclose(identity, np.eye(F.shape[0], dtype=complex), atol=1e-12)

    # -- apply -----------------------------------------------------------

    def test_apply_on_zero_state(self):
        """对 |0> 应用 QFT 应得到所有分量相等的叠加态。"""
        N = 1 << self.n_qubits
        state = np.zeros(N, dtype=complex)
        state[0] = 1.0
        result = self.qft.apply(state)
        # 每个分量应为 1/√N
        expected_abs = 1.0 / np.sqrt(N)
        for val in result:
            assert abs(abs(val) - expected_abs) < 1e-12

    def test_apply_preserves_norm(self):
        """QFT 是酉变换，应保持模长。"""
        N = 1 << self.n_qubits
        rng = np.random.RandomState(42)
        state = rng.randn(N) + 1j * rng.randn(N)
        state /= np.linalg.norm(state)
        result = self.qft.apply(state)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-12

    def test_apply_linearity(self):
        """QFT 应为线性变换。"""
        N = 1 << self.n_qubits
        rng = np.random.RandomState(99)
        a = rng.randn(N) + 1j * rng.randn(N)
        b = rng.randn(N) + 1j * rng.randn(N)
        result_combined = self.qft.apply(a + b)
        result_separate = self.qft.apply(a) + self.qft.apply(b)
        np.testing.assert_allclose(result_combined, result_separate, atol=1e-12)

    # -- 不同 n_qubits ---------------------------------------------------

    def test_single_qubit_qft_is_hadamard(self):
        """单量子比特 QFT 矩阵应等于 Hadamard 矩阵。"""
        qft1 = QuantumFourierTransform(1)
        F = qft1.get_matrix()
        expected = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        np.testing.assert_allclose(F, expected, atol=1e-12)

    def test_two_qubit_qft_shape(self):
        """2 量子比特 QFT 矩阵应为 4×4。"""
        qft2 = QuantumFourierTransform(2)
        F = qft2.get_matrix()
        assert F.shape == (4, 4)

    def test_four_qubit_qft_unitary(self):
        """4 量子比特 QFT 仍应为酉矩阵。"""
        qft4 = QuantumFourierTransform(4)
        F = qft4.get_matrix()
        identity = np.eye(F.shape[0], dtype=complex)
        np.testing.assert_allclose(F @ F.conj().T, identity, atol=1e-12)


# ============================================================================
#  TestShor
# ============================================================================

class TestShor:
    """测试 ShorSolver 类。"""

    def test_factor_15_finds_factors(self):
        """factor(15) 应找到因子 3 和 5。"""
        solver = ShorSolver(15)
        result = solver.factor()
        result_sorted = sorted(result)
        assert result_sorted == [3, 5], f"Expected [3, 5], got {result_sorted}"

    def test_factor_21_finds_factors(self):
        """factor(21) 应找到因子 3 和 7。"""
        solver = ShorSolver(21)
        result = solver.factor()
        result_sorted = sorted(result)
        assert result_sorted == [3, 7], f"Expected [3, 7], got {result_sorted}"

    def test_factor_1_cannot_instantiate(self):
        """N=1 不能实例化 ShorSolver（N 必须 >= 2）。"""
        with pytest.raises(ValueError, match="N must be >= 2"):
            ShorSolver(1)

    def test_N_less_than_2_raises_error(self):
        """N < 2 时应抛出 ValueError。"""
        with pytest.raises(ValueError, match="N must be >= 2"):
            ShorSolver(0)

    def test_even_number_6_returns_2_3(self):
        """偶数 N=6 应返回因子 [2, 3]。"""
        solver = ShorSolver(6)
        result = solver.factor()
        result_sorted = sorted(result)
        assert result_sorted == [2, 3]

    def test_factor_9_returns_3_3(self):
        """N=9（幂次）应返回 [3, 3]。"""
        solver = ShorSolver(9)
        result = solver.factor()
        result_sorted = sorted(result)
        assert result_sorted == [3, 3], f"Expected [3, 3], got {result_sorted}"

    def test_factor_35_finds_factors(self):
        """factor(35) 应找到因子 5 和 7。"""
        solver = ShorSolver(35)
        result = solver.factor()
        result_sorted = sorted(result)
        assert result_sorted == [5, 7], f"Expected [5, 7], got {result_sorted}"

    def test_factors_property(self):
        """factors 属性应在 factor() 调用后可用。"""
        solver = ShorSolver(15)
        assert solver.factors is None
        solver.factor()
        assert solver.factors is not None

    def test_repr(self):
        """__repr__ 应包含 N。"""
        solver = ShorSolver(15)
        assert "15" in repr(solver)

    def test_product_of_factors_equals_N(self):
        """因子乘积应等于 N。"""
        for N in [15, 21, 33, 35]:
            solver = ShorSolver(N)
            factors = solver.factor()
            product = 1
            for f in factors:
                product *= f
            assert product == N, f"{N}: factors {factors} multiply to {product}"

    # -- 经典辅助方法 -----------------------------------------------------

    def test_modular_pow_basic(self):
        """_modular_pow 的基本测试。"""
        assert ShorSolver._modular_pow(2, 3, 5) == 3   # 2^3 mod 5 = 8 mod 5 = 3
        assert ShorSolver._modular_pow(3, 4, 7) == 4   # 3^4 mod 7 = 81 mod 7 = 4
        assert ShorSolver._modular_pow(7, 0, 15) == 1  # a^0 = 1 mod N

    def test_is_power_detects_power(self):
        """_is_power 应检测 N = a^b 的情况。"""
        solver = ShorSolver(27)  # 27 = 3^3
        base = solver._is_power()
        assert base == 3

    def test_is_power_returns_none_for_non_power(self):
        """_is_power 对非幂次数应返回 None。"""
        solver = ShorSolver(14)
        base = solver._is_power()
        assert base is None

    def test_continued_fraction(self):
        """_continued_fraction 的基本测试。"""
        # 1/3 = 0.333... -> (1, 3)
        p, q = ShorSolver._continued_fraction(1.0 / 3.0, max_denom=100)
        assert p == 1 and q == 3, f"Expected (1, 3), got ({p}, {q})"

        # 3/7 ≈ 0.428571...
        p, q = ShorSolver._continued_fraction(3.0 / 7.0, max_denom=100)
        assert p == 3 and q == 7, f"Expected (3, 7), got ({p}, {q})"

    def test_find_period_for_known_case(self):
        """_find_period_classical 对已知情况应返回正确周期。"""
        # 对于 N=15, a=7: 7^1=7, 7^2=4, 7^3=13, 7^4=1 -> 周期 r=4
        solver = ShorSolver(15)
        r = solver._find_period_classical(7)
        assert r == 4, f"Expected period 4, got {r}"


# ============================================================================
#  TestQAOA
# ============================================================================

class TestQAOA:
    """测试 QAOA 类。"""

    @pytest.fixture
    def simple_cost_matrix_2q(self):
        """2 量子比特的简单 MaxCut 成本矩阵。"""
        # MaxCut 问题：最小化 Z0*Z1
        mat = np.zeros((2, 2))
        mat[0, 1] = mat[1, 0] = 1.0
        return mat

    @pytest.fixture
    def simple_cost_matrix_3q(self):
        """3 量子比特的简单三角形图成本矩阵。"""
        mat = np.zeros((3, 3))
        mat[0, 1] = mat[1, 0] = 1.0
        mat[1, 2] = mat[2, 1] = 1.0
        mat[0, 2] = mat[2, 0] = 1.0
        return mat

    # -- 优化测试 ---------------------------------------------------------

    def test_optimize_returns_params_and_energy(self, simple_cost_matrix_2q):
        """optimize 应返回 (params, energy)。"""
        qaoa = QAOA(n_qubits=2, cost_matrix=simple_cost_matrix_2q, p=1)
        params, energy = qaoa.optimize(maxiter=50)
        assert isinstance(params, np.ndarray)
        assert isinstance(energy, float)
        assert len(params) == 2  # p=1 → 2 个参数 (gamma, beta)

    def test_optimize_2qubit_problem(self, simple_cost_matrix_2q):
        """2 量子比特 QAOA 应能找到合理解。"""
        qaoa = QAOA(n_qubits=2, cost_matrix=simple_cost_matrix_2q, p=2)
        params, energy = qaoa.optimize(maxiter=80)
        # 能量应为有限值
        assert np.isfinite(energy)

    def test_optimize_3qubit_problem(self, simple_cost_matrix_3q):
        """3 量子比特 QAOA 应能运行。"""
        qaoa = QAOA(n_qubits=3, cost_matrix=simple_cost_matrix_3q, p=1)
        params, energy = qaoa.optimize(maxiter=50)
        assert np.isfinite(energy)

    # -- get_optimal_bitstring -------------------------------------------

    def test_get_optimal_bitstring_after_optimize(self, simple_cost_matrix_2q):
        """optimize 后 get_optimal_bitstring 应返回有效位串和成本。"""
        qaoa = QAOA(n_qubits=2, cost_matrix=simple_cost_matrix_2q, p=1)
        qaoa.optimize(maxiter=50)
        bitstring, cost = qaoa.get_optimal_bitstring()
        assert isinstance(bitstring, int)
        assert isinstance(cost, float)
        assert 0 <= bitstring < (1 << 2)
        assert np.isfinite(cost)

    def test_get_optimal_bitstring_before_optimize_raises(self, simple_cost_matrix_2q):
        """optimize 前调用 get_optimal_bitstring 应抛出 RuntimeError。"""
        qaoa = QAOA(n_qubits=2, cost_matrix=simple_cost_matrix_2q, p=1)
        with pytest.raises(RuntimeError, match="Must call optimize"):
            qaoa.get_optimal_bitstring()

    # -- sample_solutions -------------------------------------------------

    def test_sample_solutions(self, simple_cost_matrix_2q):
        """sample_solutions 应返回有效解列表。"""
        qaoa = QAOA(n_qubits=2, cost_matrix=simple_cost_matrix_2q, p=1)
        qaoa.optimize(maxiter=50)
        solutions = qaoa.sample_solutions(n_samples=4)
        assert len(solutions) <= 4
        for bitstring, cost in solutions:
            assert 0 <= bitstring < (1 << 2)
            assert np.isfinite(cost)

    def test_sample_solutions_before_optimize_raises(self, simple_cost_matrix_2q):
        """optimize 前调用 sample_solutions 应抛出 RuntimeError。"""
        qaoa = QAOA(n_qubits=2, cost_matrix=simple_cost_matrix_2q, p=1)
        with pytest.raises(RuntimeError, match="Must call optimize"):
            qaoa.sample_solutions()

    # -- 属性测试 ---------------------------------------------------------

    def test_optimal_energy_property(self, simple_cost_matrix_2q):
        """optimal_energy 属性应反映最后一次优化结果。"""
        qaoa = QAOA(n_qubits=2, cost_matrix=simple_cost_matrix_2q, p=1)
        assert qaoa.optimal_energy is None
        qaoa.optimize(maxiter=50)
        assert qaoa.optimal_energy is not None
        assert np.isfinite(qaoa.optimal_energy)

    def test_optimal_bitstring_property(self, simple_cost_matrix_2q):
        """optimal_bitstring 属性应反映最优位串。"""
        qaoa = QAOA(n_qubits=2, cost_matrix=simple_cost_matrix_2q, p=1)
        assert qaoa.optimal_bitstring is None
        qaoa.optimize(maxiter=50)
        assert qaoa.optimal_bitstring is not None
        assert 0 <= qaoa.optimal_bitstring < (1 << 2)

    # -- 错误处理 ---------------------------------------------------------

    def test_cost_matrix_wrong_shape_raises(self):
        """成本矩阵形状错误应抛出 ValueError。"""
        with pytest.raises(ValueError, match="cost_matrix must be"):
            QAOA(n_qubits=2, cost_matrix=np.zeros((3, 3)), p=1)

    def test_p_less_than_1_raises(self):
        """p < 1 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="p must be >= 1"):
            QAOA(n_qubits=2, cost_matrix=np.zeros((2, 2)), p=0)

    def test_invalid_cost_matrix_type_raises(self):
        """非 ndarray 类型的 cost_matrix 应抛出错误。"""
        with pytest.raises(AttributeError):
            QAOA(n_qubits=2, cost_matrix=[[0, 1], [1, 0]], p=1)
        # 注意：list 没有 .shape 属性，会在 __init__ 中触发 AttributeError

    # -- MaxCut 辅助 ------------------------------------------------------

    def test_maxcut_cost_matrix(self):
        """maxcut_cost_matrix 应返回传入的邻接矩阵。"""
        adj = np.array([[0, 1], [1, 0]])
        result = QAOA.maxcut_cost_matrix(adj)
        np.testing.assert_array_equal(result, adj)

    # -- repr -------------------------------------------------------------

    def test_repr(self, simple_cost_matrix_2q):
        """__repr__ 应包含 n_qubits 和 p。"""
        qaoa = QAOA(n_qubits=2, cost_matrix=simple_cost_matrix_2q, p=3)
        r = repr(qaoa)
        assert "2" in r
        assert "3" in r


# ============================================================================
#  TestVQE
# ============================================================================

class TestVQE:
    """测试 VQE 类。"""

    @pytest.fixture
    def h2_terms(self):
        """H₂ 哈密顿量的 Pauli 项。"""
        return VQE.h2_hamiltonian()

    # -- 优化测试 ---------------------------------------------------------

    def test_optimize_h2_returns_energy_and_state(self, h2_terms):
        """optimize 应返回 (energy, state)。"""
        vqe = VQE(n_qubits=4, hamiltonian_pauli_terms=h2_terms, ansatz_type="he", n_layers=1)
        energy, state = vqe.optimize(maxiter=100)
        assert isinstance(energy, float)
        assert isinstance(state, np.ndarray)
        assert len(state) == 16  # 2^4

    def test_optimize_h2_energy_is_negative(self, h2_terms):
        """H₂ 的基态能量应为负值。"""
        vqe = VQE(n_qubits=4, hamiltonian_pauli_terms=h2_terms, ansatz_type="he", n_layers=1)
        energy, _ = vqe.optimize(maxiter=100)
        assert energy < 0.0, f"H₂ ground state energy should be negative, got {energy}"

    def test_optimize_h2_energy_reasonable(self, h2_terms):
        """H₂ 基态能量应在合理范围内（>-2.0，对于 STO-3G）。"""
        vqe = VQE(n_qubits=4, hamiltonian_pauli_terms=h2_terms, ansatz_type="he", n_layers=1)
        energy, _ = vqe.optimize(maxiter=100)
        assert -2.0 < energy < 0.0, f"Energy {energy} out of expected range"

    # -- ground_state 属性 ------------------------------------------------

    def test_ground_state_property(self, h2_terms):
        """ground_state 属性应在 optimize 后返回有效状态向量。"""
        vqe = VQE(n_qubits=4, hamiltonian_pauli_terms=h2_terms, n_layers=1)
        vqe.optimize(maxiter=100)
        gs = vqe.ground_state
        assert isinstance(gs, np.ndarray)
        # 应为归一化向量
        norm = np.sqrt(np.sum(np.abs(gs) ** 2))
        assert abs(norm - 1.0) < 1e-10

    def test_ground_state_before_optimize_raises(self, h2_terms):
        """optimize 前访问 ground_state 应抛出 RuntimeError。"""
        vqe = VQE(n_qubits=4, hamiltonian_pauli_terms=h2_terms)
        with pytest.raises(RuntimeError, match="Must call optimize"):
            _ = vqe.ground_state

    # -- ground_energy / optimal_params 属性 ------------------------------

    def test_ground_energy_property(self, h2_terms):
        """ground_energy 属性应在 optimize 后可用。"""
        vqe = VQE(n_qubits=4, hamiltonian_pauli_terms=h2_terms, n_layers=1)
        assert vqe.ground_energy is None
        vqe.optimize(maxiter=100)
        assert vqe.ground_energy is not None
        assert np.isfinite(vqe.ground_energy)

    def test_optimal_params_property(self, h2_terms):
        """optimal_params 属性应在 optimize 后可用。"""
        vqe = VQE(n_qubits=4, hamiltonian_pauli_terms=h2_terms, n_layers=1)
        assert vqe.optimal_params is None
        vqe.optimize(maxiter=100)
        assert vqe.optimal_params is not None
        assert len(vqe.optimal_params) > 0

    # -- get_expectations -------------------------------------------------

    def test_get_expectations(self, h2_terms):
        """get_expectations 应对每个 Pauli 项返回期望值。"""
        vqe = VQE(n_qubits=4, hamiltonian_pauli_terms=h2_terms, n_layers=1)
        vqe.optimize(maxiter=100)
        exps = vqe.get_expectations()
        assert isinstance(exps, dict)
        # 应包含所有 Pauli 项
        for _, pauli in h2_terms:
            assert pauli in exps
        # 返回值应包含 (coefficient, expectation)
        for pauli, (coeff, exp_val) in exps.items():
            assert isinstance(coeff, float)
            assert isinstance(exp_val, float)

    def test_get_expectations_before_optimize_raises(self, h2_terms):
        """optimize 前调用 get_expectations 应抛出 RuntimeError。"""
        vqe = VQE(n_qubits=4, hamiltonian_pauli_terms=h2_terms)
        with pytest.raises(RuntimeError, match="Must call optimize"):
            vqe.get_expectations()

    # -- h2_hamiltonian 静态方法 ------------------------------------------

    def test_h2_hamiltonian_returns_valid_terms(self):
        """h2_hamiltonian 应返回有效的 (系数, Pauli 串) 元组列表。"""
        terms = VQE.h2_hamiltonian()
        assert isinstance(terms, list)
        assert len(terms) > 0
        for term in terms:
            assert isinstance(term, tuple)
            assert len(term) == 2
            coeff, pauli = term
            assert isinstance(coeff, float)
            assert isinstance(pauli, str)
            assert len(pauli) == 4  # H₂ 使用 4 量子比特
            for c in pauli:
                assert c in "IXYZ"

    def test_h2_hamiltonian_with_bond_length(self):
        """h2_hamiltonian 应接受 bond_length 参数。"""
        terms = VQE.h2_hamiltonian(bond_length=1.0)
        assert isinstance(terms, list)

    # -- 错误处理 ---------------------------------------------------------

    def test_invalid_n_qubits_raises(self):
        """n_qubits < 1 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="n_qubits must be >= 1"):
            VQE(n_qubits=0, hamiltonian_pauli_terms=[(1.0, "Z")])

    def test_empty_hamiltonian_raises(self):
        """空哈密顿量列表应抛出 ValueError。"""
        with pytest.raises(ValueError, match="hamiltonian_pauli_terms must be a non-empty list"):
            VQE(n_qubits=2, hamiltonian_pauli_terms=[])

    def test_invalid_pauli_characters_raise(self):
        """包含无效字符的 Pauli 串应抛出 ValueError。"""
        with pytest.raises(ValueError, match="Invalid Pauli character"):
            VQE(n_qubits=2, hamiltonian_pauli_terms=[(-1.0, "AW")])

    def test_pauli_length_mismatch_raises(self):
        """Pauli 串长度与 n_qubits 不匹配时应抛出 ValueError。"""
        with pytest.raises(ValueError, match="has length"):
            VQE(n_qubits=3, hamiltonian_pauli_terms=[(-1.0, "ZZ")])

    def test_invalid_ansatz_type_raises(self):
        """无效的 ansatz_type 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="Unknown ansatz_type"):
            VQE(n_qubits=2, hamiltonian_pauli_terms=[(-1.0, "ZZ")], ansatz_type="invalid")

    def test_invalid_n_layers_raises(self):
        """n_layers < 1 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="n_layers must be >= 1"):
            VQE(n_qubits=2, hamiltonian_pauli_terms=[(-1.0, "ZZ")], n_layers=0)

    def test_non_numeric_coefficient_raises(self):
        """非数值系数应抛出 TypeError。"""
        with pytest.raises(TypeError, match="coefficient must be numeric"):
            VQE(n_qubits=2, hamiltonian_pauli_terms=[("bad", "ZZ")])

    def test_non_string_pauli_raises(self):
        """非字符串 Pauli 串应抛出 TypeError。"""
        with pytest.raises(TypeError, match="Pauli string must be str"):
            VQE(n_qubits=2, hamiltonian_pauli_terms=[(-1.0, 123)])

    # -- UCCSD ansatz -----------------------------------------------------

    def test_uccsd_ansatz_runs(self):
        """UCCSD ansatz 应能用于简单哈密顿量。"""
        # 2 量子比特的简单哈密顿量：H = -Z0
        vqe = VQE(
            n_qubits=2,
            hamiltonian_pauli_terms=[(-1.0, "ZI"), (0.0, "IZ")],
            ansatz_type="uccsd",
            n_layers=1
        )
        energy, state = vqe.optimize(maxiter=80)
        assert np.isfinite(energy)

    def test_wrong_params_length_in_ansatz(self, h2_terms):
        """向 _ansatz 传入错误长度的参数应抛出 ValueError。"""
        vqe = VQE(n_qubits=4, hamiltonian_pauli_terms=h2_terms, n_layers=1)
        wrong_len = vqe._num_params() + 1
        with pytest.raises(ValueError, match="params has length"):
            vqe._ansatz(np.zeros(wrong_len))

    # -- repr -------------------------------------------------------------

    def test_repr(self, h2_terms):
        """__repr__ 应包含关键信息。"""
        vqe = VQE(n_qubits=4, hamiltonian_pauli_terms=h2_terms, ansatz_type="he", n_layers=1)
        r = repr(vqe)
        assert "4" in r
        assert "he" in r
        assert "15" in r  # H₂ 有 15 项

    # -- 简单哈密顿量 ------------------------------------------------------

    def test_single_qubit_vqe(self):
        """单量子比特 VQE 应能工作。"""
        # H = -Z → 基态为 |0⟩，能量为 -1
        vqe = VQE(n_qubits=1, hamiltonian_pauli_terms=[(-1.0, "Z")], n_layers=1)
        energy, state = vqe.optimize(maxiter=100)
        # 能量应接近 -1
        assert energy < 0.0, f"Expected negative energy, got {energy}"

    def test_apply_pauli_string_identity(self):
        """_apply_pauli_string 对 'I' 应返回原状态。"""
        vqe = VQE(n_qubits=2, hamiltonian_pauli_terms=[(-1.0, "ZZ")])
        state = np.array([1, 0, 0, 0], dtype=complex)
        result = vqe._apply_pauli_string(state.copy(), "II")
        np.testing.assert_array_equal(result, state)

    def test_apply_pauli_string_x(self):
        """_apply_pauli_string 对 'XI' 应翻转量子比特 0 (LSB)。"""
        vqe = VQE(n_qubits=2, hamiltonian_pauli_terms=[(-1.0, "ZZ")])
        state = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
        result = vqe._apply_pauli_string(state.copy(), "XI")
        # |00⟩ → |01⟩ (比特 0 翻转，索引 1)
        expected = np.array([0, 1, 0, 0], dtype=complex)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_expectation_pauli_for_z_on_zero_state(self):
        """⟨00|Z_0⊗I|00⟩ 应等于 +1。"""
        vqe = VQE(n_qubits=2, hamiltonian_pauli_terms=[(-1.0, "ZZ")])
        state = np.array([1, 0, 0, 0], dtype=complex)
        exp = vqe._expectation_pauli(state, "ZI")
        assert abs(exp - 1.0) < 1e-12

    def test_energy_for_known_state(self):
        """已知状态的能量应能正确计算。"""
        vqe = VQE(n_qubits=2, hamiltonian_pauli_terms=[(-1.0, "ZZ")])
        # |00⟩: ⟨ZZ⟩ = +1, E = -1
        state = np.array([1, 0, 0, 0], dtype=complex)
        energy = vqe._energy(state)
        assert abs(energy + 1.0) < 1e-12
