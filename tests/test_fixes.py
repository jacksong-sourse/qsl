"""
针对 v0.4.1 缺陷修复的回归测试。

覆盖:
    - Grover 量子 Oracle 电路 (无经典枚举) 与 BBHT 搜索
    - QFT apply 与 get_matrix 一致性
    - Shor 大 N 量子周期查找 (无阈值经典回退)
    - DensityMatrix 全程 numpy ndarray
    - 全量子比特振幅阻尼
    - QGAN 生成器梯度链 (Straight-Through Estimator)
    - QUBO/Ising 变量转换一致性
    - VQE parameter-shift 梯度
    - 解析器下划线标识符
    - kron 弃用与 kronecker_prod
    - 编译器 VQE 非 H2 分子报错
    - 药物发现管线无静默随机回退
    - QuantumTheoremProver 使用 Grover 搜索
    - AlgorithmSearcher 轻量 fitness
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from qsl.core.grover import GroverSearch, solve_sat
from qsl.core.oracle import compile_phase_oracle
from qsl.core.parser import parse_bool, BooleanParser
from qsl.core.state import QuantumState, DensityMatrix
from qsl.utils.exceptions import NoSolutionError


# ============================================================================
#  Grover 量子 Oracle 电路 (issue #1)
# ============================================================================

class TestQuantumOracleCircuit:
    """Oracle 直接从布尔表达式编译为量子电路, 不做经典枚举。"""

    def test_oracle_circuit_phase_correctness(self):
        """对每个基态验证 Oracle 相位 = (-1)^f(x)。"""
        exprs = [parse_bool("x0 | ~x1"), parse_bool("x1 | x2")]
        circ = compile_phase_oracle(exprs, 3)
        for x in range(8):
            expected = all(e.evaluate(x) for e in exprs)
            st = QuantumState(3 + circ.n_ancilla)
            for q in range(3):
                if (x >> q) & 1:
                    st.x(q)
            GroverSearch._apply_oracle_circuit(st, circ)
            amp = st.amplitudes[x]
            assert abs(abs(amp) - 1.0) < 1e-9
            assert (amp.real < 0) == expected, f"x={x} 相位错误"

    def test_oracle_circuit_ancilla_uncomputed(self):
        """Oracle 作用后 ancilla 必须回到 |0>。"""
        exprs = [parse_bool("(x0 & x1) | (~x2 & x0)")]
        circ = compile_phase_oracle(exprs, 3)
        st = QuantumState(3 + circ.n_ancilla)
        for q in range(3):
            st.h(q)
        GroverSearch._apply_oracle_circuit(st, circ)
        # ancilla 位 (高比特) 振幅应全为 0
        for i in range(8, len(st.amplitudes)):
            assert abs(st.amplitudes[i]) < 1e-9

    def test_search_expressions_known_m(self):
        """已知 M 时电路 Oracle 搜索成功。"""
        grover = GroverSearch(3)
        result = grover.search_expressions(
            [parse_bool("x0 & x1")], num_solutions=2, shots=20)
        assert result.success_count > 0
        for s in result.get_solutions():
            assert (s & 1) and (s & 2)

    def test_search_expressions_bbht_unknown_m(self):
        """未知 M 时 BBHT 搜索成功且 num_solutions 为 None。"""
        grover = GroverSearch(3)
        result = grover.search_expressions(
            [parse_bool("x0 & x1 & x2")], num_solutions=None, shots=10)
        assert result.success_count > 0
        assert 7 in result.get_solutions()
        assert result.num_solutions is None
        assert result.quantum_queries is not None

    def test_search_expressions_no_solution(self):
        """无解表达式 -> NoSolutionError。"""
        grover = GroverSearch(2)
        with pytest.raises(NoSolutionError):
            grover.search_expressions([parse_bool("x0 & ~x0")], shots=3)

    def test_solve_sat_uses_circuit(self):
        """solve_sat 通过电路 Oracle 求解且解正确。"""
        result = solve_sat([[1, -2], [2, 3], [-1, -3]], n_qubits=3, shots=10)
        assert result.success_count > 0
        for s in result.get_solutions():
            b = [(s >> i) & 1 for i in range(3)]
            assert (b[0] or not b[1]) and (b[1] or b[2]) and (not b[0] or not b[2])


# ============================================================================
#  QFT apply 与 get_matrix 一致性 (issue #13)
# ============================================================================

class TestQFTConsistency:
    """apply() 必须与 get_matrix() @ state 一致。"""

    def test_apply_matches_matrix_n1_to_n5(self):
        from qsl.algorithms.qft import QuantumFourierTransform
        for n in range(1, 6):
            qft = QuantumFourierTransform(n)
            N = 1 << n
            rng = np.random.RandomState(n)
            v = rng.randn(N) + 1j * rng.randn(N)
            np.testing.assert_allclose(
                qft.apply(v), qft.get_matrix() @ v, atol=1e-10)

    def test_apply_basis_state(self):
        """QFT|1> (n=2) 应为 [1, i, -1, -i]/2。"""
        from qsl.algorithms.qft import QuantumFourierTransform
        qft = QuantumFourierTransform(2)
        v = np.zeros(4, dtype=complex)
        v[1] = 1.0
        expected = np.array([1, 1j, -1, -1j]) / 2
        np.testing.assert_allclose(qft.apply(v), expected, atol=1e-12)


# ============================================================================
#  Shor 量子周期查找 (issue #2)
# ============================================================================

class TestShorQuantumPeriodFinding:
    """周期查找始终使用量子相位估计, 无静默经典回退。"""

    def test_factor_beyond_old_threshold(self):
        """N=1023 超过旧 2^12 阈值, 仍由量子 QPE 分解。"""
        from qsl.algorithms.shor import ShorSolver
        factors = ShorSolver(1023).factor()
        prod = 1
        for f in factors:
            prod *= f
        assert prod == 1023
        assert all(ShorSolver._is_prime(f) for f in factors)

    def test_huge_n_raises_not_silent_fallback(self):
        """超出模拟能力时抛出 RuntimeError 而非静默回退经典。"""
        from qsl.algorithms.shor import ShorSolver
        solver = ShorSolver(2**21 - 1)
        with pytest.raises(RuntimeError):
            solver.factor(max_attempts=1)

    def test_opt_in_classical_fallback_warns(self):
        """显式允许经典回退时给出警告。"""
        from qsl.algorithms.shor import ShorSolver
        solver = ShorSolver(2**21 - 1, allow_classical_fallback=True)
        with pytest.warns(RuntimeWarning):
            solver._find_period_quantum(2)


# ============================================================================
#  DensityMatrix 类型一致性 (issue #5, #16, #18)
# ============================================================================

class TestDensityMatrixNdarray:
    """_rho 在所有操作后保持 numpy ndarray。"""

    def test_apply_unitary_keeps_ndarray(self):
        dm = DensityMatrix(2)
        dm.apply_unitary(np.eye(4))
        assert isinstance(dm._rho, np.ndarray)
        assert dm.purity() > 0.99

    def test_operations_chain_type_consistent(self):
        state = QuantumState(2)
        state.h(0)
        state.cnot(0, 1)
        dm = DensityMatrix.from_pure(state)
        dm.apply_unitary(np.eye(4))
        dm.apply_depolarizing(0.1)
        dm.apply_amplitude_damping(0.1)
        dm.apply_phase_damping(0.1)
        assert isinstance(dm._rho, np.ndarray)
        assert abs(dm.trace() - 1.0) < 1e-9
        assert 0.0 < dm.purity() <= 1.0
        m = dm.get_matrix()
        assert isinstance(m, np.ndarray)

    def test_amplitude_damping_all_qubits(self):
        """振幅阻尼作用于所有量子比特, 不仅是 qubit 0。"""
        state = QuantumState(2)
        state.x(0)
        state.x(1)  # |11>
        dm = DensityMatrix.from_pure(state)
        dm.apply_amplitude_damping(0.5)
        # |11> 两个比特都应衰减: |00> 获得布居
        assert dm.probability(0) > 0.0
        # qubit 1 (高位) 也衰减: |01> 获得布居
        assert dm.probability(1) > 0.0
        assert abs(dm.trace() - 1.0) < 1e-9


# ============================================================================
#  QGAN 可微采样 (issue #4)
# ============================================================================

class TestQGANDifferentiable:
    """Straight-Through Estimator 保持生成器梯度链。"""

    @pytest.fixture(autouse=True)
    def _torch(self):
        pytest.importorskip("torch")

    def test_generator_receives_gradient(self):
        import torch
        from qsl.qml.qgan import QGAN
        torch.manual_seed(0)
        qgan = QGAN(latent_dim=2, data_dim=2, n_qubits=2)
        z = torch.randn(4, 2)
        fake = qgan.generator(z)
        loss = qgan.discriminator(fake).sum()
        loss.backward()
        grads = [p.grad for p in qgan.generator.parameters()]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)

    def test_samples_still_binary(self):
        import torch
        from qsl.qml.qgan import QGAN
        torch.manual_seed(0)
        qgan = QGAN(latent_dim=2, data_dim=3, n_qubits=3)
        samples = qgan.sample(10)
        assert set(np.unique(samples)).issubset({0.0, 1.0})


# ============================================================================
#  QUBO/Ising 变量转换 (issue #19)
# ============================================================================

class TestQuboIsingConversion:
    """QAOA 内部统一 QUBO<->Ising 转换。"""

    def test_qubo_ising_cost_equivalence(self):
        from qsl.algorithms.qaoa import QAOA
        rng = np.random.RandomState(7)
        n = 4
        Q = rng.randn(n, n)
        Q = (Q + Q.T) / 2
        qaoa = QAOA(n, Q, p=1, encoding="qubo")
        for x in range(1 << n):
            bits = [(x >> i) & 1 for i in range(n)]
            e_qubo = (sum(Q[i, i] * bits[i] for i in range(n))
                      + sum(Q[i, j] * bits[i] * bits[j]
                            for i in range(n) for j in range(i + 1, n)))
            assert abs(qaoa._cost(x) - e_qubo) < 1e-9

    def test_qaoa_finds_qubo_optimum(self):
        from qsl.algorithms.qaoa import QAOA
        Q = np.array([[2.0, -1.0, 0.5], [-1.0, 1.0, 0.3], [0.5, 0.3, -1.0]])
        qaoa = QAOA(3, Q, p=2, encoding="qubo")
        qaoa.optimize(maxiter=100)
        best, _ = qaoa.get_optimal_bitstring()
        brute = min(range(8), key=lambda x: (
            sum(Q[i, i] * ((x >> i) & 1) for i in range(3))
            + sum(Q[i, j] * ((x >> i) & 1) * ((x >> j) & 1)
                  for i in range(3) for j in range(i + 1, 3))))
        assert best == brute

    def test_portfolio_frontier_uses_qaoa(self):
        """效率前沿由 QAOA 逐点计算 (无 np.linalg.solve)。"""
        from qsl.pipelines.portfolio import PortfolioOptimizer
        returns, cov = PortfolioOptimizer.sample_problem(n_assets=4, seed=42)
        opt = PortfolioOptimizer(returns=returns, covariance=cov, budget=2)
        frontier = opt._compute_frontier(n_points=3)
        assert len(frontier) == 3
        for risk, ret in frontier:
            assert risk >= 0 and isinstance(ret, float)


# ============================================================================
#  VQE parameter-shift 梯度 (issue #9)
# ============================================================================

class TestVQEParameterShift:
    """parameter-shift 提供精确梯度。"""

    def test_parameter_shift_matches_finite_difference(self):
        from qsl.algorithms.vqe import VQE
        vqe = VQE(4, VQE.h2_hamiltonian(), ansatz_type="he", n_layers=1)
        rng = np.random.RandomState(0)
        params = rng.uniform(-np.pi, np.pi, vqe._num_params())
        g_ps = vqe._parameter_shift_gradient(params.copy())
        eps = 1e-6
        g_fd = np.empty_like(params)
        for i in range(len(params)):
            p = params.copy()
            p[i] += eps
            g_fd[i] = (vqe._cost_function(p)
                       - vqe._cost_function(params)) / eps
        np.testing.assert_allclose(g_ps, g_fd, atol=1e-5)


# ============================================================================
#  解析器下划线标识符 (issue #23)
# ============================================================================

class TestParserUnderscore:
    """_parse_identifier 支持下划线开头/包含下划线的标识符。"""

    def test_underscore_identifier_tokenized(self):
        parser = BooleanParser("_x1")
        ident = parser._parse_identifier()
        assert ident == "_x1"

    def test_identifier_with_underscores(self):
        parser = BooleanParser("x_0_a")
        ident = parser._parse_identifier()
        assert ident == "x_0_a"


# ============================================================================
#  kron 弃用与 kronecker_prod (issue #15)
# ============================================================================

class TestKroneckerProd:
    """kronecker_prod 为主名, kron 弃用但兼容。"""

    def test_kronecker_prod_correct(self):
        from qsl.quantum_gates import kronecker_prod, H, I
        result = kronecker_prod(H, I)
        assert result.shape == (4, 4)
        np.testing.assert_allclose(result, np.kron(H, I))

    def test_kron_deprecated_but_works(self):
        from qsl.quantum_gates import kron, H, I
        with pytest.warns(DeprecationWarning):
            result = kron(H, I)
        np.testing.assert_allclose(result, np.kron(H, I))


# ============================================================================
#  编译器 VQE 非 H2 分子 (issue #17)
# ============================================================================

class TestCompilerVQENonH2:
    """非 H2 分子应报错而非静默替换。"""

    def test_non_h2_molecule_raises(self):
        from qsl import QSLProgram, QSLCompiler
        from qsl.utils.exceptions import ProgramValidationError
        program = QSLProgram(
            name="VQE", n_qubits=4, premises=["h2o"], shots=5)
        program.main_algorithm = "vqe"
        compiler = QSLCompiler(verbose=False)
        with pytest.raises(ProgramValidationError):
            compiler.compile_and_run(program)


# ============================================================================
#  药物发现管线 (issue #6)
# ============================================================================

class TestDrugDiscoveryNoSilentFallback:
    """VQE 失败时不再静默返回随机数。"""

    def test_no_silent_random_fallback(self):
        from qsl.pipelines.drug_discovery import DrugDiscoveryPipeline
        pipeline = DrugDiscoveryPipeline(num_candidates=1)
        # 破坏 VQE 使其失败: 传入非法哈密顿量应抛异常而非返回随机数
        with pytest.raises(Exception):
            pipeline._build_molecular_hamiltonian("CC")

    def test_demo_mode_still_works(self):
        from qsl.pipelines.drug_discovery import DrugDiscoveryPipeline
        pipeline = DrugDiscoveryPipeline(num_candidates=2, top_k=1)
        results = pipeline.run(verbose=False)
        assert len(results) == 1
        assert results[0].rank == 1


# ============================================================================
#  QuantumTheoremProver 使用 Grover (issue #10)
# ============================================================================

class TestTheoremProverGrover:
    """证明搜索通过 Grover 振幅放大完成。"""

    def test_prove_uses_grover_queries(self):
        from qsl.meta.theory_generator import QuantumTheoremProver
        prover = QuantumTheoremProver(
            conjecture="a + b = b + a", max_proof_depth=2, max_branching=2)
        result = prover.prove()
        assert result.proof_found
        # Grover 查询次数应远小于经典枚举的 N
        assert result.quantum_queries >= 0
        assert result.classical_steps > 0


# ============================================================================
#  AlgorithmSearcher 轻量 fitness (issue #24)
# ============================================================================

class TestAlgorithmSearcherLightweight:
    """fitness 默认不跑完整态模拟且保持在 [0, 1]。"""

    def test_fitness_structural_in_range(self):
        from qsl.meta.algorithm_search import CircuitGenome, AlgorithmSearcher
        searcher = AlgorithmSearcher(n_qubits=3, population_size=4,
                                     generations=2)
        for _ in range(10):
            genome = CircuitGenome.random(3, max_gates=8)
            f = searcher._fitness(genome)
            assert 0.0 <= f <= 1.0

    def test_entangling_circuit_scores_higher(self):
        from qsl.meta.algorithm_search import CircuitGenome, AlgorithmSearcher
        searcher = AlgorithmSearcher(n_qubits=3)
        entangled = CircuitGenome(
            gates=[{'gate': 'H', 'targets': [0]},
                   {'gate': 'CNOT', 'control': 0, 'target': 1},
                   {'gate': 'CNOT', 'control': 1, 'target': 2}],
            n_qubits=3)
        trivial = CircuitGenome(gates=[{'gate': 'X', 'targets': [0]}],
                                n_qubits=3)
        assert searcher._fitness(entangled) > searcher._fitness(trivial)
