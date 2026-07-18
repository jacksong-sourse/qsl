"""
测试量子态向量 (QuantumState) 模块。

覆盖:
    - 初始化和验证
    - 单量子比特门 (X, Y, Z, H, S, T)
    - 两量子比特门 (CNOT, CZ, SWAP)
    - 三量子比特门 (Toffoli)
    - 多量子比特门 (MCZ)
    - Oracle 和扩散算子
    - 测量和采样
    - 归一化检查
    - 克隆和属性访问
    - 边界条件和错误处理
"""

import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsl.core.state import QuantumState, DensityMatrix, MAX_QUBITS
from qsl.utils.exceptions import (
    InvalidQubitCountError,
    QubitIndexError,
    DuplicateQubitError,
    StateNormalizationError,
)


class TestQuantumStateInit:
    """测试量子态初始化。"""

    def test_init_default(self):
        """默认初始态: |0...0>。"""
        for n in [1, 2, 3, 5, 10]:
            state = QuantumState(n)
            assert state.n_qubits == n
            assert state.size == (1 << n)
            assert abs(state.amplitudes[0] - 1.0) < 1e-10
            for i in range(1, 1 << n):
                assert state.amplitudes[i] == 0j

    def test_init_zero_qubits(self):
        """0 量子比特 -> 异常。"""
        try:
            QuantumState(0)
            assert False, "应该抛出异常"
        except (InvalidQubitCountError, ValueError):
            pass

    def test_init_negative_qubits(self):
        """负数量子比特 -> 异常。"""
        try:
            QuantumState(-1)
            assert False, "应该抛出异常"
        except (InvalidQubitCountError, ValueError):
            pass

    def test_init_too_many_qubits(self):
        """超过 MAX_QUBITS -> 异常。"""
        try:
            QuantumState(MAX_QUBITS + 1)
            assert False, "应该抛出异常"
        except (InvalidQubitCountError, ValueError):
            pass

    def test_init_normalization(self):
        """初始态归一化检查。"""
        state = QuantumState(4)
        assert state.check_normalization()


class TestSingleQubitGates:
    """测试单量子比特门。"""

    def test_x_gate(self):
        """X 门: X|0> = |1>。"""
        state = QuantumState(1)
        state.x(0)
        assert abs(state.amplitudes[0] - 0.0) < 1e-10
        assert abs(state.amplitudes[1] - 1.0) < 1e-10

    def test_x_gate_double(self):
        """X 门两次: 回到原状态。"""
        state = QuantumState(2)
        state.x(0)
        state.x(0)
        assert abs(state.amplitudes[0] - 1.0) < 1e-10

    def test_x_gate_multi_qubit(self):
        """X 门在多量子比特系统。"""
        state = QuantumState(3)
        state.x(1)  # 翻转第1位 -> |010>
        prob = state.probability(2)  # 010 = 2
        assert abs(prob - 1.0) < 1e-10

    def test_x_gate_out_of_range(self):
        """X 门索引越界。"""
        state = QuantumState(3)
        try:
            state.x(5)
            assert False
        except QubitIndexError:
            pass

    def test_h_gate(self):
        """H 门: H|0> = (|0>+|1>)/sqrt(2)。"""
        state = QuantumState(1)
        state.h(0)
        expected = 1.0 / math.sqrt(2)
        assert abs(abs(state.amplitudes[0]) - expected) < 1e-10
        assert abs(abs(state.amplitudes[1]) - expected) < 1e-10

    def test_h_gate_double(self):
        """H 门两次: HHH|0> = |0>。"""
        state = QuantumState(1)
        state.h(0)
        state.h(0)
        assert abs(state.amplitudes[0] - 1.0) < 1e-10

    def test_h_gate_all_qubits(self):
        """所有量子比特 H 门: 均匀叠加。"""
        n = 4
        state = QuantumState(n)
        for q in range(n):
            state.h(q)
        expected = 1.0 / math.sqrt(1 << n)
        for i in range(1 << n):
            assert abs(abs(state.amplitudes[i]) - expected) < 1e-10

    def test_z_gate(self):
        """Z 门: Z|1> = -|1>。"""
        state = QuantumState(1)
        state.x(0)  # |1>
        state.z(0)
        assert state.amplitudes[1].real < 0

    def test_y_gate(self):
        """Y 门: Y|0> = i|1>。"""
        state = QuantumState(1)
        state.y(0)
        assert abs(state.amplitudes[0]) < 1e-10
        assert abs(state.amplitudes[1] - 1j) < 1e-10

    def test_s_gate(self):
        """S 门: S|1> = i|1>。"""
        state = QuantumState(1)
        state.x(0)
        state.s(0)
        assert abs(state.amplitudes[1] - 1j) < 1e-10

    def test_t_gate(self):
        """T 门存在，无异常。"""
        state = QuantumState(1)
        state.t(0)
        assert state.check_normalization()


class TestTwoQubitGates:
    """测试两量子比特门。"""

    def test_cnot_00(self):
        """CNOT|00> = |00>。"""
        state = QuantumState(2)
        state.cnot(0, 1)
        assert abs(state.amplitudes[0] - 1.0) < 1e-10

    def test_cnot_10(self):
        """CNOT|10> = |11>。"""
        state = QuantumState(2)
        state.x(0)  # |10>
        state.cnot(0, 1)
        assert abs(state.amplitudes[3] - 1.0) < 1e-10  # |11> = 3

    def test_cnot_same_qubit(self):
        """CNOT control==target -> 异常。"""
        state = QuantumState(2)
        try:
            state.cnot(0, 0)
            assert False
        except ValueError:
            pass

    def test_cnot_out_of_range(self):
        """CNOT 索引越界。"""
        state = QuantumState(2)
        try:
            state.cnot(0, 3)
            assert False
        except QubitIndexError:
            pass

    def test_cz(self):
        """CZ|11> = -|11>。"""
        state = QuantumState(2)
        state.x(0)
        state.x(1)  # |11>
        state.cz(0, 1)
        assert state.amplitudes[3].real < 0

    def test_swap(self):
        """SWAP|01> = |10>。"""
        state = QuantumState(2)
        state.x(1)  # |01> = 2
        state.swap(0, 1)  # |10> = 1
        assert abs(state.amplitudes[1] - 1.0) < 1e-10

    def test_swap_same_qubit(self):
        """SWAP(q, q) 是恒等操作。"""
        state = QuantumState(3)
        state.x(1)
        state.swap(1, 1)
        assert abs(state.amplitudes[2] - 1.0) < 1e-10


class TestThreeQubitGates:
    """测试三量子比特门。"""

    def test_toffoli_110(self):
        """Toffoli|110> = |111>。"""
        state = QuantumState(3)
        state.x(0)
        state.x(1)  # |110>
        state.toffoli(0, 1, 2)
        assert abs(state.amplitudes[7] - 1.0) < 1e-10  # |111> = 7

    def test_toffoli_010(self):
        """Toffoli|010> 不变 (控制不全为1)。"""
        state = QuantumState(3)
        state.x(1)  # |010>
        state.toffoli(0, 1, 2)
        assert abs(state.amplitudes[2] - 1.0) < 1e-10  # 不变

    def test_toffoli_duplicate(self):
        """Toffoli 重复 qubit -> 异常。"""
        state = QuantumState(3)
        try:
            state.toffoli(0, 0, 1)
            assert False
        except ValueError:
            pass


class TestMultiQubitGates:
    """测试多量子比特门。"""

    def test_mcz_all_one(self):
        """MCZ|111> = -|111>。"""
        state = QuantumState(3)
        for q in range(3):
            state.x(q)
        state.mcz([0, 1, 2])
        assert state.amplitudes[7].real < 0

    def test_mcz_not_all_one(self):
        """MCZ|101> 不变。"""
        state = QuantumState(3)
        state.x(0)
        state.x(2)  # |101>
        state.mcz([0, 1, 2])
        assert state.amplitudes[5].real > 0

    def test_mcz_empty_list(self):
        """MCZ 空列表 -> 异常。"""
        state = QuantumState(3)
        try:
            state.mcz([])
            assert False
        except ValueError:
            pass


class TestOracleAndDiffusion:
    """测试 Oracle 和扩散算子。"""

    def test_phase_oracle_single(self):
        """单解相位 Oracle。"""
        state = QuantumState(3)
        for q in range(3):
            state.h(q)
        state.phase_oracle({5})
        assert state.amplitudes[5].real < 0

    def test_phase_oracle_empty(self):
        """空标记集 Oracle 无操作。"""
        state = QuantumState(3)
        amps_before = state.amplitudes.copy()
        state.phase_oracle(set())
        for i in range(8):
            assert state.amplitudes[i] == amps_before[i]

    def test_diffusion_maintains_normalization(self):
        """扩散算子保持归一化。"""
        n = 4
        state = QuantumState(n)
        for q in range(n):
            state.h(q)
        state.diffusion_operator()
        assert state.check_normalization()


class TestMeasurement:
    """测试量子测量。"""

    def test_measure_deterministic(self):
        """确定性测量: |0> 态总是测到 0。"""
        state = QuantumState(3)
        for _ in range(10):
            result, prob = state.measure()
            assert result == 0
            assert abs(prob - 1.0) < 1e-10

    def test_measure_uniform(self):
        """均匀叠加态测量覆盖所有可能。"""
        n = 3
        state = QuantumState(n)
        for q in range(n):
            state.h(q)
        results = set()
        for _ in range(100):
            r, _ = state.measure()
            results.add(r)
        # n=3 时 100 次测量应该覆盖大部分状态
        assert len(results) >= 4

    def test_measure_most_likely(self):
        """measure_most_likely 返回概率最大态。"""
        state = QuantumState(3)
        state.x(0)
        state.x(2)  # |101> = 5
        best, prob = state.measure_most_likely()
        assert best == 5
        assert abs(prob - 1.0) < 1e-10

    def test_sample_multiple(self):
        """多次采样。"""
        state = QuantumState(3)
        results = state.sample(5)
        assert len(results) == 5
        assert all(len(r) == 2 for r in results)

    def test_sample_zero_shots(self):
        """零次采样返回空列表。"""
        state = QuantumState(3)
        results = state.sample(0)
        assert results == []

    def test_probability_out_of_range(self):
        """越界概率返回 0。"""
        state = QuantumState(3)
        assert state.probability(100) == 0.0
        assert state.probability(-1) == 0.0


class TestNormalization:
    """测试归一化。"""

    def test_normalize(self):
        """归一化恢复概率和为1。"""
        state = QuantumState(3)
        state.amplitudes[0] = 3.0 + 0j  # 破坏归一化
        state.normalize()
        assert state.check_normalization()

    def test_normalize_all_zeros(self):
        """全零振幅无法归一化。"""
        state = QuantumState(3)
        state.amplitudes = [0j] * 8
        try:
            state.normalize()
            assert False
        except StateNormalizationError:
            pass


class TestCloneAndProperties:
    """测试克隆和属性。"""

    def test_clone_independent(self):
        """克隆是独立副本。"""
        state = QuantumState(3)
        cloned = state.clone()
        state.x(0)
        assert cloned.amplitudes[0] == 1.0 + 0j
        assert cloned.n_qubits == 3

    def test_length(self):
        """__len__ 返回维度。"""
        state = QuantumState(4)
        assert len(state) == 16

    def test_getitem(self):
        """__getitem__ 访问振幅。"""
        state = QuantumState(2)
        state.x(0)
        assert state[0] == 0j
        assert state[1] == 1.0 + 0j


class TestMeasurementCollapse:
    """测试测量坍缩（缺陷21）。"""

    def test_measure_with_collapse(self):
        """测量后坍缩到测量结果。"""
        state = QuantumState(2)
        state.h(0)  # 均匀叠加: (|00> + |10>)/sqrt(2)
        state.h(1)  # (|00> + |01> + |10> + |11>)/2

        result, prob = state.measure(collapse=True)
        assert 0 <= result < 4
        assert abs(prob - 0.25) < 0.01

        # After collapse, the state must be pure |result>
        for i in range(4):
            if i == result:
                assert abs(state.amplitudes[i]) > 0.99
            else:
                assert abs(state.amplitudes[i]) < 1e-10

    def test_measure_without_collapse(self):
        """无坍缩测量不改变状态。"""
        state = QuantumState(2)
        state.h(0)

        amps_before = state.amplitudes.copy()
        result, prob = state.measure(collapse=False)

        for i in range(4):
            assert abs(state.amplitudes[i] - amps_before[i]) < 1e-12

    def test_sample_with_collapse(self):
        """多次坍缩测量后第二次应始终返回同一结果。"""
        state = QuantumState(2)
        state.h(0)
        state.h(1)

        results = state.sample(3, collapse=True)
        first = results[0][0]
        for r, _ in results:
            assert r == first  # 坍缩后每次测得同一结果


class TestDensityMatrix:
    """测试密度矩阵（缺陷22）。"""

    def test_init_pure(self):
        """初始化为 |0...0> 纯态。"""
        dm = DensityMatrix(2)
        assert dm.purity() > 0.99
        assert abs(dm.trace() - 1.0) < 1e-10
        assert dm.probability(0) > 0.99
        assert dm.n_qubits == 2
        assert dm.dim == 4

    def test_from_pure(self):
        """从纯态创建密度矩阵。"""
        state = QuantumState(2)
        state.h(0)
        state.cnot(0, 1)  # Bell state

        dm = DensityMatrix.from_pure(state)
        assert dm.purity() > 0.99
        assert abs(dm.trace() - 1.0) < 1e-10
        # Bell state: equal prob for |00> and |11>
        p00 = dm.probability(0)
        p11 = dm.probability(3)
        assert abs(p00 - 0.5) < 1e-10
        assert abs(p11 - 0.5) < 1e-10
        assert dm.probability(1) < 1e-10
        assert dm.probability(2) < 1e-10

    def test_measure(self):
        """DensityMatrix 测量。"""
        dm = DensityMatrix(2)
        result, prob = dm.measure()
        assert result == 0
        assert abs(prob - 1.0) < 1e-10

    def test_measure_collapse(self):
        """DensityMatrix 坍缩测量。"""
        state = QuantumState(2)
        state.h(0)  # (|00> + |10>)/sqrt(2)
        dm = DensityMatrix.from_pure(state)

        result, prob = dm.measure(collapse=True)
        assert prob > 0.0
        assert dm.purity() > 0.99  # Pure after collapse

    def test_depolarizing(self):
        """Depolarizing 通道降低纯度。"""
        state = QuantumState(2)
        state.h(0)  # pure: (|00> + |10>)/sqrt(2)
        dm = DensityMatrix.from_pure(state)

        purity_before = dm.purity()
        dm.apply_depolarizing(0.3)
        purity_after = dm.purity()

        assert purity_after < purity_before
        assert abs(dm.trace() - 1.0) < 1e-10

    def test_phase_damping(self):
        """相位阻尼.保持对角元素衰减非对角。"""
        state = QuantumState(1)
        state.h(0)  # (|0> + |1>)/sqrt(2), rho = [[0.5, 0.5], [0.5, 0.5]]
        dm = DensityMatrix.from_pure(state)
        dm.apply_phase_damping(0.5)

        # Diagonals preserved
        assert abs(dm.probability(0) - 0.5) < 1e-10
        assert abs(dm.probability(1) - 0.5) < 1e-10
        # Purity reduced (off-diagonals decayed)
        assert dm.purity() < 1.0

    def test_amplitude_damping(self):
        """振幅阻尼偏向 |0>。"""
        state = QuantumState(1)
        state.x(0)  # |1>
        dm = DensityMatrix.from_pure(state)
        dm.apply_amplitude_damping(0.5)

        # |1> partially decays to |0>
        assert dm.probability(0) > 0.0

    def test_von_neumann_entropy(self):
        """熵: 纯态 -> 0, 混合态 -> > 0。"""
        dm = DensityMatrix(2)
        assert dm.von_neumann_entropy() < 1e-10

        dm.apply_depolarizing(0.5)
        assert dm.von_neumann_entropy() > 0.1

    def test_fidelity(self):
        """Fidelity 在 0 到 1 之间。"""
        dm1 = DensityMatrix(2)
        dm2 = DensityMatrix(2)
        fid = dm1.fidelity(dm2)
        assert fid > 0.99  # Nearly identical pure states

        dm2.apply_depolarizing(0.5)
        fid2 = dm1.fidelity(dm2)
        assert 0.0 <= fid2 <= 1.0
        assert fid2 < fid

    def test_partial_trace(self):
        """Partial trace 还原。"""
        state = QuantumState(2)
        state.h(0)
        dm = DensityMatrix.from_pure(state)
        reduced = dm.partial_trace(1)

        assert reduced.n_qubits == 1
        assert reduced.dim == 2
        assert abs(reduced.trace() - 1.0) < 1e-10

    def test_from_probabilities(self):
        """混合态构造。"""
        s0 = QuantumState(1)
        s1 = QuantumState(1)
        s1.x(0)

        dm = DensityMatrix.from_probabilities([(0.5, s0), (0.5, s1)])
        assert abs(dm.trace() - 1.0) < 1e-10
        assert abs(dm.probability(0) - 0.5) < 1e-10
        assert abs(dm.probability(1) - 0.5) < 1e-10
        assert dm.purity() < 1.0  # Mixed state
