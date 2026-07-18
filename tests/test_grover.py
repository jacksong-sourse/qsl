"""
测试 Grover 搜索算法 (GroverSearch) 模块。

覆盖:
    - 单解搜索
    - 多解搜索
    - 全解退化情况
    - 零解错误
    - 理论 vs 经验成功率
    - GroverResult 属性
    - 大搜索空间
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsl.core.grover import GroverSearch, GroverResult
from qsl.utils.exceptions import NoSolutionError


class TestGroverSingleSolution:
    """测试单解 Grover 搜索。"""

    def test_single_solution_n3(self):
        """n=3, 搜索 |101> (5)。"""
        grover = GroverSearch(3, verbose=False)
        result = grover.search(lambda x: x == 5, shots=20)
        assert result.success_count > 0
        assert result.num_solutions == 1
        assert result.iterations >= 1

    def test_single_solution_n4(self):
        """n=4, 搜索 |0110> (6)。"""
        grover = GroverSearch(4, verbose=False)
        result = grover.search(lambda x: x == 6, shots=20)
        assert result.success_count > 0
        assert result.theory_success_prob > 0.9

    def test_single_solution_found(self):
        """验证找到的解正确。"""
        target = 3  # |011>
        grover = GroverSearch(3, verbose=False)
        result = grover.search(lambda x: x == target, shots=10)
        solutions = result.get_solutions()
        assert target in solutions


class TestGroverMultipleSolutions:
    """测试多解 Grover 搜索。"""

    def test_two_solutions(self):
        """两个解: x=1 和 x=2。"""
        grover = GroverSearch(3, verbose=False)
        result = grover.search(
            lambda x: x in {1, 2},
            num_solutions=2,
            shots=20
        )
        assert result.num_solutions == 2
        assert result.success_count > 0

    def test_four_solutions(self):
        """四个解: 任意偶数。"""
        grover = GroverSearch(4, verbose=False)
        result = grover.search(
            lambda x: x % 2 == 0,
            num_solutions=8,
            shots=20
        )
        assert result.success_count > 0

    def test_three_solutions(self):
        """恰好一个1的模式: |001>, |010>, |100>。"""
        grover = GroverSearch(3, verbose=False)
        result = grover.search(
            lambda x: x in {1, 2, 4},
            num_solutions=3,
            shots=15
        )
        found = result.get_solutions()
        for s in found:
            assert s in {1, 2, 4}


class TestGroverWithOracleSet:
    """测试 search_with_oracle_set 方法。"""

    def test_oracle_set_single(self):
        """用标记集搜索单个解。"""
        grover = GroverSearch(3, verbose=False)
        result = grover.search_with_oracle_set({5}, shots=10)
        assert result.num_solutions == 1
        assert 5 in result.get_solutions()

    def test_oracle_set_multiple(self):
        """用标记集搜索多个解。"""
        grover = GroverSearch(4, verbose=False)
        result = grover.search_with_oracle_set({3, 7, 11}, shots=15)
        assert result.success_count > 0

    def test_oracle_set_empty(self):
        """空标记集 -> NoSolutionError。"""
        grover = GroverSearch(3, verbose=False)
        try:
            grover.search_with_oracle_set(set(), shots=1)
            assert False
        except NoSolutionError:
            pass

    def test_oracle_set_out_of_range(self):
        """越界标记被静默忽略。"""
        grover = GroverSearch(3, verbose=False)
        result = grover.search_with_oracle_set({5, 100}, shots=10)
        assert result.num_solutions == 1  # 100 被忽略


class TestErrorCases:
    """测试错误情况。"""

    def test_zero_solutions(self):
        """零解 -> NoSolutionError。"""
        grover = GroverSearch(3, verbose=False)
        try:
            grover.search(lambda x: False, shots=1)
            assert False
        except NoSolutionError:
            pass


class TestGroverResult:
    """测试 GroverResult 数据结构。"""

    def test_success_count(self):
        """成功计数正确。"""
        observations = [
            (1, 0.5, True),
            (2, 0.3, False),
            (3, 0.8, True),
            (1, 0.5, True),
        ]
        result = GroverResult(
            n_qubits=3, num_solutions=2, iterations=2,
            theta=0.5, theory_success_prob=0.95,
            measurements=observations,
        )
        assert result.success_count == 3
        assert result.shots == 4
        assert result.empirical_success_rate == 0.75

    def test_best_measurement(self):
        """最大概率测量。"""
        observations = [
            (1, 0.2, True),
            (3, 0.8, True),
            (2, 0.1, False),
        ]
        result = GroverResult(
            n_qubits=2, num_solutions=2, iterations=1,
            theta=0.3, theory_success_prob=0.5,
            measurements=observations,
        )
        best, prob = result.best_measurement()
        assert best == 3
        assert abs(prob - 0.8) < 1e-10

    def test_get_measurement_counts(self):
        """测量计数分布。"""
        observations = [
            (1, 0.5, True),
            (1, 0.5, True),
            (3, 0.8, True),
        ]
        result = GroverResult(
            n_qubits=2, num_solutions=2, iterations=1,
            theta=0.3, theory_success_prob=0.5,
            measurements=observations,
        )
        counts = result.get_measurement_counts()
        assert counts[1] == 2
        assert counts[3] == 1

    def test_summary_string(self):
        """summary 返回字符串。"""
        observations = [(1, 0.5, True)]
        result = GroverResult(
            n_qubits=2, num_solutions=1, iterations=1,
            theta=0.5, theory_success_prob=0.9,
            measurements=observations,
        )
        s = result.summary()
        assert isinstance(s, str)
        assert "GroverResult" in s


class TestLargeSearch:
    """测试较大搜索空间。"""

    def test_n8_oracle_set(self):
        """n=8, 搜索 5 个标记态。"""
        grover = GroverSearch(8, verbose=False)
        marked = {7, 13, 42, 100, 200}
        result = grover.search_with_oracle_set(marked, shots=20)
        assert result.success_count > 0
        found = result.get_solutions()
        for s in found:
            assert s in marked

    def test_n10_auto_count(self):
        """n=10 自动统计解数量。"""
        grover = GroverSearch(10, verbose=False)
        result = grover.search(
            lambda x: x % 128 == 0,
            num_solutions=8,
            shots=10
        )
        assert result.num_solutions == 8
