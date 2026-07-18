"""
测试 qsl.meta 中的自进化量子 AI 系统。

覆盖:
    - AlgorithmSearcher (遗传算法搜索)
    - QuantumCompilerAI (RL 编译器)
    - QuantumTheoremProver (定理证明器)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from qsl.meta.algorithm_search import CircuitGenome, AlgorithmSearcher
from qsl.meta.quantum_compiler_ai import QuantumCompilerAI
from qsl.meta.theory_generator import QuantumTheoremProver, ProofResult


# ============================================================================
#  TestAlgorithmSearcher
# ============================================================================

class TestAlgorithmSearcher:

    def test_circuit_genome_random_creates_valid_genome(self):
        """CircuitGenome.random 应创建包含有效字段的基因组。"""
        genome = CircuitGenome.random(n_qubits=3, max_gates=10)
        assert isinstance(genome.gates, list)
        assert len(genome.gates) >= 1
        assert genome.n_qubits == 3
        for gate in genome.gates:
            assert 'gate' in gate

    def test_to_statevector_circuit_returns_callable(self):
        """to_statevector_circuit 应返回可调用对象。"""
        genome = CircuitGenome.random(n_qubits=2, max_gates=5)
        circuit_fn = genome.to_statevector_circuit()
        assert callable(circuit_fn)
        # 测试可调用性：对 QuantumState 调用
        from qsl import QuantumState
        state = QuantumState(2)
        result = circuit_fn(state)
        assert result is not None

    def test_algorithm_searcher_initializes_correctly(self):
        """AlgorithmSearcher 应正确初始化参数。"""
        searcher = AlgorithmSearcher(
            n_qubits=2,
            population_size=10,
            generations=5,
            mutation_rate=0.1,
            crossover_rate=0.5,
        )
        assert searcher.n_qubits == 2
        assert searcher.population_size == 10
        assert searcher.generations == 5
        assert searcher.mutation_rate == 0.1
        assert searcher.crossover_rate == 0.5

    def test_search_runs(self):
        """search 方法应能运行少量代数（5 代、10 个种群）。"""
        searcher = AlgorithmSearcher(
            n_qubits=3,
            population_size=10,
            generations=5,
            mutation_rate=0.1,
            crossover_rate=0.5,
        )
        result = searcher.search(verbose=False)
        assert isinstance(result, CircuitGenome)
        assert len(result.gates) >= 1

    def test_best_genome_property_after_search(self):
        """search 后 best_genome 属性应非空。"""
        searcher = AlgorithmSearcher(
            n_qubits=3, population_size=8, generations=3,
        )
        assert searcher.best_genome is None
        searcher.search(verbose=False)
        assert searcher.best_genome is not None
        assert isinstance(searcher.best_genome, CircuitGenome)

    def test_history_records_fitness(self):
        """history 应记录每代最佳适应度。"""
        searcher = AlgorithmSearcher(
            n_qubits=3, population_size=10, generations=5,
        )
        searcher.search(verbose=False)
        history = searcher.history
        assert len(history) == 5
        assert all(isinstance(f, float) for f in history)
        assert all(0.0 <= f <= 1.0 for f in history)


# ============================================================================
#  TestQuantumCompilerAI
# ============================================================================

class TestQuantumCompilerAI:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.compiler = QuantumCompilerAI(
            n_qubits=3,
            learning_rate=0.01,
            gamma=0.95,
            epsilon=0.1,
        )

    def test_initializes_with_correct_params(self):
        """初始化参数应正确存储。"""
        assert self.compiler.n_qubits == 3
        assert self.compiler.lr == 0.01
        assert self.compiler.gamma == 0.95
        assert self.compiler.epsilon == 0.1

    def test_train_runs_with_simple_circuits(self):
        """train 应能运行简单 circuit（2 个 episode）。"""
        circuits = [
            [{'gate': 'H', 'targets': [0]}, {'gate': 'H', 'targets': [0]},
             {'gate': 'X', 'targets': [1]}, {'gate': 'X', 'targets': [1]}],
            [{'gate': 'H', 'targets': [0]}, {'gate': 'CNOT', 'control': 0, 'target': 1}],
        ]
        rewards = self.compiler.train(circuits, episodes=2, verbose=False)
        assert isinstance(rewards, list)
        assert len(rewards) == 2

    def test_compile_produces_output(self):
        """compile 应产生输出电路。"""
        circuits = [
            [{'gate': 'H', 'targets': [0]}, {'gate': 'H', 'targets': [0]},
             {'gate': 'X', 'targets': [1]}, {'gate': 'X', 'targets': [1]}],
        ]
        self.compiler.train(circuits, episodes=2, verbose=False)
        result = self.compiler.compile(circuits[0], max_steps=5)
        assert isinstance(result, list)

    def test_hash_state_returns_int(self):
        """_hash_state 应返回整数。"""
        circuit = [{'gate': 'H', 'targets': [0]}, {'gate': 'X', 'targets': [1]}]
        h = self.compiler._hash_state(circuit)
        assert isinstance(h, int)

    def test_fuse_adjacent_works(self):
        """_fuse_adjacent 应正确融合相邻同量子比特门。"""
        circuit = [
            {'gate': 'H', 'targets': [0]},
            {'gate': 'Z', 'targets': [0]},
            {'gate': 'X', 'targets': [1]},
        ]
        result = self.compiler._fuse_adjacent(circuit)
        assert isinstance(result, list)
        # 前两个单量子比特门作用于同一量子比特 0，应被融合
        fused_gates = [g['gate'] for g in result]
        assert 'FUSED_U3' in fused_gates

    def test_get_valid_actions_returns_valid_indices(self):
        """_get_valid_actions 应返回有效动作索引列表。"""
        circuit = [{'gate': 'H', 'targets': [0]}, {'gate': 'X', 'targets': [1]}]
        actions = self.compiler._get_valid_actions(circuit)
        assert isinstance(actions, list)
        assert len(actions) > 0
        for idx in actions:
            assert isinstance(idx, int)
            assert 0 <= idx < len(self.compiler._actions)


# ============================================================================
#  TestQuantumTheoremProver
# ============================================================================

class TestQuantumTheoremProver:

    def test_initializes(self):
        """QuantumTheoremProver 应正确初始化。"""
        prover = QuantumTheoremProver(
            conjecture="a + b = b + a",
            max_proof_depth=3,
            max_branching=2,
        )
        assert prover.conjecture == "a + b = b + a"
        assert prover.max_proof_depth == 3
        assert prover.max_branching == 2

    def test_generate_proof_paths_works(self):
        """_generate_proof_paths 应返回有效证明路径列表。"""
        prover = QuantumTheoremProver()
        axioms = prover._generate_axioms("arithmetic")
        paths = prover._generate_proof_paths(axioms, max_depth=2)
        assert isinstance(paths, list)
        assert len(paths) > 0
        for path in paths:
            assert isinstance(path, list)
            assert 1 <= len(path) <= 2

    def test_prove_method_runs_for_simple_conjecture(self):
        """prove 方法应对简单猜想返回结果。"""
        prover = QuantumTheoremProver(
            conjecture="a AND b = b AND a",
            max_proof_depth=2,
        )
        result = prover.prove()
        assert isinstance(result, ProofResult)
        assert result.conjecture == "a AND b = b AND a"

    def test_proof_result_fields_populated(self):
        """ProofResult 各字段应被正确填充。"""
        prover = QuantumTheoremProver(
            conjecture="a + 0 = a",
            max_proof_depth=2,
        )
        result = prover.prove()
        assert result.num_paths_explored > 0
        assert result.classical_steps > 0
        if result.proof_found:
            assert result.proof_steps is not None
            assert isinstance(result.confidence, float)
            assert 0.0 <= result.confidence <= 1.0

    def test_handles_empty_conjecture_gracefully(self):
        """空猜想应被优雅处理（不抛异常）。"""
        prover = QuantumTheoremProver(conjecture="", max_proof_depth=2)
        result = prover.prove()
        assert isinstance(result, ProofResult)
        assert result.conjecture == ""
