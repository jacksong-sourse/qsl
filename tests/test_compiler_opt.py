"""
测试编译器优化器、转译器和错误缓解模块。

覆盖:
    - 编译器优化器 (gate_fusion, commutation_optimization, depth_reduction)
    - 编译器转译器 (layout_mapping, swap_insertion, get_coupling_graph)
    - 错误缓解 (zne, readout_error_correction, build_confusion_matrix, richardson_extrapolate)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from qsl.compiler.optimizer import gate_fusion, commutation_optimization, depth_reduction
from qsl.compiler.transpiler import layout_mapping, swap_insertion, get_coupling_graph
from qsl.compiler.error_mitigation import (
    zne,
    readout_error_correction,
    build_confusion_matrix,
    richardson_extrapolate,
)


# ============================================================================
# 辅助函数：构造简单门序列
# ============================================================================

def _h(q):
    """Hadamard 门。"""
    return {"gate": "H", "targets": [q]}


def _x(q):
    """Pauli-X 门。"""
    return {"gate": "X", "targets": [q]}


def _z(q):
    """Pauli-Z 门。"""
    return {"gate": "Z", "targets": [q]}


def _cnot(control, target):
    """CNOT 门。"""
    return {"gate": "CNOT", "targets": [target], "control": control}


def _cz(control, target):
    """CZ 门。"""
    return {"gate": "CZ", "targets": [target], "control": control}


def _swap(q0, q1):
    """SWAP 门。"""
    return {"gate": "SWAP", "targets": [q0, q1]}


# ============================================================================
# TestOptimizer
# ============================================================================

class TestOptimizer:
    """测试编译器优化器：gate_fusion, commutation_optimization, depth_reduction。"""

    # ---------- gate_fusion ----------

    def test_gate_fusion_simple_sequence(self):
        """简单门序列：连续同量子比特单门应被聚合。"""
        seq = [_h(0), _x(0), _z(0)]
        result = gate_fusion(seq)
        assert len(result) == 3  # 聚合仍保留各自 gate_op，只是分在同一 pending 组
        # 所有单门都在结果中，多量子比特门可以打断 pending
        gates_in_result = [g["gate"] for g in result]
        assert "H" in gates_in_result
        assert "X" in gates_in_result
        assert "Z" in gates_in_result

    def test_gate_fusion_multi_qubit_gate_flushes_pending(self):
        """多量子比特门（targets 长度 > 1）应清空 pending 缓冲后再插入自身。"""
        # 使用 targets 长度 > 1 的门（如 SWAP）来触发 pending 清空
        seq = [_h(0), _x(0), _swap(0, 1), _z(0)]
        result = gate_fusion(seq)
        # SWAP(0,1) 应出现在 X(0) 之后、Z(0) 之前
        gates = [g["gate"] for g in result]
        h_idx = gates.index("H")
        x_idx = gates.index("X")
        swap_idx = gates.index("SWAP")
        z_idx = gates.index("Z")
        assert h_idx < x_idx < swap_idx < z_idx

    def test_gate_fusion_separate_qubits(self):
        """不同量子比特的单门互不影响。"""
        seq = [_h(0), _x(1), _z(0), _h(1)]
        result = gate_fusion(seq)
        assert len(result) == 4

    def test_gate_fusion_empty_list(self):
        """空列表应返回 []。"""
        result = gate_fusion([])
        assert result == []

    def test_gate_fusion_no_targets(self):
        """无 targets 的门应被直接追加到输出。"""
        seq = [{"gate": "BARRIER", "targets": []}]
        result = gate_fusion(seq)
        assert result == seq

    # ---------- commutation_optimization ----------

    def test_commutation_optimization_disjoint_qubits(self):
        """作用于不相交量子比特集的门应可交换重排。"""
        seq = [_h(0), _x(1), _z(0), _h(1)]
        result = commutation_optimization(seq)
        assert len(result) == len(seq)
        # 每个门都应出现在结果中
        result_gates = {(g["gate"], tuple(g["targets"])) for g in result}
        assert result_gates == {("H", (0,)), ("X", (1,)), ("Z", (0,)), ("H", (1,))}

    def test_commutation_optimization_overlapping_qubits_not_reordered(self):
        """作用于同一量子比特的门不应被错误重排。"""
        seq = [_h(0), _x(0), _z(0)]
        result = commutation_optimization(seq)
        # 作用于同一个 qubit 的门顺序必须保持
        assert result[0]["gate"] == "H"
        assert result[1]["gate"] == "X"
        assert result[2]["gate"] == "Z"

    def test_commutation_optimization_single_gate(self):
        """单门序列，应原样返回。"""
        seq = [_h(0)]
        result = commutation_optimization(seq)
        assert result == seq

    def test_commutation_optimization_empty(self):
        """空序列应返回空列表。"""
        result = commutation_optimization([])
        assert result == []

    # ---------- depth_reduction ----------

    def test_depth_reduction_reduces_depth(self):
        """depth_reduction 应降低或保持电路深度。"""
        seq = [_h(0), _h(1), _h(2), _cnot(0, 1), _cnot(1, 2)]
        optimized, orig_depth, opt_depth = depth_reduction(seq)
        assert opt_depth <= orig_depth
        assert orig_depth == len(seq)

    def test_depth_reduction_return_tuple(self):
        """返回值应为 (optimized_seq, orig_depth, opt_depth) 三元组。"""
        seq = [_h(0), _h(1), _cnot(0, 1)]
        result = depth_reduction(seq)
        assert isinstance(result, tuple)
        assert len(result) == 3
        optimized, orig_depth, opt_depth = result
        assert isinstance(optimized, list)
        assert isinstance(orig_depth, int)
        assert isinstance(opt_depth, int)

    def test_depth_reduction_preserves_all_gates(self):
        """优化后应保留所有门。"""
        seq = [_h(0), _z(0), _x(1), _h(1), _cnot(0, 1)]
        optimized, _, _ = depth_reduction(seq)
        assert len(optimized) == len(seq)
        orig_ids = {(g["gate"], tuple(g.get("targets", [])), g.get("control")) for g in seq}
        opt_ids = {(g["gate"], tuple(g.get("targets", [])), g.get("control")) for g in optimized}
        assert orig_ids == opt_ids

    def test_depth_reduction_empty_list(self):
        """空列表应返回 ([], 0, 0)。"""
        result = depth_reduction([])
        assert result == ([], 0, 0)

    def test_depth_reduction_parallel_gates_same_layer(self):
        """作用在不同量子比特上的门可以并行调度到同一层。"""
        # 三个 H 门可并行
        seq = [_h(0), _h(1), _h(2)]
        _, orig_depth, opt_depth = depth_reduction(seq)
        assert opt_depth == 1
        assert orig_depth == 3


# ============================================================================
# TestTranspiler
# ============================================================================

class TestTranspiler:
    """测试编译器转译器：layout_mapping, swap_insertion, get_coupling_graph。"""

    # ---------- layout_mapping ----------

    def test_layout_mapping_returns_dict(self):
        """layout_mapping 返回 logical->physical 映射字典。"""
        coupling = [(0, 1), (1, 2)]
        gates = [_cnot(0, 1), _cnot(1, 2)]
        mapping = layout_mapping(3, coupling, gates)
        assert isinstance(mapping, dict)

    def test_layout_mapping_covers_all_logical_qubits(self):
        """映射应覆盖所有逻辑量子比特。"""
        coupling = [(0, 1), (1, 2)]
        gates = [_cnot(0, 1), _cnot(1, 2)]
        mapping = layout_mapping(3, coupling, gates)
        assert set(mapping.keys()) == {0, 1, 2}

    def test_layout_mapping_keys_are_ints(self):
        """映射的键和值都应是整数。"""
        coupling = [(0, 1)]
        gates = [_cnot(0, 1)]
        mapping = layout_mapping(2, coupling, gates)
        for k, v in mapping.items():
            assert isinstance(k, int)
            assert isinstance(v, int)

    def test_layout_mapping_no_gates(self):
        """无门序列时也应能映射。"""
        coupling = [(0, 1), (1, 2)]
        mapping = layout_mapping(3, coupling, [])
        assert isinstance(mapping, dict)

    # ---------- swap_insertion ----------

    def test_swap_insertion_single_qubit_gates_pass_through(self):
        """单量子比特门应原样通过，不插入 SWAP。"""
        coupling = [(0, 1)]
        seq = [_h(0), _x(0), _z(1)]
        result = swap_insertion(seq, coupling)
        assert len(result) == len(seq)
        for g in result:
            assert g["gate"] != "SWAP"

    def test_swap_insertion_adjacent_two_qubit_gate_no_swap(self):
        """物理相邻量子比特上的两量子比特门不需要 SWAP。"""
        coupling = [(0, 1), (1, 2)]
        seq = [_cnot(0, 1)]
        result = swap_insertion(seq, coupling)
        # 0 和 1 在耦合图中相邻，所以不应插入 SWAP
        no_swaps = all(g.get("gate") != "SWAP" for g in result)
        assert no_swaps

    def test_swap_insertion_non_adjacent_may_insert_swap(self):
        """非相邻量子比特间的门可能插入 SWAP。"""
        # 线形耦合 0-1-2，所以 0 和 2 不相邻
        coupling = [(0, 1), (1, 2)]
        seq = [_cnot(0, 2)]
        initial_mapping = {0: 0, 1: 1, 2: 2}
        result = swap_insertion(seq, coupling, initial_mapping)
        # 0 和 2 不相邻，应插入 SWAP
        has_swap = any(g.get("gate") == "SWAP" for g in result)
        assert has_swap

    def test_swap_insertion_empty_coupling_graph(self):
        """空耦合图应原样返回门序列。"""
        seq = [_cnot(0, 1), _h(0)]
        result = swap_insertion(seq, [])
        assert result == seq

    # ---------- get_coupling_graph ----------

    def test_get_coupling_graph_simulator(self):
        """simulator 设备返回全连接拓扑。"""
        graph = get_coupling_graph("simulator")
        assert isinstance(graph, list)
        assert len(graph) > 0
        # 全连接：应有 n*(n-1)/2 条边，n=20 => 190
        assert len(graph) == 190

    def test_get_coupling_graph_ibm_sherbrooke(self):
        """IBM Sherbrooke 设备返回重六边形拓扑。"""
        graph = get_coupling_graph("ibm_sherbrooke")
        assert isinstance(graph, list)
        assert len(graph) > 0
        # 检查每条边都是二元组
        for edge in graph:
            assert isinstance(edge, tuple)
            assert len(edge) == 2
            assert all(isinstance(q, int) for q in edge)

    def test_get_coupling_graph_ionq(self):
        """IonQ 设备返回线形拓扑。"""
        graph = get_coupling_graph("ionq")
        assert isinstance(graph, list)
        assert len(graph) > 0
        # 线形拓扑：相邻量子比特之间的边，range(20) => 20 条边
        assert len(graph) == 20
        assert graph[0] == (0, 1)

    def test_get_coupling_graph_default(self):
        """默认设备应返回重六边形拓扑。"""
        graph = get_coupling_graph()
        assert isinstance(graph, list)
        assert len(graph) > 0


# ============================================================================
# TestErrorMitigation
# ============================================================================

class TestErrorMitigation:
    """测试错误缓解：zne, readout_error_correction, build_confusion_matrix, richardson_extrapolate。"""

    # ---------- zne ----------

    def test_zne_linear_extrapolation(self):
        """ZNE 线性外推：对简单线性函数应正确外推到零噪声。"""
        # E(λ) = 1.0 - 0.1*λ，零噪声值应为 1.0
        def linear_fn(x):
            return 1.0 - 0.1 * x

        result = zne(linear_fn, noise_scales=[1.0, 2.0, 3.0], extrapolation="linear")
        assert pytest.approx(result, abs=1e-6) == 1.0

    def test_zne_exponential_with_positive_values(self):
        """ZNE 指数外推：对正值指数衰减函数应正确外推。"""
        # E(λ) = 0.5 * exp(-0.5*λ)，零噪声值应为 0.5
        def exp_fn(x):
            return 0.5 * np.exp(-0.5 * x)

        result = zne(exp_fn, noise_scales=[1.0, 2.0, 3.0], extrapolation="exponential")
        assert pytest.approx(result, abs=1e-6) == 0.5

    def test_zne_raises_for_less_than_two_noise_scales(self):
        """少于两个噪声尺度时应抛出 ValueError。"""
        def dummy_fn(x):
            return x

        with pytest.raises(ValueError, match="at least 2"):
            zne(dummy_fn, noise_scales=[1.0], extrapolation="linear")

    def test_zne_default_noise_scales(self):
        """未指定 noise_scales 时使用默认值 [1.0, 2.0, 3.0]。"""
        def fn(x):
            return 2.0 - 0.5 * x

        result = zne(fn, extrapolation="linear")
        assert pytest.approx(result, abs=1e-6) == 2.0

    # ---------- readout_error_correction ----------

    def test_readout_error_correction_simple_single_qubit(self):
        """简单单量子比特读出差错校正。"""
        # 观测到 {"0": 90, "1": 10}，真实分布应接近均匀
        confusion = np.array([[0.95, 0.05], [0.05, 0.95]])
        counts = {"0": 90, "1": 10}
        corrected = readout_error_correction(counts, confusion)
        assert isinstance(corrected, dict)
        # 校正后 "0" 和 "1" 的计数应更均衡
        assert corrected["0"] > 0
        assert corrected["1"] > 0

    def test_readout_error_correction_empty_dict(self):
        """空字典返回 {}。"""
        confusion = build_confusion_matrix(0.05)
        result = readout_error_correction({}, confusion)
        assert result == {}

    # ---------- build_confusion_matrix ----------

    def test_build_confusion_matrix_shape(self):
        """混淆矩阵应为 2x2 矩阵。"""
        mat = build_confusion_matrix(0.05)
        assert mat.shape == (2, 2)

    def test_build_confusion_matrix_rows_sum_to_one(self):
        """每行的元素之和应为 1（随机矩阵/马尔可夫矩阵性质）。"""
        mat = build_confusion_matrix(0.1)
        assert pytest.approx(mat[0, 0] + mat[0, 1]) == 1.0
        assert pytest.approx(mat[1, 0] + mat[1, 1]) == 1.0

    def test_build_confusion_matrix_diagonal_dominance(self):
        """对角线元素应更大（正确读出概率更高）。"""
        mat = build_confusion_matrix(0.01)
        assert mat[0, 0] > mat[0, 1]
        assert mat[1, 1] > mat[1, 0]
        assert pytest.approx(mat[0, 0]) == 0.99
        assert pytest.approx(mat[0, 1]) == 0.01

    # ---------- richardson_extrapolate ----------

    def test_richardson_extrapolate_known_polynomial(self):
        """用已知多项式值测试 Richardson 外推。"""
        # 值来自多项式 f(λ) = 1 + 2λ + 3λ²，λ=0 时为 1
        # f(1)=6, f(2)=17, f(3)=34
        values = [6.0, 17.0, 34.0]
        noise_scales = [1.0, 2.0, 3.0]
        result = richardson_extrapolate(values, noise_scales)
        assert pytest.approx(result, abs=1e-6) == 1.0

    def test_richardson_extrapolate_two_points(self):
        """两点 Richardson 外推应给出线性外推结果。"""
        # f(λ) = 5 - 2λ，f(1)=3, f(2)=1，零噪声应为 5
        values = [3.0, 1.0]
        noise_scales = [1.0, 2.0]
        result = richardson_extrapolate(values, noise_scales)
        assert pytest.approx(result, abs=1e-6) == 5.0

    def test_richardson_extrapolate_single_value(self):
        """单值时返回该值本身。"""
        result = richardson_extrapolate([3.0], [1.0])
        assert result == 3.0

    def test_richardson_extrapolate_empty_list(self):
        """空列表返回 0.0。"""
        result = richardson_extrapolate([], [])
        assert result == 0.0
