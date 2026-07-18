"""
测试 qsl.network 中的分布式量子计算。

覆盖:
    - DistributedNode (分布式节点)
    - QuantumCluster (量子集群)
    - QuantumBlockchain (量子区块链)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from qsl.network.distributed_node import (
    DistributedNode,
    QuantumCluster,
    CircuitPartition,
)
from qsl.network.quantum_blockchain import (
    QuantumBlockchain,
    QuantumBlock,
    Transaction,
)


# ============================================================================
#  TestDistributedNode
# ============================================================================

class TestDistributedNode:

    def test_init(self):
        """DistributedNode 应正确初始化。"""
        node = DistributedNode(node_id=0, max_qubits=8, host="127.0.0.1", port=9000)
        assert node.node_id == 0
        assert node.max_qubits == 8
        assert node.host == "127.0.0.1"
        assert node.port == 9000

    def test_assign_partition_sets_state(self):
        """assign_partition 应设置内部状态并为非 None。"""
        node = DistributedNode(node_id=1, max_qubits=4)
        partition = CircuitPartition(
            node_id=1,
            gate_sequence=[{'gate': 'H', 'targets': [0]}],
            qubit_range=(0, 2),
            qubit_count=2,
        )
        node.assign_partition(partition)
        assert node._state is not None
        assert node._partition is not None

    def test_execute_gates_returns_numpy_array(self):
        """execute_gates 应返回 numpy 数组。"""
        node = DistributedNode(node_id=0, max_qubits=4)
        partition = CircuitPartition(
            node_id=0,
            gate_sequence=[
                {'gate': 'H', 'targets': [0]},
                {'gate': 'X', 'targets': [1]},
            ],
            qubit_range=(0, 2),
            qubit_count=2,
        )
        node.assign_partition(partition)
        result = node.execute_gates()
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_no_partition_raises_runtime_error(self):
        """未分配 partition 时调用 execute_gates 应抛出 RuntimeError。"""
        node = DistributedNode(node_id=0)
        with pytest.raises(RuntimeError, match="No partition assigned"):
            node.execute_gates()


# ============================================================================
#  TestQuantumCluster
# ============================================================================

class TestQuantumCluster:

    def test_init_with_nodes(self):
        """QuantumCluster 应接受节点列表初始化。"""
        nodes = [
            DistributedNode(node_id=0, max_qubits=4),
            DistributedNode(node_id=1, max_qubits=4),
        ]
        cluster = QuantumCluster(total_qubits=4, nodes=nodes)
        assert cluster.total_qubits == 4
        assert len(cluster.nodes) == 2

    def test_add_node_works(self):
        """add_node 应成功添加节点。"""
        cluster = QuantumCluster(total_qubits=4, nodes=[])
        assert len(cluster.nodes) == 0
        cluster.add_node(DistributedNode(node_id=0))
        assert len(cluster.nodes) == 1

    def test_partition_circuit_distributes_gates(self):
        """partition_circuit 应将门分配到节点。"""
        nodes = [
            DistributedNode(node_id=0, max_qubits=4),
            DistributedNode(node_id=1, max_qubits=4),
        ]
        cluster = QuantumCluster(total_qubits=4, nodes=nodes)
        circuit = [
            {'gate': 'H', 'targets': [0]},   # qubit 0 → node 0
            {'gate': 'X', 'targets': [3]},   # qubit 3 → node 1
        ]
        partitions = cluster.partition_circuit(circuit)
        assert len(partitions) == 2
        assert isinstance(partitions[0], CircuitPartition)
        assert isinstance(partitions[1], CircuitPartition)
        # node 0 应获得 qubit 0 上的门
        assert len(partitions[0].gate_sequence) >= 1

    def test_execute_parallel_returns_list(self):
        """execute_parallel 应返回状态向量列表。"""
        nodes = [
            DistributedNode(node_id=0, max_qubits=4),
            DistributedNode(node_id=1, max_qubits=4),
        ]
        cluster = QuantumCluster(total_qubits=4, nodes=nodes)
        circuit = [
            {'gate': 'H', 'targets': [0]},
            {'gate': 'X', 'targets': [3]},
        ]
        cluster.partition_circuit(circuit)
        results = cluster.execute_parallel()
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, np.ndarray)

    def test_get_cluster_state_returns_dict_with_node_id_keys(self):
        """get_cluster_state 应返回以 node_id 为键的字典。"""
        nodes = [
            DistributedNode(node_id=10, max_qubits=2),
            DistributedNode(node_id=20, max_qubits=2),
        ]
        cluster = QuantumCluster(total_qubits=4, nodes=nodes)
        circuit = [
            {'gate': 'H', 'targets': [0]},
        ]
        cluster.partition_circuit(circuit)
        state = cluster.get_cluster_state()
        assert isinstance(state, dict)
        assert 10 in state
        assert 20 in state


# ============================================================================
#  TestQuantumBlockchain
# ============================================================================

class TestQuantumBlockchain:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.blockchain = QuantumBlockchain(difficulty=0)

    def test_init_creates_genesis_block(self):
        """初始化应创建创世区块。"""
        assert self.blockchain.length == 1
        genesis = self.blockchain.chain[0]
        assert genesis.index == 0
        assert genesis.previous_hash == "0" * 64

    def test_add_transaction_adds_pending(self):
        """add_transaction 应添加待处理交易。"""
        self.blockchain.add_transaction("Alice", "Bob", 50.0)
        assert len(self.blockchain.pending_transactions) == 1
        tx = self.blockchain.pending_transactions[0]
        assert tx.sender == "Alice"
        assert tx.receiver == "Bob"
        assert tx.amount == 50.0

    def test_create_block_works_with_transactions(self):
        """有交易时 create_block 应创建新区块。"""
        self.blockchain.add_transaction("Alice", "Bob", 10.0)
        block = self.blockchain.create_block()
        assert block is not None
        assert block.index == 1
        assert len(block.transactions) == 1
        assert block.validator != ""

    def test_verify_chain_returns_true(self):
        """verify_chain 应对有效链返回 True。"""
        self.blockchain.add_transaction("A", "B", 1.0)
        self.blockchain.create_block()
        assert self.blockchain.verify_chain() is True

    def test_block_properties_index_hash_validator(self):
        """区块链应具有 index、hash、validator 属性。"""
        self.blockchain.add_transaction("X", "Y", 5.0)
        block = self.blockchain.create_block()
        if block is not None:
            assert isinstance(block.index, int)
            assert isinstance(block.hash, str)
            assert len(block.hash) > 0
            assert isinstance(block.validator, str)
            assert len(block.validator) > 0

    def test_summary_returns_string(self):
        """summary 应返回非空字符串。"""
        s = self.blockchain.summary()
        assert isinstance(s, str)
        assert len(s) > 0
        assert "Quantum Blockchain" in s

    def test_transaction_hashing_works(self):
        """Transaction.hash 应返回有效的 SHA256 哈希。"""
        tx = Transaction(sender="Alice", receiver="Bob", amount=100.0)
        h = tx.hash()
        assert isinstance(h, str)
        assert len(h) == 64  # SHA256 hex digest
        # 相同数据应产生相同哈希
        assert h == tx.hash()

    def test_quantum_block_mining_works_with_low_difficulty(self):
        """低 difficulty 下挖矿应成功。"""
        tx = Transaction(sender="Miner", receiver="Reward", amount=1.0)
        block = QuantumBlock(
            index=1,
            transactions=[tx],
            previous_hash="0" * 64,
        )
        mined = block.mine_block(difficulty=0)
        assert mined is True
