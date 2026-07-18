"""
Distributed Quantum-Classical Hybrid Computing Nodes.

*** WARNING: DEMONSTRATION ONLY — conceptual design, no actual distribution ***
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class CircuitPartition:
    """A partition of a quantum circuit assigned to one node."""
    node_id: int
    gate_sequence: List[dict]
    qubit_range: Tuple[int, int]  # (start_qubit, end_qubit)
    qubit_count: int


class DistributedNode:
    """
    A single node in a quantum-classical hybrid computing cluster.
    
    Each node simulates a subset of qubits and communicates
    state information with other nodes for cross-partition gates.
    
    Args:
        node_id: Unique node identifier
        max_qubits: Maximum qubits this node can simulate
        host: Host address (for networking, simulated locally)
        port: Port number
    """
    
    def __init__(self,
                 node_id: int,
                 max_qubits: int = 10,
                 host: str = "localhost",
                 port: int = 0):
        self.node_id = node_id
        self.max_qubits = max_qubits
        self.host = host
        self.port = port or (8000 + node_id)
        self._state = None
        self._partition: Optional[CircuitPartition] = None
    
    def assign_partition(self, partition: CircuitPartition):
        """Assign a circuit partition to this node."""
        self._partition = partition
        from ..core.state import QuantumState
        self._state = QuantumState(partition.qubit_count)
    
    def execute_gates(self) -> np.ndarray:
        """
        Execute assigned gates on local state.
        
        Returns:
            State vector after gate execution
        """
        if self._state is None:
            raise RuntimeError("No partition assigned")
        
        if self._partition is None:
            return np.array([])
        
        for gate in self._partition.gate_sequence:
            gate_type = gate.get('gate', '')
            targets = gate.get('targets', [])
            
            # Map targets to local qubit indices
            local_targets = []
            for t in targets:
                local_t = t - self._partition.qubit_range[0]
                if 0 <= local_t < self._partition.qubit_count:
                    local_targets.append(local_t)
            
            if not local_targets:
                continue
            
            if gate_type == 'H':
                for t in local_targets:
                    self._state.h(t)
            elif gate_type == 'X':
                for t in local_targets:
                    self._state.x(t)
            elif gate_type == 'CNOT':
                control = gate.get('control', -1) - self._partition.qubit_range[0]
                target = local_targets[0]
                if 0 <= control < self._partition.qubit_count:
                    self._state.cnot(control, target)
            elif gate_type == 'CZ':
                control = gate.get('control', -1) - self._partition.qubit_range[0]
                target = local_targets[0]
                if 0 <= control < self._partition.qubit_count:
                    self._state.cz(control, target)
        
        return np.array(self._state.amplitudes)
    
    def get_state(self):
        """Get current local state."""
        return self._state


class QuantumCluster:
    """
    Cluster of distributed quantum computing nodes.
    
    Automatically partitions quantum circuits across nodes,
    handles cross-node communication for multi-qubit gates.
    
    Args:
        total_qubits: Total number of qubits in the circuit
        nodes: List of DistributedNode instances
    """
    
    def __init__(self, total_qubits: int, nodes: List[DistributedNode]):
        self.total_qubits = total_qubits
        self.nodes = nodes
        self._partitions: List[CircuitPartition] = []
    
    def add_node(self, node: DistributedNode):
        """Add a compute node to the cluster."""
        self.nodes.append(node)
    
    def partition_circuit(self, 
                           circuit: List[dict]) -> List[CircuitPartition]:
        """
        Partition a circuit across available nodes.
        
        Strategy:
        1. Assign contiguous qubit ranges to each node
        2. Local gates stay on their node
        3. Cross-partition gates require communication
        
        Args:
            circuit: Full circuit gate sequence
            
        Returns:
            List of CircuitPartition for each node
        """
        n_nodes = len(self.nodes)
        if n_nodes == 0:
            return []
        
        # Distribute qubits evenly
        qubits_per_node = math.ceil(self.total_qubits / n_nodes)
        partitions = []
        
        for i, node in enumerate(self.nodes):
            start = i * qubits_per_node
            end = min((i + 1) * qubits_per_node, self.total_qubits)
            
            # Assign gates that only affect this partition's qubits
            local_gates = []
            for gate in circuit:
                targets = gate.get('targets', [])
                control = gate.get('control')
                all_qubits = set(targets)
                if control is not None:
                    all_qubits.add(control)
                
                # Check if all affected qubits are in this partition
                if all(start <= q < end for q in all_qubits):
                    local_gates.append(gate)
            
            partition = CircuitPartition(
                node_id=node.node_id,
                gate_sequence=local_gates,
                qubit_range=(start, end),
                qubit_count=end - start,
            )
            partitions.append(partition)
            node.assign_partition(partition)
        
        self._partitions = partitions
        return partitions
    
    def execute_parallel(self) -> List[np.ndarray]:
        """
        Execute all partitions in parallel (simulated sequentially).
        
        Returns:
            List of state vectors from each node
        """
        results = []
        for node in self.nodes:
            state = node.execute_gates()
            results.append(state)
        return results
    
    def get_cluster_state(self) -> Dict[int, np.ndarray]:
        """
        Get the state of all nodes in the cluster.
        
        Returns:
            Dict mapping node_id -> state_vector
        """
        return {node.node_id: node.get_state() for node in self.nodes}
    
    @property
    def partitions(self) -> List[CircuitPartition]:
        return self._partitions
