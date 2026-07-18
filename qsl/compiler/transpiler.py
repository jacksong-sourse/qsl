"""
Quantum circuit transpiler for hardware-specific compilation.

Maps logical qubits to physical qubits considering device coupling maps
and inserts SWAP gates where needed.
"""

import numpy as np
from typing import List, Dict, Tuple, Set


def layout_mapping(logical_qubits: int,
                   coupling_graph: List[Tuple[int, int]],
                   gate_sequence: List[dict]) -> Dict[int, int]:
    """
    Map logical qubits to physical qubits based on coupling graph.
    
    Uses a greedy strategy to place frequently-interacting logical 
    qubits on physically adjacent physical qubits.
    
    Args:
        logical_qubits: Number of logical qubits
        coupling_graph: List of (physical_q0, physical_q1) edges
        gate_sequence: List of gate operations with 'targets' and optional 'control'
        
    Returns:
        Dict mapping logical_index -> physical_index
    """
    # Build interaction frequency matrix
    interaction_freq = np.zeros((logical_qubits, logical_qubits))
    
    for gate_op in gate_sequence:
        targets = gate_op.get('targets', [])
        control = gate_op.get('control')
        
        qubits = list(targets)
        if control is not None:
            qubits.append(control)
        
        for i in range(len(qubits)):
            for j in range(i + 1, len(qubits)):
                a, b = qubits[i], qubits[j]
                interaction_freq[a, b] += 1
                interaction_freq[b, a] += 1
    
    # Build physical adjacency
    physical_adj = set()
    for a, b in coupling_graph:
        physical_adj.add((a, b))
        physical_adj.add((b, a))
    
    # Greedy mapping: most-interacting logical pairs -> physically adjacent
    logical_pairs = []
    for i in range(logical_qubits):
        for j in range(i + 1, logical_qubits):
            if interaction_freq[i, j] > 0:
                logical_pairs.append(((i, j), interaction_freq[i, j]))
    
    logical_pairs.sort(key=lambda x: -x[1])
    
    # Simple greedy placement
    mapping = {}
    reverse_mapping = {}
    
    for (li, lj), freq in logical_pairs:
        if li in reverse_mapping and lj in reverse_mapping:
            continue  # Both already mapped
        
        if li in reverse_mapping:
            pi = reverse_mapping[li]
            # Find adjacent physical qubit
            for pj in range(len(physical_adj) if physical_adj else logical_qubits):
                if pj not in mapping and (pi, pj) in physical_adj:
                    mapping[pj] = lj
                    reverse_mapping[lj] = pj
                    break
        
        elif lj in reverse_mapping:
            pj = reverse_mapping[lj]
            for pi in range(len(physical_adj)):
                if pi not in mapping and (pi, pj) in physical_adj:
                    mapping[pi] = li
                    reverse_mapping[li] = pi
                    break
        
        else:
            # Neither mapped: place on adjacent physical pair
            for pa, pb in physical_adj:
                if pa not in mapping and pb not in mapping:
                    mapping[pa] = li
                    mapping[pb] = lj
                    reverse_mapping[li] = pa
                    reverse_mapping[lj] = pb
                    break
    
    # Map remaining unplaced qubits
    for q in range(logical_qubits):
        if q not in reverse_mapping:
            for p in range(logical_qubits):
                if p not in mapping:
                    mapping[p] = q
                    reverse_mapping[q] = p
                    break
    
    return reverse_mapping  # logical -> physical


def swap_insertion(gate_sequence: List[dict],
                   coupling_graph: List[Tuple[int, int]],
                   initial_mapping: Dict[int, int] = None) -> List[dict]:
    """
    Insert SWAP gates to satisfy coupling constraints.
    
    For each two-qubit gate, if the qubits are not physically adjacent,
    insert SWAP gates to make them adjacent before the gate.
    
    Args:
        gate_sequence: List of gate operations
        coupling_graph: List of (phys_q0, phys_q1) edges (physical adjacency)
        initial_mapping: Initial logical-to-physical mapping
        
    Returns:
        Transpiled gate sequence with SWAP gates inserted
    """
    if not coupling_graph:
        return gate_sequence
    
    # Build adjacency set
    adjacent = set()
    for a, b in coupling_graph:
        adjacent.add((a, b))
        adjacent.add((b, a))
    
    # Current mapping (logical -> physical)
    if initial_mapping is None:
        n_qubits = max(
            max(max(g.get('targets', [0])), g.get('control', 0))
            for g in gate_sequence
        ) + 1 if gate_sequence else 0
        mapping = {q: q for q in range(n_qubits)}
    else:
        mapping = dict(initial_mapping)
    
    # Inverse mapping (physical -> logical)
    inv_mapping = {p: l for l, p in mapping.items()}
    
    transpiled = []
    
    for gate_op in gate_sequence:
        targets = gate_op.get('targets', [])
        control = gate_op.get('control')
        
        if control is not None:
            qubits = targets + [control]
        else:
            qubits = list(targets)
        
        if len(qubits) <= 1:
            transpiled.append(gate_op)
            continue
        
        # Check if physically adjacent
        phys_qubits = [mapping.get(q, q) for q in qubits]
        
        if len(phys_qubits) == 2:
            p0, p1 = phys_qubits
            if (p0, p1) not in adjacent:
                # Need SWAP: find a path
                # Simple strategy: find qubit adjacent to both
                neighbors_p0 = {b for a, b in adjacent if a == p0}
                neighbors_p1 = {b for a, b in adjacent if a == p1}
                
                # Find intermediate
                intermediates = neighbors_p0 & neighbors_p1
                if intermediates:
                    intermediate = next(iter(intermediates))
                else:
                    # Just pick any neighbor of p0
                    intermediate = next(iter(neighbors_p0)) if neighbors_p0 else p1
                
                # Insert SWAP(p0, intermediate)
                transpiled.append({
                    'gate': 'SWAP',
                    'targets': [p0, intermediate],
                    'is_swap': True
                })
                
                # Update mapping
                l0 = inv_mapping.get(p0)
                li = inv_mapping.get(intermediate)
                if l0 is not None:
                    mapping[l0] = intermediate
                    inv_mapping[intermediate] = l0
                if li is not None:
                    mapping[li] = p0
                    inv_mapping[p0] = li
        
        transpiled.append(gate_op)
    
    return transpiled


def get_coupling_graph(device_name: str = "ibm_sherbrooke") -> List[Tuple[int, int]]:
    """
    Get coupling graph for common quantum devices.
    
    Args:
        device_name: Device identifier
        
    Returns:
        List of (q0, q1) edges
    """
    # Heavy-hex topology (common IBM layout)
    heavy_hex_edges = [
        # Row 1
        (0, 1), (1, 2), (1, 4), (2, 3),
        # Row 2
        (4, 5), (4, 7), (5, 6), (5, 8),
        # Row 3
        (7, 8), (7, 10), (8, 9), (8, 11),
        # Row 4
        (10, 11), (10, 13), (11, 12), (11, 14),
        # Row 5
        (13, 14), (13, 15), (14, 16),
        # Additional connections
        (3, 5), (6, 8), (9, 11), (12, 14),
    ]
    
    # Linear topology (common for ion trap)
    linear_edges = [(i, i + 1) for i in range(20)]
    
    # All-to-all (no restriction, for simulators)
    if device_name in ("simulator", "aws_sv1", "aws_tn1"):
        n = 20
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    elif device_name in ("ionq", "rigetti_line"):
        return linear_edges
    elif device_name.startswith("ibm"):
        return heavy_hex_edges
    else:
        return heavy_hex_edges
