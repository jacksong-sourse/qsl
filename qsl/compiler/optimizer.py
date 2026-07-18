"""Circuit optimization: gate fusion, commutation, depth reduction."""

import numpy as np


def gate_fusion(gate_sequence: list) -> list:
    """
    Merge consecutive single-qubit gates into equivalent U3 gates.

    Adjacent gates on the same qubit can be combined:
    U2 * U1 = U_combined, reducing gate count and circuit depth.

    Computes the actual matrix product of consecutive single-qubit gates
    on the same qubit and replaces them with a single FUSED_U3 gate.

    Args:
        gate_sequence: List of {'gate': str, 'targets': [...], 'params': {...}}

    Returns:
        Optimized gate sequence
    """
    if not gate_sequence:
        return []

    # Gate matrix representations in computational basis
    def _get_1q_matrix(gate: dict) -> np.ndarray:
        """Return 2x2 unitary matrix for a single-qubit gate."""
        gtype = gate.get('gate', '')
        # Identity
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        S = np.array([[1, 0], [0, 1j]], dtype=complex)
        T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

        gate_map = {'X': X, 'Y': Y, 'Z': Z, 'H': H, 'S': S, 'T': T,
                    'I': I, 'ID': I}
        if gtype in gate_map:
            return gate_map[gtype]
        elif gtype == 'RX':
            theta = gate.get('params', {}).get('theta', 0)
            c = np.cos(theta / 2); s = np.sin(theta / 2)
            return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
        elif gtype == 'RY':
            theta = gate.get('params', {}).get('theta', 0)
            c = np.cos(theta / 2); s = np.sin(theta / 2)
            return np.array([[c, -s], [s, c]], dtype=complex)
        elif gtype == 'RZ':
            phi = gate.get('params', {}).get('phi', 0)
            return np.array([[np.exp(-1j * phi / 2), 0],
                            [0, np.exp(1j * phi / 2)]], dtype=complex)
        elif gtype == 'FUSED_U3':
            return I  # already fused
        return I

    optimized = []
    i = 0
    while i < len(gate_sequence):
        gate = gate_sequence[i]
        targets = gate.get('targets', [])

        # Multi-qubit gates: flush any pending single-qubit and output directly
        if len(targets) > 1 or 'control' in gate:
            optimized.append(gate)
            i += 1
            continue

        if not targets:
            optimized.append(gate)
            i += 1
            continue

        q = targets[0]

        # Collect consecutive single-qubit gates on the same qubit q
        fused_matrix = np.eye(2, dtype=complex)
        j = i
        while j < len(gate_sequence):
            g = gate_sequence[j]
            g_targets = g.get('targets', [])
            if len(g_targets) == 1 and g_targets[0] == q and 'control' not in g:
                # Multiply: U_new = U_j * U_j-1 * ... * U_i (right-to-left convention)
                fused_matrix = _get_1q_matrix(g) @ fused_matrix
                j += 1
            else:
                break

        if j > i + 1:
            # Emit a single FUSED_U3 with the combined matrix
            optimized.append({
                'gate': 'FUSED_U3',
                'targets': [q],
                'params': {'matrix': fused_matrix.tolist(), 'fused': True}
            })
        else:
            optimized.append(gate)
        i = j

    return optimized


def commutation_optimization(gate_sequence: list) -> list:
    """
    Reorder commuting gates to reduce circuit depth.

    Gates acting on disjoint qubit sets commute. This optimization
    collects all commuting gates that can be moved adjacent to each
    other and performs a full topological reordering.

    Strategy: For each gate at position i, find all gates j > i that
    act on disjoint qubits. Collect the full set, then move them all
    to be next to gate i in a single pass.

    Args:
        gate_sequence: List of gate operations

    Returns:
        Reordered gate sequence
    """
    if len(gate_sequence) < 2:
        return gate_sequence

    result = []
    remaining = list(gate_sequence)

    while remaining:
        gate = remaining.pop(0)
        targets_i = set(gate.get('targets', []))
        result.append(gate)

        # Collect all commuting gates that can be moved adjacent
        commuted = []
        next_remaining = []
        for other in remaining:
            targets_j = set(other.get('targets', []))
            if targets_i.isdisjoint(targets_j):
                commuted.append(other)
            else:
                next_remaining.append(other)

        result.extend(commuted)
        remaining = next_remaining

    return result


def depth_reduction(gate_sequence: list) -> tuple:
    """
    Minimize circuit depth by parallelizing commuting gates.

    Analyzes gate dependencies and schedules gates in layers
    where all gates in a layer act on disjoint qubits.

    Args:
        gate_sequence: List of gate operations

    Returns:
        (optimized_sequence, original_depth, optimized_depth)
    """
    if not gate_sequence:
        return gate_sequence, 0, 0

    original_depth = len(gate_sequence)

    layers = []
    remaining = list(gate_sequence)

    while remaining:
        layer = []
        used_qubits = set()
        next_remaining = []

        for gate in remaining:
            targets = set(gate.get('targets', []))
            controls = {gate.get('control')} if 'control' in gate else set()
            all_qubits = targets | controls

            if all_qubits.isdisjoint(used_qubits):
                layer.append(gate)
                used_qubits.update(all_qubits)
            else:
                next_remaining.append(gate)

        layers.append(layer)
        remaining = next_remaining

    optimized = []
    for layer in layers:
        optimized.extend(layer)

    optimized_depth = 0
    if not optimized:
        return optimized, original_depth, optimized_depth

    qubit_last_layer: dict = {}
    max_layer = 0
    for gate in optimized:
        targets = set(gate.get('targets', []))
        controls = {gate.get('control')} if 'control' in gate else set()
        all_qubits = targets | controls

        pred_depth = 0
        for q in all_qubits:
            pred_depth = max(pred_depth, qubit_last_layer.get(q, -1) + 1)

        current_layer = pred_depth
        for q in all_qubits:
            qubit_last_layer[q] = current_layer

        max_layer = max(max_layer, current_layer)

    optimized_depth = max_layer + 1

    return optimized, original_depth, optimized_depth
