"""
Reinforcement Learning for Quantum Circuit Compilation.

Trains a tabular Q-learning agent (NOT a deep Q-network / DQN)
to discover circuit compilation strategies: gate fusion, SWAP insertion,
layout optimization.
"""

import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict


class QuantumCompilerAI:
    """
    RL-based automatic quantum circuit compiler (tabular Q-learning).

    Trains a tabular Q-learning agent where:
    - State = circuit DAG representation (simplified as sequence of gates)
    - Actions = equivalent transformations (fuse, swap, reorder)
    - Reward = -final_circuit_depth (minimize depth)
    
    Args:
        n_qubits: Number of qubits
        learning_rate: Q-learning rate
        gamma: Discount factor
        epsilon: Exploration rate
    """
    
    def __init__(self,
                 n_qubits: int = 4,
                 learning_rate: float = 0.01,
                 gamma: float = 0.95,
                 epsilon: float = 0.1):
        self.n_qubits = n_qubits
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Simple Q-table (state_hash -> {action -> value})
        self._q_table: Dict[int, Dict[int, float]] = {}
        
        # Available actions
        self._actions = [
            "fuse_adjacent",      # Fuse adjacent single-qubit gates
            "commute_swap",       # Commute non-overlapping gates
            "cancel_inverse",     # Cancel H-H = I, X-X = I pairs
            "combine_rotations",  # Combine consecutive rotations
            "reorder_independent", # Reorder independent gates
        ]
        
        self._training_history: List[float] = []
    
    def _hash_state(self, circuit: List[dict]) -> int:
        """Hash a circuit state for Q-table lookup."""
        hash_str = str([(g.get('gate', ''), 
                        tuple(g.get('targets', [])),
                        g.get('control', -1))
                        for g in circuit])
        return hash(hash_str) % (2**31)
    
    def _get_valid_actions(self, circuit: List[dict]) -> List[int]:
        """Get indices of valid actions for the current circuit."""
        valid = []
        
        if len(circuit) >= 2:
            valid.append(0)  # fuse_adjacent always valid if 2+ gates
        
        if len(circuit) >= 3:
            valid.append(1)  # commute_swap possible
            valid.append(4)  # reorder_independent possible
        
        if len(circuit) >= 2:
            valid.append(2)  # cancel_inverse always possible to try
        
        if len(circuit) >= 2:
            valid.append(3)  # combine_rotations always valid to try
        
        return valid
    
    def _apply_action(self, circuit: List[dict], action_idx: int) -> List[dict]:
        """Apply a compilation transformation to the circuit."""
        action = self._actions[action_idx]
        
        if action == "fuse_adjacent":
            return self._fuse_adjacent(circuit)
        elif action == "commute_swap":
            return self._commute_swap(circuit)
        elif action == "cancel_inverse":
            return self._cancel_inverse(circuit)
        elif action == "combine_rotations":
            return self._combine_rotations(circuit)
        elif action == "reorder_independent":
            return self._reorder_independent(circuit)
        else:
            return circuit
    
    def _fuse_adjacent(self, circuit: List[dict]) -> List[dict]:
        """Fuse adjacent single-qubit gates on the same qubit."""
        if len(circuit) < 2:
            return circuit
        
        new_circuit = []
        i = 0
        while i < len(circuit):
            if i + 1 < len(circuit):
                g1 = circuit[i]
                g2 = circuit[i + 1]
                
                t1 = set(g1.get('targets', []))
                t2 = set(g2.get('targets', []))
                
                # Both are single-qubit on same qubit
                if len(t1) == 1 and t1 == t2:
                    # Fuse into a single mark
                    new_circuit.append({
                        'gate': 'FUSED_U3',
                        'targets': list(t1),
                        'params': {'fused': True},
                    })
                    i += 2
                    continue
            
            new_circuit.append(circuit[i])
            i += 1
        
        return new_circuit
    
    def _commute_swap(self, circuit: List[dict]) -> List[dict]:
        """Swap commuting gates (simple implementation)."""
        if len(circuit) < 2:
            return circuit
        
        # Randomly swap two adjacent gates if they commute
        i = random.randint(0, len(circuit) - 2)
        g1 = circuit[i]
        g2 = circuit[i + 1]
        
        t1 = set(g1.get('targets', []))
        t2 = set(g2.get('targets', []))
        
        # They commute if acting on disjoint qubits
        if t1.isdisjoint(t2):
            new_circuit = list(circuit)
            new_circuit[i], new_circuit[i + 1] = new_circuit[i + 1], new_circuit[i]
            return new_circuit
        
        return circuit
    
    def _cancel_inverse(self, circuit: List[dict]) -> List[dict]:
        """Cancel self-inverse gate pairs (H-H, X-X, etc.)."""
        self_inverse = {'H', 'X', 'Y', 'Z', 'CNOT'}
        
        new_circuit = []
        i = 0
        while i < len(circuit):
            if i + 1 < len(circuit):
                g1 = circuit[i]
                g2 = circuit[i + 1]
                
                if (g1.get('gate') == g2.get('gate') and
                    g1.get('gate') in self_inverse and
                    g1.get('targets') == g2.get('targets') and
                    g1.get('control') == g2.get('control')):
                    # Cancel the pair
                    i += 2
                    continue
            
            new_circuit.append(circuit[i])
            i += 1
        
        return new_circuit
    
    def _combine_rotations(self, circuit: List[dict]) -> List[dict]:
        """Combine consecutive rotation gates on same qubit."""
        if len(circuit) < 2:
            return circuit
        
        rotation_gates = {'RX', 'RY', 'RZ'}
        new_circuit = []
        i = 0
        
        while i < len(circuit):
            if i + 1 < len(circuit):
                g1 = circuit[i]
                g2 = circuit[i + 1]
                
                if (g1.get('gate') == g2.get('gate') and
                    g1.get('gate') in rotation_gates and
                    g1.get('targets') == g2.get('targets')):
                    
                    theta1 = g1.get('params', {}).get('theta', 0)
                    theta2 = g2.get('params', {}).get('theta', 0)
                    
                    new_circuit.append({
                        'gate': g1['gate'],
                        'targets': g1['targets'],
                        'params': {'theta': theta1 + theta2},
                    })
                    i += 2
                    continue
            
            new_circuit.append(circuit[i])
            i += 1
        
        return new_circuit
    
    def _reorder_independent(self, circuit: List[dict]) -> List[dict]:
        """Reorder independent gates to cluster similar operations."""
        if len(circuit) < 3:
            return circuit
        
        # Simple: swap two random gates
        i = random.randint(0, len(circuit) - 2)
        j = random.randint(0, len(circuit) - 2)
        if i != j:
            new_circuit = list(circuit)
            new_circuit[i], new_circuit[j] = new_circuit[j], new_circuit[i]
            return new_circuit
        
        return circuit
    
    def _circuit_depth(self, circuit: List[dict]) -> int:
        """Estimate circuit depth."""
        return len(circuit)
    
    def _get_action_value(self, state_hash: int, action_idx: int) -> float:
        """Get Q-value for state-action pair."""
        if state_hash not in self._q_table:
            self._q_table[state_hash] = {a: 0.0 for a in range(len(self._actions))}
        return self._q_table[state_hash].get(action_idx, 0.0)
    
    def _update_q(self, state_hash: int, action_idx: int, 
                  reward: float, next_state_hash: int):
        """Q-learning update."""
        current = self._get_action_value(state_hash, action_idx)
        
        next_values = [
            self._get_action_value(next_state_hash, a)
            for a in range(len(self._actions))
        ]
        max_next = max(next_values) if next_values else 0.0
        
        new_value = current + self.lr * (reward + self.gamma * max_next - current)
        
        if state_hash not in self._q_table:
            self._q_table[state_hash] = {}
        self._q_table[state_hash][action_idx] = new_value
    
    def train(self, initial_circuits: List[List[dict]], 
              episodes: int = 200, verbose: bool = True) -> List[float]:
        """
        Train the RL compiler agent.
        
        Args:
            initial_circuits: Training circuits
            episodes: Number of training episodes
            verbose: Print progress
            
        Returns:
            Training reward history
        """
        rewards_history = []
        
        for ep in range(episodes):
            # Randomly select a training circuit
            circuit = random.choice(initial_circuits)
            original_depth = self._circuit_depth(circuit)
            
            episode_reward = 0
            state = circuit
            state_hash = self._hash_state(state)
            
            for step in range(10):  # Max 10 actions per circuit
                valid_actions = self._get_valid_actions(state)
                
                # Epsilon-greedy action selection
                if random.random() < self.epsilon:
                    action_idx = random.choice(valid_actions)
                else:
                    q_values = [self._get_action_value(state_hash, a) 
                               for a in valid_actions]
                    action_idx = valid_actions[q_values.index(max(q_values))]
                
                # Apply action
                new_state = self._apply_action(state, action_idx)
                new_state_hash = self._hash_state(new_state)
                
                # Reward = depth reduction
                depth_before = self._circuit_depth(state)
                depth_after = self._circuit_depth(new_state)
                reward = depth_before - depth_after  # Positive if reduced
                
                # Update Q-table
                self._update_q(state_hash, action_idx, reward, new_state_hash)
                
                episode_reward += reward
                state = new_state
                state_hash = new_state_hash
                
                if reward <= 0:
                    break  # No more improvement
            
            rewards_history.append(episode_reward)
            
            if verbose and ep % 20 == 0:
                avg_reward = np.mean(rewards_history[-20:]) if rewards_history else 0
                print(f"  Episode {ep:4d}: reward = {episode_reward:.1f}, "
                      f"avg = {avg_reward:.3f}")
        
        self._training_history = rewards_history
        return rewards_history
    
    def compile(self, circuit: List[dict], max_steps: int = 10) -> List[dict]:
        """
        Compile a circuit using the trained agent.
        
        Args:
            circuit: Circuit gate sequence
            max_steps: Maximum compilation steps
            
        Returns:
            Optimized circuit
        """
        state = circuit
        original_depth = self._circuit_depth(state)
        
        for step in range(max_steps):
            state_hash = self._hash_state(state)
            valid_actions = self._get_valid_actions(state)
            
            if not valid_actions:
                break
            
            # Greedy selection (no exploration during inference)
            q_values = [self._get_action_value(state_hash, a) 
                       for a in valid_actions]
            best_action = valid_actions[np.argmax(q_values)]
            
            new_state = self._apply_action(state, best_action)
            
            depth_new = self._circuit_depth(new_state)
            if depth_new < self._circuit_depth(state):
                state = new_state
            else:
                break
        
        final_depth = self._circuit_depth(state)
        return state
    
    @property
    def history(self) -> List[float]:
        return self._training_history

    def save_q_table(self, path: str):
        """Persist Q-table to disk (JSON format)."""
        import json
        # Convert dict keys to strings for JSON serialization
        serializable = {
            str(k): {str(a): v for a, v in actions.items()}
            for k, actions in self._q_table.items()
        }
        with open(path, 'w') as f:
            json.dump(serializable, f)

    def load_q_table(self, path: str):
        """Load Q-table from disk (JSON format)."""
        import json
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self._q_table = {
                int(k): {int(a): v for a, v in actions.items()}
                for k, actions in data.items()
            }
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            pass  # Keep empty Q-table if file is missing or corrupted
