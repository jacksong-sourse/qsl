"""
Genetic Programming for Quantum Circuit Discovery.

*** WARNING: DEMONSTRATION ONLY — requires deap, ray; conceptual placeholder ***

Uses evolutionary algorithms to automatically discover new quantum circuit
topologies that perform better on benchmark problems.
"""

import random
import numpy as np
from typing import List, Tuple, Callable
from dataclasses import dataclass


# Gate vocabulary for evolution
GATE_VOCAB = [
    {'gate': 'H', 'n_qubits': 1},
    {'gate': 'X', 'n_qubits': 1},
    {'gate': 'Y', 'n_qubits': 1},
    {'gate': 'Z', 'n_qubits': 1},
    {'gate': 'S', 'n_qubits': 1},
    {'gate': 'T', 'n_qubits': 1},
    {'gate': 'RX', 'n_qubits': 1},
    {'gate': 'RY', 'n_qubits': 1},
    {'gate': 'RZ', 'n_qubits': 1},
    {'gate': 'CNOT', 'n_qubits': 2},
    {'gate': 'CZ', 'n_qubits': 2},
    {'gate': 'SWAP', 'n_qubits': 2},
    {'gate': 'TOFFOLI', 'n_qubits': 3},
]


@dataclass
class CircuitGenome:
    """A genome representing a quantum circuit."""
    gates: List[dict]
    n_qubits: int
    fitness: float = 0.0
    
    @staticmethod
    def random(n_qubits: int, max_gates: int = 10) -> 'CircuitGenome':
        """Generate a random circuit genome."""
        n_gates = random.randint(1, max_gates)
        gates = []
        for _ in range(n_gates):
            available_gates = [g for g in GATE_VOCAB if g['n_qubits'] <= n_qubits]
            gate_info = random.choice(available_gates)
            if gate_info['n_qubits'] == 1:
                targets = [random.randint(0, n_qubits - 1)]
                gate = {'gate': gate_info['gate'], 'targets': targets}
                if gate_info['gate'] in ('RX', 'RY', 'RZ'):
                    gate['params'] = {'theta': random.uniform(0, 2 * np.pi)}
            elif gate_info['n_qubits'] == 2:
                q0, q1 = random.sample(range(n_qubits), 2)
                if gate_info['gate'] == 'CNOT':
                    gate = {'gate': 'CNOT', 'control': q0, 'target': q1}
                elif gate_info['gate'] == 'CZ':
                    gate = {'gate': 'CZ', 'control': q0, 'target': q1}
                else:
                    gate = {'gate': gate_info['gate'], 'targets': [q0, q1]}
            else:  # 3-qubit gates
                q0, q1, q2 = random.sample(range(n_qubits), 3)
                gate = {'gate': gate_info['gate'], 'targets': [q0, q1, q2]}
            
            gates.append(gate)
        
        return CircuitGenome(gates=gates, n_qubits=n_qubits)
    
    def to_statevector_circuit(self) -> Callable:
        """Convert genome to a callable circuit function for QuantumState."""
        def circuit_fn(state):
            for g in self.gates:
                gate_type = g['gate']
                targets = g.get('targets', [])
                
                if gate_type == 'H':
                    for t in targets:
                        state.h(t)
                elif gate_type == 'X':
                    for t in targets:
                        state.x(t)
                elif gate_type == 'Y':
                    for t in targets:
                        state.y(t)
                elif gate_type == 'Z':
                    for t in targets:
                        state.z(t)
                elif gate_type == 'S':
                    for t in targets:
                        state.s(t)
                elif gate_type == 'T':
                    for t in targets:
                        state.t(t)
                elif gate_type == 'CNOT':
                    state.cnot(g['control'], g['target'])
                elif gate_type == 'CZ':
                    state.cz(g['control'], g['target'])
                elif gate_type == 'SWAP':
                    state.swap(targets[0], targets[1])
                elif gate_type == 'TOFFOLI':
                    state.toffoli(targets[0], targets[1], targets[2])
            return state
        return circuit_fn
    
    @property
    def circuit_depth(self) -> int:
        """Estimate circuit depth."""
        # Simplify: count as number of layers after greedy scheduling
        return len(self.gates)
    
    def copy(self) -> 'CircuitGenome':
        """Deep copy the genome."""
        gates_copy = [{k: v.copy() if isinstance(v, list) else v for k, v in g.items()} for g in self.gates]
        return CircuitGenome(gates=gates_copy, n_qubits=self.n_qubits, fitness=self.fitness)


class AlgorithmSearcher:
    """
    Evolutionary algorithm-based quantum circuit topology discovery.
    
    Uses genetic programming (GP) to automatically discover new quantum circuits:
    1. Randomly generate initial population of gate sequences
    2. Evaluate fitness on benchmark problems
    3. Crossover + Mutate -> Select top performers
    4. Repeat for N generations
    
    Args:
        n_qubits: Number of qubits
        population_size: Size of population per generation
        generations: Number of evolution generations
        mutation_rate: Probability of mutation per gene
        crossover_rate: Probability of crossover
    """
    
    def __init__(self,
                 n_qubits: int = 3,
                 population_size: int = 20,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.5,
                 use_simulation_fitness: bool = False):
        import warnings
        if n_qubits > 6 and use_simulation_fitness:
            warnings.warn(
                f"AlgorithmSearcher with n_qubits={n_qubits} > 6 and "
                f"use_simulation_fitness=True is extremely slow. "
                f"Forcing cap at n_qubits=6.",
                RuntimeWarning, stacklevel=2
            )
            n_qubits = 6
        self.n_qubits = n_qubits
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        # 默认使用轻量评估 (无完整态模拟); 设为 True 可退回旧的
        # 基于 QuantumState 完整模拟的 fitness (慢, 仅适合小 n)
        self.use_simulation_fitness = use_simulation_fitness
        self._best_genome: CircuitGenome = None
        self._history: List[float] = []

    # 纠缠门类型 (能产生纠缠的多量子比特门)
    _ENTANGLING_GATES = frozenset(("CNOT", "CZ", "TOFFOLI"))

    def _fitness(self, genome: CircuitGenome) -> float:
        """
        轻量电路适应度评估 (默认路径, 无量子态模拟)。

        通过电路结构分析估计其产生纠缠的能力:
        1. 纠缠门占比与位置分布 (CNOT/CZ/TOFFOLI)
        2. 量子比特覆盖率 (多少比特被纠缠门连接)
        3. 门多样性 (单比特门类型丰富度)
        4. 深度惩罚 (过深电路在 NISQ 上噪声大)

        use_simulation_fitness=True 时退回旧的完整态模拟评估
        (每次 fitness 一次 O(2^n) 模拟, 仅适合 n <= 6)。
        """
        if self.use_simulation_fitness:
            return self._fitness_by_simulation(genome)
        return self._fitness_structural(genome)

    def _fitness_structural(self, genome: CircuitGenome) -> float:
        """基于电路结构分析的轻量 fitness, 复杂度 O(n_gates)。"""
        gates = genome.gates
        if not gates:
            return 0.0

        n_gates = len(gates)
        entangling = [g for g in gates
                      if g['gate'] in self._ENTANGLING_GATES]

        # 1. 纠缠门占比 (目标区间 20%-60%)
        ent_ratio = len(entangling) / n_gates
        ent_score = 1.0 - abs(ent_ratio - 0.4) / 0.4
        ent_score = max(0.0, min(1.0, ent_score))

        # 2. 纠缠门覆盖的量子比特比例
        covered = set()
        for g in entangling:
            if 'control' in g:
                covered.add(g['control'])
                covered.add(g['target'])
            for t in g.get('targets', []):
                covered.add(t)
        coverage = len(covered) / self.n_qubits if self.n_qubits else 0.0

        # 3. 单比特门多样性 (旋转门/H/S/T 种类)
        single_types = {g['gate'] for g in gates
                        if g['gate'] not in self._ENTANGLING_GATES
                        and g['gate'] != 'SWAP'}
        diversity = min(1.0, len(single_types) / 4.0)

        # 4. 深度惩罚 (超过 20 门开始惩罚)
        depth_penalty = min(1.0, genome.circuit_depth / 20.0)

        fitness = (0.35 * ent_score + 0.30 * coverage
                   + 0.15 * diversity + 0.20 * (1.0 - depth_penalty))
        return float(max(0.0, min(1.0, fitness)))

    def _fitness_by_simulation(self, genome: CircuitGenome) -> float:
        """
        基于完整量子态模拟的 fitness (旧路径, 慢)。

        每次评估运行一次 O(2^n) QuantumState 模拟, 仅适合 n <= 6。
        """
        try:
            from ..core.state import QuantumState

            state = QuantumState(self.n_qubits)
            circuit_fn = genome.to_statevector_circuit()
            circuit_fn(state)

            # Fitness metrics:
            # 1. Normalization preserved
            norm_score = 1.0 if state.check_normalization() else 0.0

            # 2. Entanglement: measure by non-separability of probabilities
            probs = state.probabilities()
            entropy = -sum(p * np.log(p + 1e-12) for p in probs if p > 0)
            max_entropy = np.log(len(probs))
            entropy_score = entropy / (max_entropy + 1e-12)

            # 3. Circuit depth penalty (shorter is better)
            depth_penalty = min(1.0, genome.circuit_depth / 20.0)

            # Combined score
            fitness = norm_score * 0.3 + entropy_score * 0.5 + (1 - depth_penalty) * 0.2

            return fitness

        except Exception:
            return 0.0
    
    def _crossover(self, parent1: CircuitGenome, 
                   parent2: CircuitGenome) -> CircuitGenome:
        """Single-point crossover between two circuits."""
        if min(len(parent1.gates), len(parent2.gates)) < 2:
            return parent1.copy()
        
        # Choose crossover points
        cut1 = random.randint(1, len(parent1.gates) - 1)
        cut2 = random.randint(1, len(parent2.gates) - 1)
        
        child_gates = parent1.gates[:cut1] + parent2.gates[cut2:]
        
        return CircuitGenome(gates=child_gates, n_qubits=self.n_qubits)
    
    def _mutate(self, genome: CircuitGenome) -> CircuitGenome:
        """Random mutation of a circuit."""
        new_genome = genome.copy()
        
        for i in range(len(new_genome.gates)):
            if random.random() < self.mutation_rate:
                gate = new_genome.gates[i]
                gate_type = gate['gate']
                
                if gate_type in ('H', 'X', 'Y', 'Z', 'S', 'T'):
                    gate['targets'] = [random.randint(0, self.n_qubits - 1)]
                elif gate_type in ('RX', 'RY', 'RZ'):
                    gate['params'] = {'theta': random.uniform(0, 2 * np.pi)}
                elif gate_type in ('CNOT', 'CZ', 'SWAP'):
                    a, b = random.sample(range(self.n_qubits), 2)
                    if 'control' in gate:
                        gate['control'] = a
                        gate['target'] = b
                    else:
                        gate['targets'] = [a, b]
        
        # Random gate insertion
        if random.random() < self.mutation_rate:
            new_gate = CircuitGenome.random(self.n_qubits, 1).gates[0]
            pos = random.randint(0, len(new_genome.gates))
            new_genome.gates.insert(pos, new_gate)
        
        # Random gate deletion
        if len(new_genome.gates) > 2 and random.random() < self.mutation_rate:
            del new_genome.gates[random.randint(0, len(new_genome.gates) - 1)]
        
        return new_genome
    
    def search(self, verbose: bool = True) -> CircuitGenome:
        """
        Run the evolutionary search for optimal quantum circuits.
        
        Args:
            verbose: Print progress
            
        Returns:
            Best circuit genome found
        """
        # Initialize random population
        population = [
            CircuitGenome.random(self.n_qubits)
            for _ in range(self.population_size)
        ]
        
        # Evaluate initial fitness
        for genome in population:
            genome.fitness = self._fitness(genome)
        
        best_overall = max(population, key=lambda g: g.fitness)
        
        for gen in range(self.generations):
            # Selection (tournament)
            new_population = []
            
            # Elitism: keep top 2
            population.sort(key=lambda g: -g.fitness)
            new_population.append(population[0].copy())
            new_population.append(population[1].copy())
            
            while len(new_population) < self.population_size:
                # Tournament selection
                tournament = random.sample(population, min(3, len(population)))
                parent1 = max(tournament, key=lambda g: g.fitness)
                tournament = random.sample(population, min(3, len(population)))
                parent2 = max(tournament, key=lambda g: g.fitness)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation
                child = self._mutate(child)
                
                # Evaluate
                child.fitness = self._fitness(child)
                new_population.append(child)
            
            population = new_population[:self.population_size]
            
            # Track best
            gen_best = max(population, key=lambda g: g.fitness)
            if gen_best.fitness > best_overall.fitness:
                best_overall = gen_best.copy()
            
            self._history.append(gen_best.fitness)
            
            if verbose and gen % 10 == 0:
                print(f"  Gen {gen:4d}: best fitness = {gen_best.fitness:.4f}, "
                      f"depth = {gen_best.circuit_depth}")
        
        self._best_genome = best_overall
        
        if verbose:
            print(f"\n  Evolution complete: {self.generations} generations")
            print(f"  Best circuit depth: {best_overall.circuit_depth}")
            print(f"  Best fitness: {best_overall.fitness:.4f}")
            print(f"  Gate sequence: {[g['gate'] for g in best_overall.gates]}")
        
        return best_overall
    
    @property
    def best_genome(self) -> CircuitGenome:
        return self._best_genome
    
    @property
    def history(self) -> List[float]:
        return self._history
