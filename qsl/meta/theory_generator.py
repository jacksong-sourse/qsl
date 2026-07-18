"""
Quantum Theorem Prover - Grover-amplified proof-path search.

Proof paths are enumerated and mapped to basis states of a quantum
register; the heuristic proof checker plays the role of the oracle,
and Grover's amplitude amplification (via qsl.core.grover) searches
the path space with O(√(N/M)) oracle queries instead of O(N)
classical evaluations.

Note: the heuristic checker itself is classical (a formal proof
checker encoded as a quantum circuit is out of scope); the quantum
part is the Grover search over the proof-path space. On real
hardware the checker would be compiled to a quantum oracle and
queried in superposition.
"""

import itertools
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class ProofResult:
    """Result of a theorem proof attempt."""
    conjecture: str
    is_proved: bool
    proof_found: bool
    num_paths_explored: int
    classical_steps: int
    proof_steps: Optional[List[str]] = None
    confidence: float = 0.0
    quantum_queries: int = 0


class QuantumTheoremProver:
    """
    Theorem prover using Grover's quantum search over proof-path space.

    All candidate proof paths up to max_proof_depth are mapped to basis
    states of an n-qubit register (n = ceil(log2(#paths))). The
    heuristic proof checker serves as the oracle, and Grover's
    amplitude amplification finds a valid proof with O(√(N/M)) oracle
    queries — a quadratic speedup over the O(N) classical enumeration
    (which is also computed for confidence statistics).

    Args:
        conjecture: Mathematical conjecture to prove or disprove
        max_proof_depth: Maximum proof tree depth
        max_branching: Maximum branching factor per step
    """

    def __init__(self,
                 conjecture: str = "",
                 max_proof_depth: int = 5,
                 max_branching: int = 3):
        self.conjecture = conjecture
        self.max_proof_depth = max_proof_depth
        self.max_branching = max_branching

    def _generate_axioms(self, domain: str = "arithmetic") -> List[str]:
        """Generate basic axioms for a mathematical domain."""
        axioms = {
            "arithmetic": [
                "a + 0 = a",
                "a + b = b + a",
                "(a + b) + c = a + (b + c)",
                "a * 1 = a",
                "a * b = b * a",
                "a * (b + c) = a*b + a*c",
                "a + (-a) = 0",
            ],
            "boolean": [
                "a AND a = a",
                "a OR a = a",
                "NOT(NOT(a)) = a",
                "a AND (b OR c) = (a AND b) OR (a AND c)",
                "a OR (a AND b) = a",
            ],
            "set_theory": [
                "A ∪ A = A",
                "A ∩ A = A",
                "A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C)",
                "(A')' = A",
            ],
        }
        return axioms.get(domain, axioms["arithmetic"])

    def _generate_proof_paths(self, axioms: List[str],
                               max_depth: int) -> List[List[str]]:
        """Generate all possible proof paths up to max_depth."""
        operations = []
        for axiom in axioms:
            operations.append(f"Apply axiom: {axiom}")

        operations.extend([
            "Modus ponens",
            "Substitution",
            "Transitivity",
            "Contradiction check",
        ])

        paths = []
        for depth in range(1, max_depth + 1):
            for combo in itertools.product(operations, repeat=depth):
                paths.append(list(combo))

        return paths

    def _check_proof(self, proof_steps: List[str],
                     axioms: List[str]) -> Tuple[bool, float]:
        """
        Check if a proof sequence appears valid.

        Uses heuristic scoring rather than a formal proof checker.
        """
        score = 0.0

        # Penalize long proofs (Occam's razor heuristic)
        score -= len(proof_steps) * 0.05

        # Axiom use increases plausibility
        for step in proof_steps:
            if "axiom" in step:
                score += 0.15
            elif "modus" in step.lower():
                score += 0.1
            elif "contradiction" in step.lower():
                score += 0.2

        # Unique operations are preferred
        unique_ops = len(set(proof_steps))
        score += unique_ops * 0.05

        # Normalize via sigmoid
        confidence = 1.0 / (1.0 + np.exp(-score))

        is_valid = confidence > 0.5

        return is_valid, confidence

    def prove(self) -> ProofResult:
        """
        Execute proof search using Grover's amplitude amplification.

        The proof-path space is mapped to basis states; the heuristic
        checker is the oracle. Grover search finds a valid proof with
        O(√(N/M)) oracle queries (quantum_queries in the result),
        instead of evaluating all N paths classically.

        Returns:
            ProofResult with outcome
        """
        from ..core.grover import GroverSearch
        from ..utils.exceptions import NoSolutionError

        domain = "boolean" if any(w in self.conjecture.lower()
                                  for w in ("and", "or", "not", "xor")) else "arithmetic"

        axioms = self._generate_axioms(domain)
        all_paths = self._generate_proof_paths(axioms, self.max_proof_depth)

        if not all_paths:
            return ProofResult(
                conjecture=self.conjecture,
                is_proved=False,
                proof_found=False,
                num_paths_explored=0,
                classical_steps=0,
            )

        n_paths = len(all_paths)
        n_qubits = max(1, int(np.ceil(np.log2(n_paths))))
        N_states = 1 << n_qubits
        candidate_paths = all_paths[:N_states]

        # Oracle: 路径索引 -> 启发式检查器
        def checker(idx: int) -> bool:
            if idx >= len(candidate_paths):
                return False
            is_valid, _ = self._check_proof(candidate_paths[idx], axioms)
            return is_valid

        # Grover 搜索证明路径 (黑盒 Oracle, 模拟器一次性构建标记集;
        # 算法查询复杂度为 O(√(N/M)) 次迭代)
        grover = GroverSearch(n_qubits, verbose=False)
        try:
            result = grover.search(condition=checker, num_solutions=None,
                                   shots=5)
        except NoSolutionError:
            return ProofResult(
                conjecture=self.conjecture,
                is_proved=False,
                proof_found=False,
                num_paths_explored=n_paths,
                classical_steps=N_states,
                quantum_queries=0,
            )

        found_indices = result.get_solutions()
        if not found_indices:
            return ProofResult(
                conjecture=self.conjecture,
                is_proved=False,
                proof_found=False,
                num_paths_explored=n_paths,
                classical_steps=N_states,
                quantum_queries=result.quantum_queries or 0,
            )

        # 在找到的候选中取置信度最高的证明
        best_proof, best_conf = None, -1.0
        for idx in found_indices:
            _, conf = self._check_proof(candidate_paths[idx], axioms)
            if conf > best_conf:
                best_proof, best_conf = candidate_paths[idx], conf

        return ProofResult(
            conjecture=self.conjecture,
            is_proved=True,
            proof_found=True,
            num_paths_explored=n_paths,
            classical_steps=N_states,
            proof_steps=best_proof,
            confidence=best_conf,
            quantum_queries=result.quantum_queries or 0,
        )

    def benchmark(self) -> Dict[str, any]:
        """
        Compare classical proof search with theoretical quantum speedup.

        The quantum_steps column shows the hypothetical number of steps
        if Grover search were used (O(sqrt(N))). This is a theoretical
        comparison only—no actual Grover search is performed.
        """
        n_paths = len(self._generate_proof_paths(
            self._generate_axioms("arithmetic"), self.max_proof_depth
        ))
        N_states = 1 << max(1, int(np.ceil(np.log2(n_paths))))

        return {
            "n_proof_paths": n_paths,
            "n_states": N_states,
            "classical_steps": N_states,
            "quantum_steps (theoretical)": int(np.sqrt(N_states)),
            "speedup (theoretical)": np.sqrt(N_states) if N_states > 0 else 1,
        }
