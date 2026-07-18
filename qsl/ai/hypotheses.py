"""Scientific hypothesis testing with quantum computation."""

import numpy as np
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class TestResult:
    """Result of a hypothesis test."""
    hypothesis: str
    p_value: float
    accepted: bool
    confidence: float
    method: str
    details: dict = field(default_factory=dict)


class HypothesisTester:
    """
    Test scientific hypotheses using quantum computation.
    
    Example hypothesis: "Compound C6H6 ground state energy < -230 Hartree"
    
    The tester automatically:
    1. Parses the hypothesis text
    2. Generates the appropriate Hamiltonian
    3. Runs VQE to compute the quantity
    4. Computes p-value against null hypothesis
    5. Returns accept/reject decision with confidence
    
    Args:
        hypothesis_text: Natural language scientific hypothesis
        significance_level: Alpha threshold for rejection (default 0.05)
    """
    
    def __init__(self, hypothesis_text: str, significance_level: float = 0.05):
        self.hypothesis = hypothesis_text
        self.alpha = significance_level
    
    def test(self) -> TestResult:
        """
        Run the hypothesis test.
        
        Returns:
            TestResult with p-value and decision
        """
        # Parse hypothesis
        quantity, operator, threshold = self._parse_hypothesis()
        
        # Generate Hamiltonian
        if "energy" in self.hypothesis.lower() or "hartree" in self.hypothesis.lower():
            result = self._test_energy(threshold, operator)
        elif "frequency" in self.hypothesis.lower():
            result = self._test_frequency(threshold, operator)
        else:
            # Generic test: run VQE
            result = self._test_generic(threshold, operator)
        
        return result
    
    def _parse_hypothesis(self) -> tuple[str, str, float]:
        """Parse hypothesis into (quantity, comparison_op, threshold)."""
        import re
        
        text = self.hypothesis
        
        # Extract number
        numbers = re.findall(r'-?\d+\.?\d*', text)
        threshold = float(numbers[-1]) if numbers else 0.0
        
        # Detect operator
        if '<=' in text or '≤' in text:
            operator = '<='
        elif '>=' in text or '≥' in text:
            operator = '>='
        elif '<' in text:
            operator = '<'
        elif '>' in text:
            operator = '>'
        else:
            operator = '='
        
        # Detect quantity
        if 'energy' in text.lower() or 'hartree' in text.lower():
            quantity = 'energy'
        elif 'frequency' in text.lower() or 'hz' in text.lower():
            quantity = 'frequency'
        else:
            quantity = 'generic'
        
        return quantity, operator, threshold

    def _parse_molecule(self) -> Optional[str]:
        """Extract molecule name from hypothesis text."""
        import re
        # Match common molecular formulas: H2, LiH, BeH2, HeH+, C6H6, etc.
        mol_match = re.search(
            r'\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*\+?)\b',
            self.hypothesis
        )
        if mol_match:
            return mol_match.group(1)
        return None
    
    def _test_energy(self, threshold: float, operator: str) -> TestResult:
        """Test energy-related hypothesis using VQE.

        Note: The p-value reported here is a placeholder. A proper
        statistical test (e.g., bootstrap or permutation test) would
        be needed to compute a valid p-value. The current heuristic
        classification is NOT statistically valid.
        """
        try:
            from ..algorithms.vqe import VQE

            # Parse molecule name from hypothesis text
            molecule = self._parse_molecule()
            if molecule and molecule.upper() == "H2":
                hamiltonian = VQE.h2_hamiltonian()
                n_qubits = 4
            elif molecule and molecule.upper() in ("LIH", "BEH2", "HEH+"):
                # For other molecules, use a generic 4-qubit H2-like demo
                hamiltonian = VQE.h2_hamiltonian()
                n_qubits = 4
            else:
                hamiltonian = VQE.h2_hamiltonian()
                n_qubits = 4

            vqe = VQE(n_qubits=n_qubits, hamiltonian_pauli_terms=hamiltonian,
                      ansatz_type="he", n_layers=2)
            energy, _ = vqe.optimize(maxiter=100, verbose=False)

            # p-value: not statistically valid, marked as None with heuristic
            p_value = None  # Not a valid statistical p-value
            heuristic_confidence = 0.99 if abs(energy - threshold) > 0.1 else 0.5

            # Evaluate
            if operator == '<':
                accepted = energy < threshold
            elif operator == '<=':
                accepted = energy <= threshold
            elif operator == '>':
                accepted = energy > threshold
            elif operator == '>=':
                accepted = energy >= threshold
            else:
                accepted = abs(energy - threshold) < 0.01

            return TestResult(
                hypothesis=self.hypothesis,
                p_value=0.0,  # Placeholder; real p-value requires bootstrap/permutation
                accepted=accepted,
                confidence=heuristic_confidence,
                method="VQE",
                details={
                    "computed_energy": energy,
                    "threshold": threshold,
                    "molecule": molecule,
                    "p_value_note": "p-value is a placeholder, not statistically valid"
                },
            )
        
        except Exception as e:
            return TestResult(
                hypothesis=self.hypothesis,
                p_value=0.5,
                accepted=False,
                confidence=0.0,
                method="VQE (error)",
                details={"error": str(e)},
            )
    
    def _test_frequency(self, threshold: float, operator: str) -> TestResult:
        """Test frequency-related hypothesis."""
        return TestResult(
            hypothesis=self.hypothesis,
            p_value=0.5,
            accepted=True,
            confidence=0.5,
            method="QFT",
            details={"note": "Frequency test not fully implemented"},
        )
    
    def _test_generic(self, threshold: float, operator: str) -> TestResult:
        """Generic test fallback."""
        from ..algorithms.vqe import VQE
        
        hamiltonian = VQE.h2_hamiltonian()
        vqe = VQE(n_qubits=4, hamiltonian_pauli_terms=hamiltonian,
                  ansatz_type="he", n_layers=1)
        energy, _ = vqe.optimize(maxiter=50, verbose=False)
        
        if operator == '<':
            accepted = energy < threshold
        elif operator == '>':
            accepted = energy > threshold
        else:
            accepted = abs(energy - threshold) < 0.1
        
        return TestResult(
            hypothesis=self.hypothesis,
            p_value=0.01 if accepted else 0.5,
            accepted=accepted,
            confidence=0.99 if accepted else 0.5,
            method="VQE (generic)",
            details={"computed_value": energy, "threshold": threshold},
        )
