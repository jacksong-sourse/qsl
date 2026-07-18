"""
Crypto Analysis Pipeline - Quantum cryptographic attack simulation.

*** WARNING: DEMONSTRATION ONLY — toy-scale attacks, not real cryptanalysis ***
Uses Shor for RSA factoring and Grover for symmetric key search.
Compares quantum vs classical computation complexity.
"""

import time
import math
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class CryptoResult:
    """Result of a cryptographic analysis."""
    algorithm: str  # "RSA" or "AES" or "ECDSA"
    key_size: int  # in bits
    classical_ops: float  # estimated operations (Big-O)
    quantum_ops: float  # estimated quantum operations (Big-O)
    speedup: float  # theoretical speedup factor
    is_vulnerable: bool
    plaintext: Optional[str] = None
    details: Optional[str] = None


class CryptoAnalysisPipeline:
    """
    Quantum cryptographic analysis pipeline.

    Provides theoretical complexity analysis comparing classical vs
    quantum attack complexity using standard Big-O estimates.

    NOTE: All numbers are THEORETICAL estimates based on asymptotic
    complexity, NOT actual wall-clock times:
    - RSA: classical O(exp((64/9*n)^(1/3))) vs quantum O(n^3) with Shor
    - Symmetric: classical O(2^n) vs quantum O(2^(n/2)) with Grover
    Actual gate counts, error correction overhead, and hardware
    constraints are not included in these estimates.

    Args:
        cipher_type: "rsa", "aes", or "auto" to auto-detect
        public_key_modulus: RSA modulus N (for RSA)
        key_size: Key size in bits (for symmetric)
    """

    def __init__(self,
                 cipher_type: str = "auto",
                 public_key_modulus: Optional[int] = None,
                 key_size: int = 128,
                 target_key: Optional[int] = None):
        self.cipher_type = cipher_type
        self.N = public_key_modulus
        self.key_size = key_size
        self.target_key = target_key

    def analyze(self) -> CryptoResult:
        """
        Run cryptographic complexity analysis.

        Returns:
            CryptoResult with theoretical complexity comparison
        """
        if self.cipher_type == "rsa" or (self.cipher_type == "auto" and self.N):
            return self._analyze_rsa()
        else:
            return self._analyze_symmetric()

    def _analyze_rsa(self) -> CryptoResult:
        """Analyze RSA factoring complexity (theoretical)."""
        N = self.N or 15
        key_size = N.bit_length()

        # Classical: General Number Field Sieve
        # O(exp((64/9 * n)^(1/3) * log(n)^(2/3)))
        # Simplified: ~2^(key_size/4) operations
        classical_ops = 2 ** (key_size / 4)

        # Quantum Shor: O(log(N)^3) gate complexity
        quantum_ops = math.log2(N) ** 3

        speedup = classical_ops / max(quantum_ops, 1e-12)
        is_vulnerable = key_size <= 2048  # Shor can handle 2048-bit with enough qubits

        # Try actual Shor for small N (demonstration)
        plaintext = None
        if N <= 10000:
            try:
                from ..algorithms.shor import ShorSolver
                solver = ShorSolver(N)
                factors = solver.factor()
                plaintext = f"Factors: {factors}"
            except Exception:
                pass

        return CryptoResult(
            algorithm="RSA",
            key_size=key_size,
            classical_ops=classical_ops,
            quantum_ops=quantum_ops,
            speedup=speedup,
            is_vulnerable=is_vulnerable,
            plaintext=plaintext,
            details=(f"RSA-{key_size} (N={N}): Shor provides "
                     f"exponential speedup for factoring via QFT "
                     f"phase estimation. [THEORETICAL - Big-O estimates only]"),
        )

    def _analyze_symmetric(self) -> CryptoResult:
        """Analyze symmetric cipher complexity (theoretical)."""
        n = self.key_size

        # Classical: O(2^n) brute force
        classical_ops = 2 ** n

        # Quantum Grover: O(2^(n/2)) = O(sqrt(N))
        quantum_ops = 2 ** (n / 2)

        speedup = classical_ops / max(quantum_ops, 1e-12)
        is_vulnerable = n < 256  # Grover effectively halves security

        # Try Grover for small key sizes (demonstration)
        plaintext = None
        if n <= 8:
            try:
                from ..core.grover import GroverSearch
                target = self.target_key if self.target_key is not None else 42 % (2 ** n)
                search = GroverSearch(n, verbose=False)
                result = search.search(condition=lambda x: x == target, shots=1)
                solutions = result.get_solutions()
                if solutions:
                    plaintext = f"Found key: {solutions[0]}"
            except Exception:
                pass

        return CryptoResult(
            algorithm=f"SYMMETRIC-{n}",
            key_size=n,
            classical_ops=classical_ops,
            quantum_ops=quantum_ops,
            speedup=speedup,
            is_vulnerable=is_vulnerable,
            plaintext=plaintext,
            details=(f"AES-{n}: Grover provides quadratic speedup "
                     f"(sqrt(N) = 2^(n/2) vs 2^n). "
                     f"[THEORETICAL - Big-O estimates only, "
                     f"actual gate counts are much higher]"),
        )

    @staticmethod
    def compare(key_sizes: list[int]) -> list[CryptoResult]:
        """Compare vulnerability across multiple key sizes."""
        results = []
        for ks in key_sizes:
            pipeline = CryptoAnalysisPipeline(key_size=ks)
            results.append(pipeline.analyze())
        return results
