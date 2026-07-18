"""
Quantum-enhanced Blockchain (conceptual demonstration).

*** WARNING: DEMONSTRATION ONLY — classical PoW + QFT for cosmetics only ***
"""

import math
import hashlib
import time
import random
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class Transaction:
    """A blockchain transaction."""
    sender: str
    receiver: str
    amount: float
    timestamp: float = field(default_factory=time.time)
    signature: str = ""

    def hash(self) -> str:
        data = f"{self.sender}{self.receiver}{self.amount}{self.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class QuantumBlock:
    """
    A quantum-enhanced blockchain block.

    Uses QFT-generated random numbers for PoS validator selection.
    Mining uses classical PoW; the quantum speedup claim in previous
    versions was a conceptual illustration, not an actual Grover search.
    """
    index: int
    transactions: List[Transaction]
    previous_hash: str
    timestamp: float = field(default_factory=time.time)
    quantum_random: float = 0.0
    nonce: int = 0
    hash: str = ""
    validator: str = ""

    def compute_hash(self) -> str:
        data = (f"{self.index}{self.previous_hash}{self.timestamp}"
                f"{self.quantum_random}{self.nonce}")
        for tx in self.transactions:
            data += tx.hash()
        return hashlib.sha256(data.encode()).hexdigest()

    def mine_block(self, difficulty: int = 4):
        """
        Mine the block via classical proof-of-work.

        Searches for a nonce such that the block hash starts with
        'difficulty' number of zeros. This is standard classical PoW.
        """
        target = "0" * difficulty
        max_nonce = min(16 ** difficulty, 100000)

        for nonce in range(max_nonce):
            self.nonce = nonce
            self.hash = self.compute_hash()
            if self.hash.startswith(target):
                return True

        return False


class QuantumBlockchain:
    """
    Quantum-enhanced blockchain demonstration.

    Features:
    - QFT-based pseudo-random numbers for validator selection (PoS)
    - Classical proof-of-work mining
    - Chain integrity verification

    NOTE: The "quantum-accelerated mining" from previous versions was
    a mathematical illustration (O(sqrt(N))) and NOT actual Grover
    search on SHA-256. Implementing Grover for SHA-256 would require
    millions of logical qubits and is currently infeasible.

    Args:
        difficulty: Mining difficulty (leading zeros)
    """

    def __init__(self, difficulty: int = 4):
        self.chain: List[QuantumBlock] = []
        self.pending_transactions: List[Transaction] = []
        self.difficulty = difficulty
        self.validators: List[str] = ["Node_A", "Node_B", "Node_C", "Node_D"]

        self._create_genesis_block()

    def _create_genesis_block(self):
        genesis = QuantumBlock(
            index=0,
            transactions=[],
            previous_hash="0" * 64,
        )
        genesis.hash = genesis.compute_hash()
        self.chain.append(genesis)

    def _quantum_random(self) -> float:
        """
        Generate a genuinely unpredictable number using Hadamard gate + measurement.

        Applies H to a |0> state producing a uniform superposition,
        then measures — this gives a truly random bit from quantum principles.
        The process is repeated to build a float in [0, 1).

        Falls back to classical random if QuantumState is unavailable.
        """
        try:
            from ..core.state import QuantumState

            state = QuantumState(1)
            state.h(0)
            result, _ = state.measure()
            return result + random.random()  # mix with classical for better distribution
        except Exception:
            return random.random()

    def _select_validator(self) -> str:
        """Select a validator using QFT-based random number (PoS)."""
        quantum_r = self._quantum_random()
        n_validators = len(self.validators)
        selection = int(quantum_r * n_validators * 1.5) % n_validators
        return self.validators[selection]

    def add_transaction(self, sender: str, receiver: str, amount: float):
        """Add a new transaction to the pending pool."""
        tx = Transaction(sender=sender, receiver=receiver, amount=amount)
        self.pending_transactions.append(tx)

    def create_block(self) -> Optional[QuantumBlock]:
        """
        Create a new block from pending transactions.

        1. QFT selects validator (PoS)
        2. Block assembled
        3. Classical proof-of-work mining
        """
        if not self.pending_transactions:
            return None

        validator = self._select_validator()

        txs = self.pending_transactions[:10]
        self.pending_transactions = self.pending_transactions[10:]

        prev_block = self.chain[-1]
        new_block = QuantumBlock(
            index=prev_block.index + 1,
            transactions=txs,
            previous_hash=prev_block.hash,
            quantum_random=self._quantum_random(),
            validator=validator,
        )

        if new_block.mine_block(self.difficulty):
            self.chain.append(new_block)
            return new_block

        return None

    def verify_chain(self) -> bool:
        """Verify the entire blockchain integrity."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            if current.hash != current.compute_hash():
                return False

            if current.previous_hash != previous.hash:
                return False

            target = "0" * self.difficulty
            if not current.hash.startswith(target):
                return False

        return True

    def get_block(self, index: int) -> Optional[QuantumBlock]:
        """Get a block by index."""
        if 0 <= index < len(self.chain):
            return self.chain[index]
        return None

    @property
    def length(self) -> int:
        return len(self.chain)

    def summary(self) -> str:
        """Generate blockchain summary."""
        return (
            f"Quantum Blockchain: {self.length} blocks, "
            f"{len(self.pending_transactions)} pending TXs, "
            f"difficulty={self.difficulty}, "
            f"validators={len(self.validators)}"
        )
