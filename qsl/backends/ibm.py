"""
IBM Quantum backend.

Connects to IBM Quantum cloud service to run Grover search on real
quantum hardware or cloud simulators.

Prerequisites:
    pip install qiskit>=1.0.0 qiskit-aer>=0.14.0 qiskit-ibm-runtime>=0.20.0

Authentication:
    Requires IBM Quantum account and API Token.
    Provide via:
        1. Environment variable: IBMQ_TOKEN
        2. Init parameter: IBMBackend(token="your_token")
        3. ~/.qiskit/ configuration file

Available backends:
    - Real hardware: ibm_brisbane, ibm_sherbrooke, ibm_kyoto, etc.
    - Cloud simulators: ibmq_qasm_simulator, simulator_statevector, etc.
"""

import os
import math
import time
from typing import Callable, Optional, List, Tuple, Dict

from .base import AbstractBackend
from ..core.grover import GroverResult
from ..utils.exceptions import (
    DependencyNotInstalledError,
    BackendConnectionError,
    BackendJobError,
    ConfigurationError,
    BackendNotAvailableError,
)


class IBMBackend(AbstractBackend):
    """
    IBM Quantum cloud backend.

    Connects to IBM Quantum Experience to execute Grover search on
    real quantum hardware.

    Usage:
        >>> backend = IBMBackend(token="your_ibm_token")
        >>> result = backend.run_grover_search(
        ...     n_qubits=3,
        ...     oracle=lambda x: x == 5,
        ...     num_solutions=1,
        ...     shots=100
        ... )

    Note: Real quantum hardware currently has noise, so search results
    are less precise than the simulator.
    """

    DEFAULT_MAX_QUBITS = 127

    def __init__(self,
                 name: str = "ibm",
                 token: Optional[str] = None,
                 hub: str = "ibm-q",
                 group: str = "open",
                 project: str = "main",
                 backend_instance: Optional[str] = None,
                 optimization_level: int = 3,
                 max_wait_seconds: int = 3600,
                 **options):
        super().__init__(name=name, **options)
        self._token = token or os.environ.get("IBMQ_TOKEN")
        self._hub = hub
        self._group = group
        self._project = project
        self._backend_instance = backend_instance
        self._optimization_level = optimization_level
        self._max_wait_seconds = max_wait_seconds

        self._provider = None
        self._backend = None
        self._qiskit_available = None

    @property
    def max_qubits(self) -> int:
        return self.DEFAULT_MAX_QUBITS

    def _ensure_qiskit(self):
        """Ensure qiskit is installed and initialized."""
        if self._qiskit_available is False:
            raise DependencyNotInstalledError(
                "qiskit",
                "pip install qiskit>=1.0.0 qiskit-aer>=0.14.0 qiskit-ibm-runtime>=0.20.0"
            )

        if self._qiskit_available is True and self._provider is not None:
            return

        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError as e:
            self._qiskit_available = False
            raise DependencyNotInstalledError(
                "qiskit-ibm-runtime",
                "pip install qiskit-ibm-runtime>=0.20.0"
            ) from e

        self._qiskit_available = True

        try:
            if self._token:
                self._provider = QiskitRuntimeService(
                    channel="ibm_quantum",
                    token=self._token,
                    instance=f"{self._hub}/{self._group}/{self._project}"
                )
            else:
                self._provider = QiskitRuntimeService(
                    channel="ibm_quantum",
                    instance=f"{self._hub}/{self._group}/{self._project}"
                )
        except Exception as e:
            raise BackendConnectionError(
                self._name,
                f"IBM Quantum connection failed: {e}. "
                f"Please ensure API token is correct and network is available."
            ) from e

    def _get_ibm_backend(self, min_qubits: int):
        """Get an available IBM quantum backend (cached after first success)."""
        if self._backend is not None:
            return self._backend

        self._ensure_qiskit()

        if self._backend_instance:
            try:
                self._backend = self._provider.backend(self._backend_instance)
            except Exception as e:
                raise BackendNotAvailableError(
                    self._backend_instance
                ) from e
        else:
            try:
                backends = self._provider.backends(
                    min_num_qubits=min_qubits,
                    operational=True,
                    simulator=False,
                )
                if backends:
                    self._backend = min(backends, key=lambda b: b.num_qubits)
                else:
                    backends = self._provider.backends(
                        min_num_qubits=min_qubits,
                        simulator=True,
                    )
                    if backends:
                        self._backend = backends[0]
                    else:
                        raise BackendNotAvailableError("ibm")
            except Exception as e:
                if isinstance(e, (BackendNotAvailableError, BackendConnectionError)):
                    raise
                raise BackendConnectionError(
                    self._name,
                    f"Failed to get IBM backend list: {e}"
                )

        return self._backend

    def _build_oracle(self, qc, oracle: Callable[[int], bool], n_qubits: int):
        """
        Build oracle circuit for Grover search from a black-box callable.

        WARNING: A black-box Python callable cannot be compiled to a
        quantum circuit (on real hardware it would have to be expressed
        as a Boolean circuit first). This method therefore performs
        O(2^n) classical enumeration of the marked states — a simulator
        convenience, NOT a real quantum oracle. Whenever the oracle is
        available as a Boolean expression, use _build_oracle_from_expressions
        instead, which compiles the AST directly into an ancilla-based
        quantum circuit with zero classical enumeration.
        """
        from qiskit import QuantumCircuit

        N = 1 << n_qubits
        marked = [x for x in range(N) if oracle(x)]

        if not marked:
            return qc

        # Use the last qubit as the target for multi-controlled gates
        target = n_qubits - 1
        controls = list(range(n_qubits - 1))

        for target_state in marked:
            # Flip bits that are 0 in target state
            for q in range(n_qubits):
                if ((target_state >> q) & 1) == 0:
                    qc.x(q)

            # Multi-controlled Z: H * MCX * H on target
            qc.h(target)
            if n_qubits > 2:
                qc.mcx(controls, target)
            elif n_qubits == 2:
                qc.cx(controls[0], target)
            qc.h(target)

            # Restore flipped bits
            for q in range(n_qubits):
                if ((target_state >> q) & 1) == 0:
                    qc.x(q)

        return qc

    def _build_oracle_from_expressions(self, expressions, n_qubits: int):
        """
        从布尔表达式 AST 直接构建量子 Oracle 电路 (无经典枚举)。

        使用 qsl.core.oracle 的后端无关编译器, 将表达式编译为
        X/CNOT/Toffoli/Z 可逆电路 (ancilla 辅助), 再翻译为 Qiskit 门。

        返回:
            (qc, total_qubits): Oracle 电路与总量子比特数
            (主寄存器 n_qubits + ancilla)
        """
        from qiskit import QuantumCircuit
        from ..core.oracle import compile_phase_oracle

        circ = compile_phase_oracle(expressions, n_qubits)
        total = n_qubits + circ.n_ancilla
        qc = QuantumCircuit(total)

        for gate, qs in circ.gates:
            if gate == "X":
                qc.x(qs[0])
            elif gate == "Z":
                qc.z(qs[0])
            elif gate == "CNOT":
                qc.cx(qs[0], qs[1])
            elif gate == "TOFFOLI":
                qc.ccx(qs[0], qs[1], qs[2])
            else:
                raise ValueError(f"未知的 Oracle 门: {gate}")

        return qc, total

    def _build_diffusion(self, qc, n_qubits: int, offset: int = 0):
        """Build Grover diffusion operator on the main register."""
        from qiskit import QuantumCircuit

        all_qubits = list(range(offset, offset + n_qubits))
        target = offset + n_qubits - 1
        controls = list(range(offset, offset + n_qubits - 1))

        for q in all_qubits:
            qc.h(q)
            qc.x(q)

        qc.h(target)
        if n_qubits > 2:
            qc.mcx(controls, target)
        elif n_qubits == 2:
            qc.cx(controls[0], target)
        qc.h(target)

        for q in all_qubits:
            qc.x(q)
            qc.h(q)

        return qc

    def _build_grover_circuit(self, n_qubits: int,
                               oracle: Callable[[int], bool],
                               iterations: int,
                               oracle_expressions=None):
        """Build Grover search quantum circuit.

        提供 oracle_expressions 时使用量子电路 Oracle (无枚举),
        否则退回黑盒枚举路径。返回 (qc, total_qubits, main_qubits)。
        """
        from qiskit import QuantumCircuit

        if oracle_expressions:
            oracle_qc, total = self._build_oracle_from_expressions(
                oracle_expressions, n_qubits)
            qc = QuantumCircuit(total)
            main = list(range(n_qubits))
            for q in main:
                qc.h(q)
            for _ in range(iterations):
                qc.compose(oracle_qc, inplace=True)
                qc = self._build_diffusion(qc, n_qubits)
            return qc, total, main

        qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            qc.h(q)
        for _ in range(iterations):
            qc = self._build_oracle(qc, oracle, n_qubits)
            qc = self._build_diffusion(qc, n_qubits)
        return qc, n_qubits, list(range(n_qubits))

    def run_grover_search(self,
                           n_qubits: int,
                           oracle: Callable[[int], bool],
                           num_solutions: Optional[int],
                           shots: int,
                           verbose: bool = False,
                           **run_options) -> GroverResult:
        """
        Run Grover search on IBM Quantum hardware.

        Args:
            n_qubits: Number of qubits
            oracle: Boolean oracle function (用于测量结果验证)
            num_solutions: Number of solutions (None 则 BBHT 指数搜索)
            shots: Number of measurements
            verbose: Print progress
            **run_options:
                - use_aer_simulator: Force local Aer simulator (default False)
                - max_wait_seconds: Override default wait time
                - oracle_expressions: BooleanExpr 列表。提供时 Oracle 直接
                  从布尔表达式编译为 ancilla 量子电路, 不做 O(2^n) 经典枚举

        Returns:
            GroverResult
        """
        self.validate_request(n_qubits, shots)

        oracle_expressions = run_options.get("oracle_expressions")

        if num_solutions is None:
            # BBHT 指数搜索: 解数量未知时逐步增大迭代次数
            return self._run_bbht_search(
                n_qubits, oracle, shots, verbose,
                oracle_expressions=oracle_expressions, **run_options)

        N = 1 << n_qubits
        theta = math.asin(math.sqrt(num_solutions / N))
        t_opt = max(1, round((math.pi / 2 - theta) / (2 * theta)))
        theory_prob = math.sin((2 * t_opt + 1) * theta) ** 2

        counts, total, main_qubits = self._execute_grover_circuit(
            n_qubits, oracle, t_opt, shots, verbose,
            oracle_expressions=oracle_expressions, **run_options)

        measurements = self._parse_counts(n_qubits, counts, oracle, shots)

        if verbose:
            self._print_results(measurements, counts)

        return GroverResult(
            n_qubits=n_qubits,
            num_solutions=num_solutions,
            iterations=t_opt,
            theta=theta,
            theory_success_prob=theory_prob,
            measurements=measurements,
            quantum_queries=t_opt,
        )

    def _run_bbht_search(self,
                         n_qubits: int,
                         oracle: Callable[[int], bool],
                         shots: int,
                         verbose: bool,
                         oracle_expressions=None,
                         lam: float = 1.34,
                         **run_options) -> GroverResult:
        """
        BBHT 指数搜索 (解数量未知): 每轮随机选择 t ∈ [0, m) 次迭代,
        找到解则停止, 否则 m *= lam 直到超过 √N。期望 Oracle 查询
        复杂度 O(√(N/M)), 无需预先经典枚举解空间。
        """
        import random as _random

        N = 1 << n_qubits
        sqrt_N = math.sqrt(N)
        m = 1.0
        total_queries = 0

        while m <= sqrt_N * lam:
            t = _random.randint(0, max(0, int(m)))
            total_queries += t

            counts, _, _ = self._execute_grover_circuit(
                n_qubits, oracle, t, shots, verbose=False,
                oracle_expressions=oracle_expressions, **run_options)

            measurements = self._parse_counts(n_qubits, counts, oracle, shots)
            if any(is_sol for _, _, is_sol in measurements):
                if verbose:
                    print(f"  BBHT: t={t} 次迭代后找到解 "
                          f"(累计 {total_queries} 次 Oracle 查询)")
                return GroverResult(
                    n_qubits=n_qubits,
                    num_solutions=None,
                    iterations=None,
                    theta=None,
                    theory_success_prob=None,
                    measurements=measurements,
                    quantum_queries=total_queries,
                )

            m = min(m * lam, sqrt_N * lam + 1.0)

        from ..utils.exceptions import NoSolutionError
        raise NoSolutionError(
            premises=["<oracle>"],
            n_qubits=n_qubits,
        )

    def _execute_grover_circuit(self,
                                n_qubits: int,
                                oracle: Callable[[int], bool],
                                iterations: int,
                                shots: int,
                                verbose: bool,
                                oracle_expressions=None,
                                **run_options):
        """构建、编译并执行单次 Grover 电路, 返回 (counts, total, main)。"""
        use_aer = run_options.get("use_aer_simulator", False)
        max_wait = run_options.get("max_wait_seconds", self._max_wait_seconds)

        qc, total, main_qubits = self._build_grover_circuit(
            n_qubits, oracle, iterations,
            oracle_expressions=oracle_expressions)

        # 仅测量主寄存器 (ancilla 已逆计算为 |0>)
        if total == n_qubits:
            qc.measure_all()
        else:
            from qiskit import ClassicalRegister
            creg = ClassicalRegister(n_qubits)
            qc.add_register(creg)
            qc.measure(main_qubits, creg)

        if verbose:
            print(f"\n  IBM Quantum - Grover Search")
            print(f"  Qubits: {n_qubits} (+{total - n_qubits} ancilla)")
            print(f"  Iterations: {iterations}")
            print(f"  Shots: {shots}")

        if use_aer:
            from qiskit_aer import AerSimulator
            backend_for_run = AerSimulator()
            if verbose:
                print(f"  Backend: Local Aer Simulator")
        else:
            ibm_backend = self._get_ibm_backend(total)
            backend_for_run = ibm_backend
            if verbose:
                print(f"  Backend: {ibm_backend.name}")

        try:
            from qiskit import transpile
            tqc = transpile(
                qc,
                backend=backend_for_run,
                optimization_level=self._optimization_level,
            )
        except Exception as e:
            raise BackendJobError(
                "transpile", self._name,
                f"Circuit compilation failed: {e}"
            )

        if verbose:
            print(f"  Submitting job...")

        try:
            job = backend_for_run.run(tqc, shots=shots)
        except Exception as e:
            raise BackendJobError(
                "submit", self._name,
                f"Job submission failed: {e}"
            )

        if verbose:
            if use_aer:
                print(f"  Waiting for simulation...")
            else:
                print(f"  Waiting for quantum hardware (max {max_wait}s)...")

        try:
            # Qiskit >= 1.0 重构后 JobStatus 路径可能变化, 逐级回退
            try:
                from qiskit.providers.jobstatus import JobStatus
            except ImportError:
                from qiskit.providers import JobStatus

            elapsed = 0
            wait_interval = 10 if not use_aer else 1

            while job.status() not in [
                JobStatus.DONE, JobStatus.CANCELLED, JobStatus.ERROR
            ]:
                time.sleep(wait_interval)
                elapsed += wait_interval
                if verbose and elapsed % 30 == 0 and not use_aer:
                    print(f"    Waiting... ({elapsed}s, status: {job.status()})")
                if elapsed > max_wait:
                    raise BackendJobError(
                        job.job_id(), self._name,
                        f"Job timed out (waited {elapsed}s)"
                    )

            if job.status() == JobStatus.ERROR:
                raise BackendJobError(
                    job.job_id(), self._name,
                    "Job execution error"
                )
            if job.status() == JobStatus.CANCELLED:
                raise BackendJobError(
                    job.job_id(), self._name,
                    "Job cancelled"
                )

            counts = job.result().get_counts()
        except Exception as e:
            if isinstance(e, BackendJobError):
                raise
            job_id_val = getattr(job, 'job_id', None)
            if callable(job_id_val):
                job_id_val = job_id_val()
            elif job_id_val is None:
                job_id_val = 'unknown'
            raise BackendJobError(
                str(job_id_val),
                self._name,
                f"Failed to get results: {e}"
            )

        return counts, total, main_qubits

    def _parse_counts(self,
                       n_qubits: int,
                       counts: Dict[str, int],
                       oracle: Callable[[int], bool],
                       total_shots: int) -> List[Tuple[int, float, bool]]:
        """Convert qiskit counts dict to QSL standard measurement format."""
        measurements = []
        for bitstring, count in counts.items():
            bits = bitstring[::-1]  # reverse to match QSL convention
            try:
                result_int = int(bits, 2)
            except ValueError:
                result_int = 0
            prob = count / total_shots if total_shots > 0 else 0.0
            is_solution = oracle(result_int)
            for _ in range(count):
                measurements.append((result_int, prob, is_solution))
        return measurements

    def _print_results(self,
                        measurements: List[Tuple[int, float, bool]],
                        counts: Dict[str, int]):
        """Print IBM backend result summary."""
        if not measurements:
            print("  (no measurement results)")
            return

        print(f"\n  Measurement distribution:")
        for bitstring, count in counts.items():
            bits_rev = bitstring[::-1]
            try:
                val = int(bits_rev, 2)
            except ValueError:
                val = 0
            print(f"    |{bits_rev}> (int={val}): {count} times")

        success_count = sum(1 for m in measurements if m[2])
        print(f"\n  Success: {success_count}/{len(measurements)}")
        print(f"  {'='*60}\n")
