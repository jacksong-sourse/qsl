"""AWS Braket quantum backend."""

import os
import time
from typing import Callable, Optional
import numpy as np
from .base import AbstractBackend
from ..core.grover import GroverResult
from ..utils.exceptions import (
    DependencyNotInstalledError,
    BackendConnectionError,
    BackendJobError,
)


class AWSBraketBackend(AbstractBackend):
    """
    AWS Braket quantum computing backend.

    Supports:
    - SV1: State vector simulator (up to 34 qubits)
    - TN1: Tensor network simulator
    - Rigetti: Aspen-M series
    - IonQ: Trapped ion devices
    - QuEra: Neutral atom (Aquila)

    Args:
        region: AWS region (default: us-east-1)
        s3_bucket: S3 bucket for results (reads from env AWS_BRAKET_BUCKET)
        device_arn: Specific device ARN (auto-selects SV1 if None)
        max_wait_seconds: Maximum wait time
    """

    DEFAULT_MAX_QUBITS = 1000

    def __init__(self,
                 name: str = "aws_sv1",
                 region: str = "us-east-1",
                 s3_bucket: Optional[str] = None,
                 device_arn: Optional[str] = None,
                 max_wait_seconds: int = 7200,
                 **options):
        super().__init__(name=name, **options)
        self._region = region
        self._s3_bucket = s3_bucket or os.environ.get("AWS_BRAKET_BUCKET", "amazon-braket-bucket")
        self._device_arn = device_arn
        self._max_wait_seconds = max_wait_seconds

        self._device_map = {
            "aws_sv1": "SV1",
            "aws_tn1": "TN1",
            "aws_dm1": "DM1",
            "aws_rigetti": "Rigetti",
            "aws_ionq": "IonQ",
            "aws_quera": "QuEra",
        }

    @property
    def max_qubits(self) -> int:
        return self.DEFAULT_MAX_QUBITS

    def _get_device(self):
        """Get or auto-select AWS Braket device."""
        try:
            import boto3
            from braket.aws import AwsDevice
        except ImportError:
            raise DependencyNotInstalledError(
                "amazon-braket-sdk",
                "pip install amazon-braket-sdk boto3"
            )

        if self._device_arn:
            return AwsDevice(self._device_arn)

        device_name = self._device_map.get(self.name, "SV1")

        try:
            session = boto3.Session(region_name=self._region)

            if device_name in ("SV1", "TN1", "DM1"):
                return AwsDevice(f"arn:aws:braket:::device/quantum-simulator/amazon/{device_name.lower()}")
            elif device_name == "Rigetti":
                devices = AwsDevice.get_devices(
                    provider_names=["Rigetti"],
                    statuses=["ONLINE"],
                    region=self._region
                )
                if devices:
                    return devices[0]
            elif device_name == "IonQ":
                devices = AwsDevice.get_devices(
                    provider_names=["IonQ"],
                    statuses=["ONLINE"],
                    region=self._region
                )
                if devices:
                    return devices[0]
            elif device_name == "QuEra":
                devices = AwsDevice.get_devices(
                    provider_names=["QuEra"],
                    statuses=["ONLINE"],
                    region=self._region
                )
                if devices:
                    return devices[0]

            return AwsDevice(f"arn:aws:braket:::device/quantum-simulator/amazon/sv1")

        except Exception as e:
            raise BackendConnectionError(self.name, str(e))

    def _build_grover_circuit(self, n_qubits: int,
                               oracle: Callable[[int], bool],
                               iterations: int):
        """Build a Braket Grover circuit."""
        try:
            from braket.circuits import Circuit
        except ImportError:
            raise DependencyNotInstalledError(
                "amazon-braket-sdk",
                "pip install amazon-braket-sdk"
            )

        circuit = Circuit()

        # Hadamard on all qubits
        for q in range(n_qubits):
            circuit = circuit.h(q)

        # Grover iterations
        N = 1 << n_qubits
        marked = [x for x in range(N) if oracle(x)]

        for _ in range(iterations):
            # Oracle: mark solution states
            if marked:
                # Build oracle by marking marked states with Z-phase
                for target in marked:
                    for q in range(n_qubits):
                        if ((target >> q) & 1) == 0:
                            circuit = circuit.x(q)
                    # Multi-controlled Z
                    if n_qubits > 2:
                        circuit = circuit.h(n_qubits - 1)
                        circuit = circuit.cnot(list(range(n_qubits - 1)), n_qubits - 1)
                        circuit = circuit.h(n_qubits - 1)
                    else:
                        circuit = circuit.cz(0, 1)
                    for q in range(n_qubits):
                        if ((target >> q) & 1) == 0:
                            circuit = circuit.x(q)

            # Diffusion operator
            for q in range(n_qubits):
                circuit = circuit.h(q)
                circuit = circuit.x(q)
            if n_qubits > 2:
                circuit = circuit.h(n_qubits - 1)
                circuit = circuit.cnot(list(range(n_qubits - 1)), n_qubits - 1)
                circuit = circuit.h(n_qubits - 1)
            else:
                circuit = circuit.cz(0, 1)
            for q in range(n_qubits):
                circuit = circuit.x(q)
                circuit = circuit.h(q)

        return circuit

    def submit(self, circuit, shots: int = 1000) -> str:
        """
        Submit a quantum circuit to AWS Braket.

        Args:
            circuit: A Braket circuit object (braket.circuits.Circuit)
            shots: Number of measurement shots

        Returns:
            job_id string
        """
        try:
            from braket.aws import AwsQuantumTask
        except ImportError:
            raise DependencyNotInstalledError(
                "amazon-braket-sdk",
                "pip install amazon-braket-sdk"
            )

        device = self._get_device()

        task = device.run(
            circuit,
            s3_destination_folder=(self._s3_bucket, "qsl-jobs/"),
            shots=shots,
        )

        return task.id

    def result(self, job_id: str, max_wait: int = None) -> dict:
        """
        Get results from a submitted job.

        Args:
            job_id: The job ID returned by submit()
            max_wait: Maximum wait time in seconds

        Returns:
            Dict with measurement_counts
        """
        try:
            from braket.aws import AwsQuantumTask
        except ImportError:
            raise DependencyNotInstalledError(
                "amazon-braket-sdk",
                "pip install amazon-braket-sdk"
            )

        max_wait = max_wait or self._max_wait_seconds

        task = AwsQuantumTask(arn=job_id)

        elapsed = 0
        wait_interval = 5

        while task.state() not in ("COMPLETED", "FAILED", "CANCELLED"):
            time.sleep(wait_interval)
            elapsed += wait_interval
            if elapsed > max_wait:
                raise BackendJobError(job_id, self.name, "Job timed out")

        if task.state() == "FAILED":
            raise BackendJobError(job_id, self.name, "Job failed")
        if task.state() == "CANCELLED":
            raise BackendJobError(job_id, self.name, "Job cancelled")

        result = task.result()
        counts = result.measurement_counts

        return {"measurement_counts": dict(counts), "job_id": job_id}

    def run_grover_search(self,
                           n_qubits: int,
                           oracle: Callable[[int], bool],
                           num_solutions: Optional[int],
                           shots: int,
                           verbose: bool = False,
                           **run_options) -> GroverResult:
        """
        Run Grover search on AWS Braket.

        Builds a Braket circuit and submits it to the AWS Braket
        device (SV1 simulator by default, or a QPU if configured).
        Falls back to local simulator if Braket SDK is not available.

        num_solutions 为 None 时, 委托本地 BBHT 指数搜索
        (避免在未知解数量时做 O(2^n) 经典枚举)。
        """
        import math

        self.validate_request(n_qubits, shots)

        if num_solutions is None:
            # 解数量未知: 使用本地 BBHT 搜索 (电路 Oracle, 无经典枚举)
            from ..core.grover import GroverSearch
            search = GroverSearch(n_qubits, verbose=verbose)
            oracle_expressions = run_options.get("oracle_expressions")
            if oracle_expressions:
                return search.search_expressions(
                    expressions=oracle_expressions,
                    num_solutions=None, shots=shots)
            return search.search(condition=oracle, num_solutions=None,
                                 shots=shots)

        N = 1 << n_qubits
        theta = math.asin(math.sqrt(max(1, num_solutions) / N))
        t_opt = max(1, round((math.pi / 2 - theta) / (2 * theta)))

        try:
            from braket.circuits import Circuit, Observable

            circuit = self._build_grover_circuit(n_qubits, oracle, t_opt)
            # Add measurement
            circuit = circuit.state_vector()

            if verbose:
                print(f"\n  AWS Braket - Grover Search")
                print(f"  Device: {self.name}")
                print(f"  Qubits: {n_qubits}, Iterations: {t_opt}, Shots: {shots}")

            job_id = self.submit(circuit, shots=shots)
            if verbose:
                print(f"  Job ID: {job_id}")

            result_data = self.result(job_id)
            counts = result_data.get("measurement_counts", {})

        except (DependencyNotInstalledError, BackendConnectionError):
            if verbose:
                print("  AWS Braket SDK not available, falling back to local simulator.")
            from ..core.grover import GroverSearch
            search = GroverSearch(n_qubits, verbose=verbose)
            return search.search(condition=oracle, num_solutions=num_solutions, shots=shots)

        # Parse results
        measurements = []
        for bitstring, count in counts.items():
            try:
                val = int(bitstring, 2)
            except ValueError:
                val = 0
            prob = count / shots if shots > 0 else 0.0
            is_sol = oracle(val)
            for _ in range(count):
                measurements.append((val, prob, is_sol))

        theory_prob = math.sin((2 * t_opt + 1) * theta) ** 2

        return GroverResult(
            n_qubits=n_qubits,
            num_solutions=num_solutions,
            iterations=t_opt,
            theta=theta,
            theory_success_prob=theory_prob,
            measurements=measurements,
        )

    def __repr__(self) -> str:
        return f"AWSBraketBackend(device='{self.name}', region='{self._region}')"
