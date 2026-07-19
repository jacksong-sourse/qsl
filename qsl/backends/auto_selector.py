"""Auto backend selector for optimal quantum resource allocation."""

from typing import Optional
from .base import AbstractBackend
from ..utils.exceptions import BackendConnectionError, BackendNotAvailableError


class AutoBackend:
    """
    Automatically select the best backend based on problem requirements.

    Decision logic:
    - n_qubits <= 20: Local simulator (fastest, zero cost)
    - 21 <= n_qubits <= 127: IBM Quantum (cloud access)
    - n_qubits > 127: AWS Braket (largest capacity)

    Args:
        max_qubits: Maximum qubits needed (default 20, the local
                    simulator limit — the most common safe choice)
        need_error_mitigation: Whether error mitigation is needed
        prefer: Backend preference hint ("simulator", "ibm", "aws")
    """

    def __init__(self,
                 max_qubits: int = 20,
                 need_error_mitigation: bool = False,
                 prefer: str = "auto",
                 ibm_token: Optional[str] = None,
                 aws_region: str = "us-east-1"):
        self.max_qubits = max_qubits
        self.need_error_mitigation = need_error_mitigation
        self.prefer = prefer
        self.ibm_token = ibm_token
        self.aws_region = aws_region

    def select(self) -> tuple:
        """
        Select the optimal backend.

        Returns:
            (backend_instance, backend_type_string)
        """
        if self.prefer == "simulator" or self.max_qubits <= 20:
            from .simulator import SimulatorBackend
            return SimulatorBackend(name="simulator"), "simulator"

        if self.prefer == "ibm" or 21 <= self.max_qubits <= 127:
            try:
                from .ibm import IBMBackend
                backend = IBMBackend(
                    name="ibm",
                    token=self.ibm_token,
                )
                return backend, "ibm"
            except (ImportError, BackendConnectionError):
                pass  # Fall through to AWS

        if self.max_qubits > 127 or self.prefer == "aws":
            try:
                from .aws_braket import AWSBraketBackend
                backend = AWSBraketBackend(
                    name="aws_sv1",
                    region=self.aws_region,
                )
                return backend, "aws"
            except (ImportError, BackendConnectionError):
                # Ultimate fallback: local simulator
                from .simulator import SimulatorBackend
                return SimulatorBackend(name="simulator"), "simulator"

        # Default fallback
        from .simulator import SimulatorBackend
        return SimulatorBackend(name="simulator"), "simulator"

    @staticmethod
    def get_backend_for_device(device_name: str, **options) -> AbstractBackend:
        """
        Directly get a backend by device name.

        Supported names:
            "auto" - Auto-select
            "simulator" - Local simulator
            "ibm_kyoto", "ibm_sherbrooke", etc. - IBM devices
            "aws_sv1", "aws_tn1", "aws_rigetti" - AWS Braket devices

        Args:
            device_name: Backend device name
            **options: Backend-specific options

        Returns:
            Backend instance
        """
        from ..utils.exceptions import BackendNotAvailableError

        device_name = device_name.lower()

        if device_name == "auto":
            return AutoBackend(max_qubits=options.get("max_qubits", 20)).select()[0]

        if device_name == "simulator":
            from .simulator import SimulatorBackend
            return SimulatorBackend(**options)

        if device_name.startswith("ibm"):
            try:
                from .ibm import IBMBackend
                return IBMBackend(name=device_name, **options)
            except ImportError:
                raise BackendNotAvailableError(device_name)

        if device_name.startswith("aws"):
            try:
                from .aws_braket import AWSBraketBackend
                return AWSBraketBackend(name=device_name, **options)
            except ImportError:
                raise BackendNotAvailableError(device_name)

        raise BackendNotAvailableError(device_name)
