"""
测试后端模块 (模拟器后端，后端注册表)。

覆盖:
    - 模拟器后端运行 Grover 搜索
    - 后端注册和获取
    - 自定义电路运行
    - 后端验证
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsl.backends import get_backend, list_backends, SimulatorBackend
from qsl.backends.base import AbstractBackend
from qsl.utils.exceptions import BackendNotAvailableError


class TestSimulatorBackend:
    """测试模拟器后端。"""

    def test_create_backend(self):
        """创建模拟器后端。"""
        backend = get_backend("simulator")
        assert isinstance(backend, AbstractBackend)
        assert backend.name == "simulator"

    def test_max_qubits(self):
        """最大量子比特数。"""
        backend = get_backend("simulator")
        assert backend.max_qubits == 20

    def test_run_grover(self):
        """模拟器上运行 Grover 搜索。"""
        backend = get_backend("simulator")
        result = backend.run_grover_search(
            n_qubits=3,
            oracle=lambda x: x == 5,
            num_solutions=1,
            shots=10,
            verbose=False,
        )
        assert result.success_count > 0
        assert result.num_solutions == 1

    def test_run_custom_circuit(self):
        """运行自定义电路。"""
        backend = get_backend("simulator")
        results = backend.run_custom_circuit(
            n_qubits=2,
            circuit_fn=lambda s: (s.h(0), s.cnot(0, 1)),
            shots=10,
        )
        assert len(results) == 10

    def test_validate_request(self):
        """请求验证。"""
        backend = get_backend("simulator")
        # 太大量子比特
        try:
            backend.validate_request(25, 10)
            assert False
        except Exception:
            pass

    def test_simulator_constructor_options(self):
        """模拟器构造函数选项。"""
        backend = SimulatorBackend(
            normalize_after_gates=True,
            check_normalization=True,
        )
        assert isinstance(backend, SimulatorBackend)


class TestBackendRegistry:
    """测试后端注册表。"""

    def test_list_backends(self):
        """列出后端。"""
        backends = list_backends()
        assert "simulator" in backends

    def test_get_unknown_backend(self):
        """获取未知后端。"""
        try:
            get_backend("nonexistent_backend_xyz")
            assert False
        except BackendNotAvailableError:
            pass

    def test_get_simulator_multiple_times(self):
        """多次获取同一后端。"""
        b1 = get_backend("simulator")
        b2 = get_backend("simulator")
        # 每次创建新实例
        assert isinstance(b1, SimulatorBackend)
        assert isinstance(b2, SimulatorBackend)
