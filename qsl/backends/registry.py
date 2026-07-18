"""
后端注册表。

管理所有可用的后端实现 (模拟器 + 真实量子计算机)。
"""

from typing import Dict, Type, Optional
from .base import AbstractBackend


# 全局后端注册表
_BACKEND_REGISTRY: Dict[str, Type[AbstractBackend]] = {}


def register_backend(name: str, backend_cls: Type[AbstractBackend]):
    """
    注册后端实现。

    参数:
        name: 后端名称 (如 "simulator", "ibm_brisbane")
        backend_cls: 实现了 AbstractBackend 的类
    """
    _BACKEND_REGISTRY[name.lower()] = backend_cls


def get_backend(name: str, **options) -> AbstractBackend:
    """
    获取后端实例。

    参数:
        name: 后端名称
        **options: 传递给后端构造函数的选项

    返回:
        AbstractBackend 实例

    失败模式:
        - 后端未注册: 抛出 BackendNotAvailableError
        - 后端初始化失败: 将原始异常包装为 BackendConnectionError
    """
    from ..utils.exceptions import BackendNotAvailableError, BackendConnectionError

    name = name.lower()

    if name not in _BACKEND_REGISTRY:
        # 尝试延迟导入
        if name == "simulator":
            from .simulator import SimulatorBackend
            register_backend("simulator", SimulatorBackend)
        elif name == "ibm" or name.startswith("ibm"):
            from .ibm import IBMBackend
            register_backend(name, IBMBackend)

    if name not in _BACKEND_REGISTRY:
        available = list(_BACKEND_REGISTRY.keys())
        error_msg = f"后端 '{name}' 未注册。可用后端: {available}"
        # 检查是否是 IBM 后端
        if name.startswith("ibm"):
            error_msg += (
                "\n使用 IBM 量子后端需要安装 qiskit: pip install qiskit"
            )
            error_msg += (
                "\n然后使用: QSLCompiler(backend='ibm')"
            )
        raise BackendNotAvailableError(name)

    backend_cls = _BACKEND_REGISTRY[name]
    try:
        return backend_cls(name=name, **options)
    except Exception as e:
        raise BackendConnectionError(name, str(e))


def list_backends() -> list:
    """
    列出所有已注册的后端名称。

    返回:
        后端名称列表
    """
    # 自动注册内置后端
    _register_builtins()
    return sorted(_BACKEND_REGISTRY.keys())


def _register_builtins():
    """注册内置后端 (延迟加载)。"""
    if "simulator" not in _BACKEND_REGISTRY:
        try:
            from .simulator import SimulatorBackend
            register_backend("simulator", SimulatorBackend)
        except ImportError:
            pass
