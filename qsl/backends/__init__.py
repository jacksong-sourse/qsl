"""
QSL 后端模块 - 经典模拟器 和 IBM 量子计算机。

使用:
    >>> from qsl.backends import get_backend
    >>> backend = get_backend("simulator")  # 经典模拟
    >>> backend = get_backend("ibm")        # IBM 量子计算机
"""

from .base import AbstractBackend
from .simulator import SimulatorBackend
from .registry import get_backend, list_backends, register_backend

# 预注册内置后端
from .registry import _register_builtins
_register_builtins()

__all__ = [
    "AbstractBackend",
    "SimulatorBackend",
    "get_backend",
    "list_backends",
    "register_backend",
]
