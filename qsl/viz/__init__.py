"""
qsl.viz — 基于 matplotlib 的可视化模块。

所有函数延迟导入 matplotlib; 未安装时抛出 ImportError
(提示 pip install matplotlib)。支持无头 (Agg) 后端运行。
"""

from .circuit_drawer import draw_circuit_mpl
from .state_viz import (
    plot_amplitudes,
    plot_bloch_sphere,
    plot_histogram,
    plot_qsphere,
    plot_state_city,
)

__all__ = [
    "draw_circuit_mpl",
    "plot_bloch_sphere",
    "plot_state_city",
    "plot_amplitudes",
    "plot_qsphere",
    "plot_histogram",
]
