"""matplotlib 出版级电路绘图器。

风格: 学术白底, 门淡蓝填充 (#DCE9F7) + 深蓝描边 (#2E5A87)。
matplotlib 采用延迟导入, 未安装时抛出带安装提示的 ImportError。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from ..circuit.circuit import QuantumCircuit, Instruction

_DEFAULT_STYLE = {
    "line_color": "#1A1A1A",
    "gate_facecolor": "#DCE9F7",
    "gate_edgecolor": "#2E5A87",
    "text_color": "#1A1A1A",
    "background": "#FFFFFF",
    "fontsize": 11,
}

# 预设风格 (与 Qiskit 风格名称兼容)
_PRESET_STYLES = {
    "iqp": {
        "line_color": "#000000",
        "gate_facecolor": "#FAF8D3",
        "gate_edgecolor": "#000000",
        "text_color": "#000000",
        "background": "#FFFFFF",
        "fontsize": 11,
    },
    "default": dict(_DEFAULT_STYLE),
    "bw": {
        "line_color": "#000000",
        "gate_facecolor": "#FFFFFF",
        "gate_edgecolor": "#000000",
        "text_color": "#000000",
        "background": "#FFFFFF",
        "fontsize": 11,
    },
    "clifford": {
        "line_color": "#1A1A1A",
        "gate_facecolor": "#D4FFD4",
        "gate_edgecolor": "#2E872E",
        "text_color": "#1A1A1A",
        "background": "#FFFFFF",
        "fontsize": 11,
    },
    "textbook": {
        "line_color": "#000000",
        "gate_facecolor": "#E6E6FA",
        "gate_edgecolor": "#4B0082",
        "text_color": "#000000",
        "background": "#FFFFFF",
        "fontsize": 11,
    },
}

# 受控 + 方框目标的两比特门 (控制点 + 目标方框)
_CONTROLLED_BOX = {"cy", "ch", "cs", "csdg", "ct", "ctdg", "crx", "cry", "cu"}
# 受控相位门: 两个实心圆点连线
_CONTROLLED_PHASE = {"cz", "cp", "crz"}
_PARAM_NAMES = ("θ", "φ", "λ", "γ")


def _import_pyplot():
    """延迟导入 matplotlib.pyplot, 未安装时给出安装提示。"""
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "qsl.viz 需要 matplotlib, 请先安装: pip install matplotlib"
        ) from e
    return plt


def _format_label(gate) -> str:
    """门标签, 含数值参数时格式为 label(θ=0.785) (3 位小数)。"""
    label = gate.label or gate.name.upper()
    if not gate.params:
        return label
    parts: List[str] = []
    for i, p in enumerate(gate.params):
        name = _PARAM_NAMES[i] if i < len(_PARAM_NAMES) else f"p{i}"
        try:
            if hasattr(p, "parameters") and p.parameters:  # 未绑定符号参数
                parts.append(str(p))
            else:
                parts.append(f"{name}={float(p):.3f}")
        except (TypeError, ValueError):
            parts.append(str(p))
    return f"{label}({', '.join(parts)})"


def draw_circuit_mpl(circuit: QuantumCircuit, ax=None,
                     style: Optional[Dict] = None, fold: int = -1):
    """
    用 matplotlib 绘制出版级量子电路图。

    参数:
        circuit: QuantumCircuit
        ax: 可选的 matplotlib Axes; 不传则新建 figure
        style: 可选 dict 或字符串, 覆盖键 line_color / gate_facecolor /
               gate_edgecolor / text_color / background / fontsize;
               字符串可选 'iqp' / 'default' / 'bw' / 'clifford' / 'textbook'
        fold: >0 时每行最多 fold 列后换行; -1 不换行

    返回:
        (fig, ax)
    """
    plt = _import_pyplot()
    from matplotlib.patches import Circle, FancyBboxPatch

    st = dict(_DEFAULT_STYLE)
    if style is not None:
        if isinstance(style, str):
            preset = _PRESET_STYLES.get(style)
            if preset is None:
                raise ValueError(
                    f"未知风格 {style!r}, 可选: {sorted(_PRESET_STYLES)}")
            st.update(preset)
        elif isinstance(style, dict):
            st.update(style)
        else:
            raise TypeError(
                f"style 必须是 dict 或字符串,  got {type(style).__name__}")
    fs = st["fontsize"]

    n = circuit.num_qubits
    insts: List[Instruction] = list(circuit.data)

    # 分行 (fold>0 时每行最多 fold 列)
    if fold and fold > 0:
        bands = [insts[i:i + fold] for i in range(0, len(insts), fold)] or [[]]
    else:
        bands = [insts]
    band_gap = 1.7
    band_h = n + band_gap
    n_cols_max = max((len(b) for b in bands), default=0)
    y_low = -((len(bands) - 1) * band_h + (n - 1))

    if ax is None:
        w = max(3.5, 0.85 * (n_cols_max + 1.8))
        h = max(1.8, 0.62 * ((len(bands) - 1) * band_h + n + 1.2))
        fig, ax = plt.subplots(figsize=(w, h))
    else:
        fig = ax.figure

    fig.patch.set_facecolor(st["background"])
    ax.set_facecolor(st["background"])

    def y_of(q: int, band: int) -> float:
        return -(band * band_h) - q

    # ----------------------------------------------------------------
    # 绘图基元
    # ----------------------------------------------------------------
    def box(cx, cy, w, h):
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.09",
            facecolor=st["gate_facecolor"], edgecolor=st["gate_edgecolor"],
            lw=1.4, zorder=3))

    def gate_text(cx, cy, text):
        shrink = max(fs - 4, 6) if len(text) > 8 else fs - 1
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=shrink, color=st["text_color"], zorder=5)

    def vline(x, ya, yb, dashed=False):
        ax.plot([x, x], [ya, yb], color=st["line_color"],
                lw=1.2 if dashed else 1.4,
                ls=(0, (4, 3)) if dashed else "-", zorder=2)

    def control_dot(x, y):
        ax.add_patch(Circle((x, y), 0.10, facecolor=st["gate_edgecolor"],
                            edgecolor=st["gate_edgecolor"], zorder=4))

    def target_x(x, y, r=0.17):
        ax.plot([x - r, x + r], [y - r, y + r], color=st["gate_edgecolor"],
                lw=2.0, zorder=4, solid_capstyle="round")
        ax.plot([x - r, x + r], [y + r, y - r], color=st["gate_edgecolor"],
                lw=2.0, zorder=4, solid_capstyle="round")

    def target_plus(x, y, r=0.23):
        ax.add_patch(Circle((x, y), r, facecolor="none",
                            edgecolor=st["gate_edgecolor"], lw=1.6, zorder=4))
        ax.plot([x - r, x + r], [y, y], color=st["gate_edgecolor"],
                lw=1.6, zorder=4)
        ax.plot([x, x], [y - r, y + r], color=st["gate_edgecolor"],
                lw=1.6, zorder=4)

    # ----------------------------------------------------------------
    # 比特线与标签
    # ----------------------------------------------------------------
    for bi, band in enumerate(bands):
        x_end = max(len(band) + 0.5, 0.6)
        for q in range(n):
            y = y_of(q, bi)
            ax.plot([-0.3, x_end], [y, y], color=st["line_color"],
                    lw=1.2, zorder=1, solid_capstyle="round")
            ax.text(-0.45, y, f"q{q}", ha="right", va="center",
                    fontsize=fs, color=st["text_color"])

    # ----------------------------------------------------------------
    # 逐指令绘制
    # ----------------------------------------------------------------
    for bi, band in enumerate(bands):
        for ci, inst in enumerate(band):
            x = ci + 1.0
            g = inst.gate
            name = g.name
            qs = list(inst.qubits)
            ys = [y_of(q, bi) for q in qs]
            y_top, y_bot = max(ys), min(ys)

            if name == "barrier":
                vline(x, y_bot - 0.45, y_top + 0.45, dashed=True)
                continue

            label = _format_label(g)

            if len(qs) == 1:
                box(x, ys[0], 0.62, 0.62)
                gate_text(x, ys[0], label)
            elif name == "cx":
                vline(x, y_bot, y_top)
                control_dot(x, ys[0])
                target_plus(x, ys[1])
            elif name in _CONTROLLED_PHASE:
                vline(x, y_bot, y_top)
                control_dot(x, ys[0])
                control_dot(x, ys[1])
            elif name in _CONTROLLED_BOX:
                vline(x, y_bot, y_top)
                control_dot(x, ys[0])
                box(x, ys[1], 0.62, 0.62)
                gate_text(x, ys[1], label)
            elif name in ("swap", "iswap"):
                vline(x, y_bot, y_top)
                target_x(x, ys[0])
                target_x(x, ys[1])
                if name == "iswap":
                    ax.text(x, y_top + 0.34, "i", ha="center", va="center",
                            fontsize=fs - 1, color=st["text_color"], zorder=5)
            elif name == "cswap":
                vline(x, y_bot, y_top)
                control_dot(x, ys[0])
                target_x(x, ys[1])
                target_x(x, ys[2])
            elif name == "ccx" or name.startswith("mcx"):
                vline(x, y_bot, y_top)
                for y in ys[:-1]:
                    control_dot(x, y)
                target_plus(x, ys[-1])
            elif name.startswith("mcz"):
                vline(x, y_bot, y_top)
                for y in ys:
                    control_dot(x, y)
            else:
                # 通用多比特门 (unitary/rxx/dcx/ecr 等): 跨比特大矩形
                cy = (y_top + y_bot) / 2
                box(x, cy, 0.72, (y_top - y_bot) + 0.62)
                gate_text(x, cy, label)

    ax.set_xlim(-1.0, max(n_cols_max + 0.8, 1.0))
    ax.set_ylim(y_low - 0.8, 0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


__all__ = ["draw_circuit_mpl"]
