"""量子态可视化 (matplotlib)。

包含 Bloch 球 / 密度矩阵城市图 / 振幅柱状图 / Q 球 / 计数直方图。
matplotlib 采用延迟导入, 未安装时抛出带安装提示的 ImportError。
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

_FACE_BLUE = "#4C86C6"
_LIGHT_BLUE = "#9FC5E8"
_ORANGE = "#E8862E"
_EDGE = "#2E5A87"


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


# ----------------------------------------------------------------
# 输入归一化
# ----------------------------------------------------------------
def _as_amplitudes(state) -> np.ndarray:
    """QuantumState 或长度为 2^n 的复向量 -> 振幅 ndarray。"""
    from ..core.state import QuantumState
    if isinstance(state, QuantumState):
        return np.asarray(state.amplitudes, dtype=complex).ravel()
    arr = np.asarray(state, dtype=complex).ravel()
    if arr.size < 1 or (arr.size & (arr.size - 1)) != 0:
        raise ValueError(f"态向量长度必须是 2 的幂, 得到 {arr.size}")
    return arr


def _as_density_matrix(obj) -> np.ndarray:
    """QuantumState / DensityMatrix / 态向量 / 密度矩阵 -> rho ndarray。"""
    from ..core.state import QuantumState
    if isinstance(obj, QuantumState):
        amps = np.asarray(obj.amplitudes, dtype=complex).ravel()
        return np.outer(amps, amps.conj())
    if hasattr(obj, "get_matrix"):
        return np.asarray(obj.get_matrix(), dtype=complex)
    arr = np.asarray(obj, dtype=complex)
    if arr.ndim == 1:
        return np.outer(arr, arr.conj())
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"密度矩阵必须是方阵, 得到形状 {arr.shape}")
    return arr


def _as_bloch_vector(state_or_vector) -> Tuple[float, float, float]:
    """QuantumState / (x,y,z) / 2 维复向量 -> Bloch 向量。"""
    from ..core.state import QuantumState
    if isinstance(state_or_vector, QuantumState):
        return tuple(float(c) for c in state_or_vector.bloch_vector(0))
    arr = np.asarray(state_or_vector, dtype=complex).ravel()
    if arr.size == 2:
        a, b = arr
        norm = math.sqrt(abs(a) ** 2 + abs(b) ** 2)
        if norm > 0:
            a, b = a / norm, b / norm
        x = 2.0 * float(np.real(np.conj(a) * b))
        y = 2.0 * float(np.imag(np.conj(a) * b))
        z = float(abs(a) ** 2 - abs(b) ** 2)
        return (x, y, z)
    if arr.size == 3:
        return tuple(float(np.real(c)) for c in arr)
    raise ValueError(
        "plot_bloch_sphere 输入须为 QuantumState / (x,y,z) / 2 维复向量, "
        f"得到长度 {arr.size}"
    )


def _wireframe_sphere(ax, color="#B9C4CE", alpha=0.5):
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 13)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color=color, lw=0.5, alpha=alpha)


# ----------------------------------------------------------------
# Bloch 球
# ----------------------------------------------------------------
def plot_bloch_sphere(state_or_vector, ax=None):
    """
    单比特 Bloch 球: 线框球体 + 坐标轴 + 态矢量箭头 + |0>/|1> 标注。

    参数:
        state_or_vector: QuantumState (任意比特数, 取 qubit=0 的
                         bloch_vector) / (x, y, z) 三元组 / 2 维复向量
        ax: 可选 3D Axes; 不传则新建 figure

    返回:
        (fig, ax)
    """
    plt = _import_pyplot()
    x, y, z = _as_bloch_vector(state_or_vector)

    if ax is None:
        fig = plt.figure(figsize=(5.5, 5.5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    _wireframe_sphere(ax)

    L = 1.25
    ax.plot([-L, L], [0, 0], [0, 0], color="#9AA5AE", lw=0.8)
    ax.plot([0, 0], [-L, L], [0, 0], color="#9AA5AE", lw=0.8)
    ax.plot([0, 0], [0, 0], [-L, L], color="#9AA5AE", lw=0.8)
    ax.text(L + 0.10, 0, 0, "x", fontsize=11)
    ax.text(0, L + 0.10, 0, "y", fontsize=11)
    ax.text(0, 0, L + 0.12, "z", fontsize=11)
    ax.text(0, 0, L + 0.30, r"$|0\rangle$", ha="center", fontsize=12)
    ax.text(0, 0, -L - 0.32, r"$|1\rangle$", ha="center", fontsize=12)

    ax.quiver(0, 0, 0, x, y, z, color="#C0392B", linewidth=2.2,
              arrow_length_ratio=0.15)

    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-L, L)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title("Bloch sphere", fontsize=12)
    return fig, ax


# ----------------------------------------------------------------
# 密度矩阵 3D 城市图
# ----------------------------------------------------------------
def plot_state_city(rho_or_state, ax=None):
    """
    密度矩阵 3D 城市图 (bar3d), 实部/虚部两个子图。

    参数:
        rho_or_state: QuantumState (自动转 rho = |psi><psi|) /
                      DensityMatrix / 密度矩阵 ndarray / 态向量
        ax: 可选的两个 3D Axes 组成的 (ax_re, ax_im) 元组; 不传则新建

    返回:
        (fig, (ax_re, ax_im))
    """
    plt = _import_pyplot()
    rho = _as_density_matrix(rho_or_state)
    dim = rho.shape[0]
    n = int(round(math.log2(dim))) if dim > 1 else 0
    labels = [format(i, f"0{n}b") for i in range(dim)]

    if ax is None:
        fig = plt.figure(figsize=(10, 4.6))
        axes = (fig.add_subplot(121, projection="3d"),
                fig.add_subplot(122, projection="3d"))
    elif isinstance(ax, (list, tuple)) and len(ax) == 2:
        axes = (ax[0], ax[1])
        fig = axes[0].figure
    else:
        fig = ax.figure
        axes = (ax, fig.add_subplot(122, projection="3d"))

    xpos, ypos = np.meshgrid(np.arange(dim), np.arange(dim), indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros(dim * dim)
    dx = dy = 0.68
    max_abs = float(np.max(np.abs(rho))) if rho.size else 1.0
    zlim = max(max_abs * 1.15, 1e-6)

    for axi, part, color, title in (
        (axes[0], np.real(rho), _FACE_BLUE, r"Re($\rho$)"),
        (axes[1], np.imag(rho), _ORANGE, r"Im($\rho$)"),
    ):
        axi.bar3d(xpos, ypos, zpos, dx, dy, part.ravel(),
                  color=color, shade=True, alpha=0.95,
                  edgecolor="#FFFFFF", linewidth=0.2)
        axi.set_zlim(-zlim, zlim)
        axi.set_title(title, fontsize=12)
        if dim <= 8:
            axi.set_xticks(np.arange(dim) + dx / 2)
            axi.set_xticklabels(labels, fontsize=7)
            axi.set_yticks(np.arange(dim) + dy / 2)
            axi.set_yticklabels(labels, fontsize=7)
        else:
            axi.set_xticks([])
            axi.set_yticks([])
    return fig, axes


# ----------------------------------------------------------------
# 振幅柱状图
# ----------------------------------------------------------------
def plot_amplitudes(state, ax=None):
    """
    振幅柱状图: 每个基态 |alpha|^2 的概率柱, x 轴为二进制标签。

    参数:
        state: QuantumState 或 2^n 维复向量
        ax: 可选 Axes; 不传则新建 figure

    返回:
        (fig, ax)
    """
    plt = _import_pyplot()
    amps = _as_amplitudes(state)
    n = int(round(math.log2(amps.size)))
    probs = np.abs(amps) ** 2
    labels = [format(i, f"0{n}b") for i in range(amps.size)]

    if ax is None:
        w = max(4.5, 0.55 * amps.size + 1.5)
        fig, ax = plt.subplots(figsize=(w, 3.6))
    else:
        fig = ax.figure

    xs = np.arange(amps.size)
    ax.bar(xs, probs, color=_FACE_BLUE, edgecolor=_EDGE, lw=0.7, width=0.72)
    many = amps.size > 8
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9,
                       rotation=45 if many else 0,
                       ha="right" if many else "center")
    ax.set_ylim(0, min(1.05, max(float(probs.max()) * 1.18, 1e-6)))
    ax.set_ylabel("Probability")
    ax.set_title("Basis-state amplitudes")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


# ----------------------------------------------------------------
# Q 球 (简化版)
# ----------------------------------------------------------------
def plot_qsphere(state, ax=None):
    """
    Q 球简化版: 3D 球面上每个基态一个点 (按 Hamming 权重分环),
    点大小 = 概率, 颜色 = 相位 (hsv colormap)。

    参数:
        state: QuantumState 或 2^n 维复向量
        ax: 可选 3D Axes; 不传则新建 figure

    返回:
        (fig, ax)
    """
    plt = _import_pyplot()
    amps = _as_amplitudes(state)
    size = amps.size
    n = int(round(math.log2(size)))
    probs = np.abs(amps) ** 2
    phases = np.angle(amps)

    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # 半透明球面
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    ax.plot_surface(np.outer(np.cos(u), np.sin(v)),
                    np.outer(np.sin(u), np.sin(v)),
                    np.outer(np.ones_like(u), np.cos(v)),
                    color=_LIGHT_BLUE, alpha=0.10,
                    edgecolor="none", shade=False)

    rings = [[] for _ in range(n + 1)]
    for i in range(size):
        rings[bin(i).count("1")].append(i)

    hsv = plt.get_cmap("hsv")
    px, py, pz, sizes, colors = [], [], [], [], []
    for w, idxs in enumerate(rings):
        if not idxs:
            continue
        theta = math.pi * w / n
        m = len(idxs)
        for k, i in enumerate(idxs):
            phi = 2 * math.pi * k / m
            px.append(math.sin(theta) * math.cos(phi))
            py.append(math.sin(theta) * math.sin(phi))
            pz.append(math.cos(theta))
            sizes.append(36 + 640 * probs[i])
            colors.append(hsv((phases[i] + math.pi) / (2 * math.pi)))

    ax.scatter(px, py, pz, s=sizes, c=colors, alpha=0.95,
               edgecolors="#333333", linewidths=0.4, depthshade=True)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_zlim(-1.15, 1.15)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title("Q-sphere", fontsize=12)
    return fig, ax


# ----------------------------------------------------------------
# 计数直方图
# ----------------------------------------------------------------
def plot_histogram(counts: Dict, ax=None, highlight=None,
                   title: Optional[str] = None, sort: bool = True):
    """
    计数直方图。

    参数:
        counts: {比特串: 次数} (如 ExecutionResult.get_counts())
        ax: 可选 Axes; 不传则新建 figure
        highlight: 需高亮的比特串集合 (橙色 #E8862E), 其余淡蓝
        title: 图标题
        sort: True 时按计数降序排列

    返回:
        (fig, ax)
    """
    plt = _import_pyplot()
    items = [(str(k), int(v)) for k, v in counts.items()]
    if sort:
        items.sort(key=lambda kv: (-kv[1], kv[0]))
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    hl = {str(h) for h in (highlight or [])}
    colors = [_ORANGE if k in hl else _LIGHT_BLUE for k in labels]

    if ax is None:
        w = max(4.5, 0.62 * max(len(items), 1) + 1.6)
        fig, ax = plt.subplots(figsize=(w, 3.8))
    else:
        fig = ax.figure

    xs = np.arange(len(items))
    ax.bar(xs, values, color=colors, edgecolor=_EDGE, lw=0.7, width=0.72)
    vmax = max(values) if values else 1
    for x, v in zip(xs, values):
        ax.text(x, v + vmax * 0.015, str(v), ha="center", va="bottom",
                fontsize=9)
    many = len(items) > 6
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9,
                       rotation=45 if many else 0,
                       ha="right" if many else "center")
    ax.set_ylim(0, vmax * 1.14)
    ax.set_ylabel("Counts")
    if title:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


__all__ = [
    "plot_bloch_sphere",
    "plot_state_city",
    "plot_amplitudes",
    "plot_qsphere",
    "plot_histogram",
]
