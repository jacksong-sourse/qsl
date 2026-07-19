"""qsl.viz 可视化模块测试 (matplotlib Agg 无头后端)。"""

import sys

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qsl.circuit.circuit import QuantumCircuit
from qsl.core.state import QuantumState
from qsl.viz import (
    draw_circuit_mpl,
    plot_amplitudes,
    plot_bloch_sphere,
    plot_histogram,
    plot_qsphere,
    plot_state_city,
)


def _demo_circuit() -> QuantumCircuit:
    """覆盖 cx/ccx/mcx/mcz/barrier/unitary/参数门等画法。"""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.rx(0, 0.785)
    qc.ry(1, 1.234)
    qc.u(2, 0.1, 0.2, 0.3)
    qc.cx(0, 1)
    qc.cy(1, 2)
    qc.cz(0, 2)
    qc.cp(0.5, 1, 3)
    qc.crz(0.25, 2, 3)
    qc.swap(0, 3)
    qc.iswap(1, 2)
    qc.cswap(0, 1, 2)
    qc.ccx(0, 1, 2)
    qc.mcx([0, 1, 2], 3)
    qc.mcz([0, 1, 3])
    qc.rxx(0.7, 0, 2)
    qc.barrier()
    qc.unitary(np.eye(4, dtype=complex), [2, 3], label="V")
    return qc


# ----------------------------------------------------------------
# circuit_drawer
# ----------------------------------------------------------------
def test_draw_circuit_mpl_returns_fig_ax():
    fig, ax = draw_circuit_mpl(_demo_circuit())
    assert fig is not None and ax is not None
    assert ax.figure is fig
    plt.close(fig)


def test_draw_circuit_mpl_existing_ax():
    fig, ax = plt.subplots()
    fig2, ax2 = draw_circuit_mpl(_demo_circuit(), ax=ax)
    assert fig2 is fig and ax2 is ax
    plt.close(fig)


def test_draw_circuit_mpl_style_override():
    style = {"gate_facecolor": "#FFEEEE", "gate_edgecolor": "#880000",
             "background": "#FAFAFA", "fontsize": 9}
    fig, ax = draw_circuit_mpl(_demo_circuit(), style=style)
    assert fig.get_facecolor() is not None
    plt.close(fig)


def test_draw_circuit_mpl_fold():
    fig, ax = draw_circuit_mpl(_demo_circuit(), fold=5)
    assert fig is not None
    plt.close(fig)


def test_draw_circuit_mpl_empty():
    fig, ax = draw_circuit_mpl(QuantumCircuit(2))
    assert fig is not None
    plt.close(fig)


def test_circuit_draw_output_mpl():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    fig, ax = qc.draw(output="mpl")
    assert fig is not None and ax is not None
    plt.close(fig)


# ----------------------------------------------------------------
# state_viz
# ----------------------------------------------------------------
def test_plot_bloch_sphere_from_state():
    st = QuantumState(2)
    st.h(0)
    fig, ax = plot_bloch_sphere(st)
    assert fig is not None and ax is not None
    plt.close(fig)


def test_plot_bloch_sphere_from_vector_and_tuple():
    fig1, _ = plot_bloch_sphere((0.0, 0.0, 1.0))
    fig2, _ = plot_bloch_sphere(np.array([1, 1j]) / np.sqrt(2))
    assert fig1 is not None and fig2 is not None
    plt.close(fig1)
    plt.close(fig2)


def test_plot_state_city_from_state_and_rho():
    st = QuantumState(2)
    st.h(0)
    st.cnot(0, 1)
    fig, axes = plot_state_city(st)
    assert len(axes) == 2
    plt.close(fig)
    amps = np.asarray(st.amplitudes)
    rho = np.outer(amps, amps.conj())
    fig2, axes2 = plot_state_city(rho)
    assert len(axes2) == 2
    plt.close(fig2)


def test_plot_amplitudes():
    st = QuantumState(3)
    st.h(0)
    st.h(1)
    fig, ax = plot_amplitudes(st)
    assert fig is not None
    plt.close(fig)


def test_plot_qsphere():
    st = QuantumState(2)
    st.h(0)
    st.cnot(0, 1)
    fig, ax = plot_qsphere(st)
    assert fig is not None
    plt.close(fig)


def test_plot_histogram_highlight_sort_title():
    counts = {"00": 512, "11": 480, "01": 20, "10": 12}
    fig, ax = plot_histogram(counts, highlight={"11"}, title="Bell counts")
    texts = [t.get_text() for t in ax.texts]
    assert "512" in texts and "480" in texts
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels[0] == "00"  # 按计数降序
    assert ax.get_title() == "Bell counts"
    # highlight 的柱子应为橙色 #E8862E
    orange = tuple(round(c, 3) for c in matplotlib.colors.to_rgb("#E8862E"))
    facecolors = [tuple(round(c, 3) for c in p.get_facecolor()[:3])
                  for p in ax.patches]
    assert orange in facecolors
    plt.close(fig)


def test_plot_histogram_no_sort():
    fig, ax = plot_histogram({"b": 1, "a": 2}, sort=False)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["b", "a"]
    plt.close(fig)


def test_plot_histogram_from_execution_result():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    counts = qc.execute(shots=256, seed=7).get_counts()
    fig, ax = plot_histogram(counts)
    assert fig is not None
    plt.close(fig)


# ----------------------------------------------------------------
# 未安装 matplotlib 时的 ImportError
# ----------------------------------------------------------------
def test_matplotlib_missing_raises_importerror(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match="matplotlib"):
        draw_circuit_mpl(QuantumCircuit(1))
    with pytest.raises(ImportError, match="matplotlib"):
        plot_histogram({"0": 1})
    with pytest.raises(ImportError, match="matplotlib"):
        plot_bloch_sphere((0.0, 0.0, 1.0))
    with pytest.raises(ImportError, match="matplotlib"):
        plot_amplitudes(QuantumState(1))
    with pytest.raises(ImportError, match="matplotlib"):
        plot_qsphere(QuantumState(1))
    with pytest.raises(ImportError, match="matplotlib"):
        plot_state_city(QuantumState(1))
