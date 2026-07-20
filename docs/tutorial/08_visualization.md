[← 返回文档中心](../index.md)

# QSL 可视化指南：绘制电路图、直方图与量子态

本教程将带你学习 QSL 的可视化功能，包括 ASCII 文本电路图、Matplotlib 高清电路图、测量结果直方图、Bloch 球、态城市图等多种可视化方式。

---

## 📦 安装可视化依赖

可视化功能基于 matplotlib，需要安装额外依赖：

```bash
pip install "qsl-quantum[viz]"
```

这会安装 matplotlib 和 numpy（如果尚未安装）。安装完成后，你就可以使用所有绘图功能了。

> 💡 **提示**：如果你只需要 ASCII 文本电路图，无需安装任何额外依赖，核心包已内置支持。

---

## 🎨 电路图绘制

QSL 支持两种电路图输出方式：文本（ASCII）和 Matplotlib 矢量图。

### 1. 文本绘制：ASCII 电路图

最简单的方式是使用文本输出，无需任何依赖，直接在终端打印：

```python
from qsl import QuantumCircuit

# 创建一个演示电路：量子傅里叶变换（3 比特）
qc = QuantumCircuit(3)
qc.h(0)
qc.cp(3.14159/2, 1, 0)
qc.cp(3.14159/4, 2, 0)
qc.h(1)
qc.cp(3.14159/2, 2, 1)
qc.h(2)
qc.swap(0, 2)

# 方式一：直接打印（默认 text 模式）
print("=== 方式一：直接 print(qc) ===")
print(qc)

# 方式二：显式调用 draw 方法
print("\n=== 方式二：qc.draw(output='text') ===")
text_drawing = qc.draw(output="text")
print(text_drawing)
print(f"\ntype(text_drawing) = {type(text_drawing)}")
```

### 预期输出（ASCII 电路）

```
     ┌───┐            ┌───┐
q_0: ┤ H ├─■──────■───┤ H ├──────X─
     └───┘ │P(π/2)│   └───┘┌───┐ │
q_1: ──────■──────┼────────┤ H ├─┼─
                  │P(π/4)  └───┘ │
q_2: ─────────────■──────────────X─

type(text_drawing) = <class 'str'>
```

`qc.draw(output="text")` 返回一个字符串，你可以将它保存到文件、打印到日志，或者嵌入到文档中。

---

### 2. Matplotlib 绘制：高清矢量图

如果需要用于论文、报告或演示，可以使用 Matplotlib 绘制专业风格的电路图：

```python
from qsl import QuantumCircuit
import matplotlib.pyplot as plt

# 创建 Bell 态电路
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# 使用 matplotlib 绘制，返回 (fig, ax)
fig, ax = qc.draw(output="mpl", style="iqp")

# 添加标题
fig.suptitle("Bell 态制备电路", fontsize=16, fontweight="bold")

# 显示图形
plt.tight_layout()
plt.show()
```

`qc.draw(output="mpl")` 返回一个元组 `(fig, ax)`，即 Matplotlib 的 Figure 和 Axes 对象，你可以进一步自定义样式。

---

### 3. 内置绘图风格

QSL 提供了 5 种内置风格，满足不同场景需求：

| 风格名称 | 描述 | 适用场景 |
|----------|------|----------|
| `'iqp'` | IBM Quantum 风格（默认推荐） | 论文、报告、演示 |
| `'default'` | QSL 默认简洁风格 | 快速查看、日常使用 |
| `'bw'` | 黑白单色风格 | 黑白打印、出版物 |
| `'clifford'` | Clifford 组高亮风格 | 教学、突出 Clifford 门 |
| `'textbook'` | 量子教科书经典风格 | 教材、课程讲义 |

### 风格对比示例

```python
from qsl import QuantumCircuit
import matplotlib.pyplot as plt

# 创建一个包含多种门的演示电路
qc = QuantumCircuit(3)
qc.h(0)
qc.x(1)
qc.ry(0.5, 2)
qc.cx(0, 1)
qc.cz(1, 2)
qc.t(0)
qc.s(2)
qc.swap(0, 2)
qc.measure_all()

styles = ['iqp', 'default', 'bw', 'clifford', 'textbook']

fig, axes = plt.subplots(1, 5, figsize=(25, 5))
for i, style in enumerate(styles):
    fig_i, ax_i = qc.draw(output="mpl", style=style, ax=axes[i])
    axes[i].set_title(f"style='{style}'", fontsize=12, pad=20)

plt.suptitle("不同绘图风格对比", fontsize=16, y=1.05)
plt.tight_layout()
plt.show()
```

---

### 4. 自定义风格字典

你还可以传入自定义字典来精细控制每个元素的颜色和样式：

```python
from qsl import QuantumCircuit
import matplotlib.pyplot as plt

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# 自定义配色：赛博朋克风格
custom_style = {
    "gate_color": {
        "h": "#00ffff",       # H 门：青色
        "x": "#ff00ff",       # X 门：品红
        "cx": "#ffff00",      # CNOT：黄色
        "measure": "#00ff00", # 测量：绿色
    },
    "bg_color": "#1a1a2e",     # 背景：深蓝紫
    "wire_color": "#e0e0e0",   # 导线：浅灰
    "text_color": "#ffffff",   # 文字：白色
}

fig, ax = qc.draw(output="mpl", style=custom_style)
fig.patch.set_facecolor("#1a1a2e")
fig.suptitle("赛博朋克风格电路", color="#00ffff", fontsize=14)
plt.tight_layout()
plt.show()
```

---

## 📊 测量结果可视化：直方图

`plot_histogram` 是最常用的可视化函数，用于展示量子测量的统计分布。

### 基础用法

```python
from qsl import QuantumCircuit, plot_histogram
import matplotlib.pyplot as plt

# 创建 Grover 搜索电路（搜索 |11⟩）
qc = QuantumCircuit(2)
qc.h([0, 1])
qc.cz(0, 1)
qc.h([0, 1])
qc.x([0, 1])
qc.cz(0, 1)
qc.x([0, 1])
qc.h([0, 1])
qc.measure_all()

# 执行模拟
res = qc.execute(shots=4096)

# 绘制直方图
fig, ax = plt.subplots(figsize=(10, 6))
plot_histogram(res.counts, title="Grover 搜索结果 (4096 次采样)", ax=ax)
plt.show()
```

### highlight 参数：高亮正确解

你可以使用 `highlight` 参数标记出正确答案，让结果一目了然：

```python
from qsl import QuantumCircuit, plot_histogram
import matplotlib.pyplot as plt

qc = QuantumCircuit(2)
qc.h([0, 1])
qc.cz(0, 1)
qc.h([0, 1])
qc.x([0, 1])
qc.cz(0, 1)
qc.x([0, 1])
qc.h([0, 1])
qc.measure_all()
res = qc.execute(shots=4096)

fig, ax = plt.subplots(figsize=(10, 6))
# highlight 接受二进制字符串或整数
plot_histogram(
    res.counts,
    title="Grover 搜索：目标 |11⟩ 被高亮显示",
    highlight="11",  # 也可以用 highlight=3（整数形式）
    ax=ax
)
plt.show()
```

正确解的柱状条会用醒目的颜色（默认红色）高亮，其他柱状条为蓝色。

### 多结果对比

你可以传入多个 counts 字典进行对比：

```python
from qsl import QuantumCircuit, plot_histogram
import matplotlib.pyplot as plt

# 无 Grover 迭代（仅均匀叠加）
qc0 = QuantumCircuit(3)
qc0.h([0, 1, 2])
qc0.measure_all()
res0 = qc0.execute(shots=4096)

# 1 次 Grover 迭代
qc1 = QuantumCircuit(3)
qc1.h([0, 1, 2])
# ... 此处省略 oracle 和 diffusion 构造
qc1.h([0, 1, 2])
qc1.x([0, 1, 2])
qc1.h(2)
qc1.ccx(0, 1, 2)
qc1.h(2)
qc1.x([0, 1, 2])
qc1.h([0, 1, 2])
qc1.measure_all()
res1 = qc1.execute(shots=4096)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
plot_histogram(res0.counts, title="无迭代（均匀分布）", ax=ax1)
plot_histogram(res1.counts, title="1 次 Grover 迭代", highlight="101", ax=ax2)
plt.tight_layout()
plt.show()
```

---

## 🔵 单比特态可视化：Bloch 球

`plot_bloch_sphere` 可以在 Bloch 球上直观展示单量子比特的状态。

### Bloch 球基础

```python
from qsl import QuantumCircuit, plot_bloch_sphere
import matplotlib.pyplot as plt

# 制备几种不同的单比特态
states = [
    ("|0⟩", [1, 0]),
    ("|1⟩", [0, 1]),
    ("|+⟩ (H|0⟩)", [1/2**0.5, 1/2**0.5]),
    ("|-⟩ (H|1⟩)", [1/2**0.5, -1/2**0.5]),
]

fig = plt.figure(figsize=(16, 8))
for i, (name, state) in enumerate(states):
    ax = fig.add_subplot(2, 4, i+1, projection='3d')
    plot_bloch_sphere(state, ax=ax)
    ax.set_title(name, fontsize=12)

plt.suptitle("常见单比特态在 Bloch 球上的表示", fontsize=14)
plt.tight_layout()
plt.show()
```

### 观察量子门对 Bloch 球的影响

```python
from qsl import QuantumCircuit, plot_bloch_sphere
import numpy as np
import matplotlib.pyplot as plt

def apply_gate_get_state(gate_name):
    qc = QuantumCircuit(1)
    if gate_name == "H":
        qc.h(0)
    elif gate_name == "X":
        qc.x(0)
    elif gate_name == "Y":
        qc.y(0)
    elif gate_name == "Z":
        qc.z(0)
    elif gate_name == "T":
        qc.t(0)
    elif gate_name == "S":
        qc.s(0)
    res = qc.execute()
    return res.statevector()

gates = ["H", "X", "Y", "Z", "S", "T"]
fig = plt.figure(figsize=(18, 6))
for i, gate in enumerate(gates):
    ax = fig.add_subplot(1, 6, i+1, projection='3d')
    state = apply_gate_get_state(gate)
    plot_bloch_sphere(state, ax=ax)
    ax.set_title(f"{gate}|0⟩", fontsize=12)

plt.suptitle("各种量子门作用于 |0⟩ 后的 Bloch 球表示", fontsize=14)
plt.tight_layout()
plt.show()
```

---

## 🏙️ 密度矩阵可视化：3D 城市图

`plot_state_city` 用于可视化密度矩阵，适合观察混态和含噪声系统。

### 纯态 vs 混态对比

```python
from qsl import QuantumCircuit, plot_state_city
import numpy as np
import matplotlib.pyplot as plt

# 纯态：Bell 态
qc_bell = QuantumCircuit(2)
qc_bell.h(0)
qc_bell.cx(0, 1)
res_bell = qc_bell.execute()
rho_pure = res_bell.density_matrix()

# 完全混态：I/4
rho_mixed = np.eye(4) / 4

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
plot_state_city(rho_pure, ax=ax1, title="Bell 纯态 |Φ⁺⟩⟨Φ⁺|")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
plot_state_city(rho_mixed, ax=ax2, title="完全混态 I/4")

plt.tight_layout()
plt.show()
```

你会看到纯态的城市图有高耸的"建筑"代表量子相干项，而完全混态的城市图是平坦的。

---

## 📈 概率幅可视化：plot_amplitudes

`plot_amplitudes` 用于直接绘制复概率幅的实部和虚部，适合态向量分析。

```python
from qsl import QuantumCircuit, plot_amplitudes
import matplotlib.pyplot as plt

# 创建 3 比特 GHZ 态
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(0, 2)
res = qc.execute()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_amplitudes(res.statevector(), ax=ax1, show_imag=True)
ax1.set_title("GHZ 态复概率幅")
ax1.set_xlabel("计算基矢")

# 测量后统计
res_shots = qc.execute(shots=4096)
from qsl import plot_histogram
plot_histogram(res_shots.counts, ax=ax2, title="测量结果统计")
ax2.set_xlabel("测量结果（十进制）")

plt.tight_layout()
plt.show()
```

---

## 🌐 Q-Sphere 可视化：plot_qsphere

`plot_qsphere` 是一种更高级的量子态可视化方式，将所有基矢投影到一个球面上：
- 北极是 |0...0⟩，南极是 |1...1⟩
- 点的大小代表概率大小
- 颜色代表相位

```python
from qsl import QuantumCircuit, plot_qsphere
import matplotlib.pyplot as plt

# GHZ 态
qc_ghz = QuantumCircuit(3)
qc_ghz.h(0)
qc_ghz.cx(0, 1)
qc_ghz.cx(0, 2)
res_ghz = qc_ghz.execute()

# W 态
import numpy as np
w_state = np.array([0, 1/3**0.5, 1/3**0.5, 0, 1/3**0.5, 0, 0, 0], dtype=complex)

fig = plt.figure(figsize=(14, 7))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
plot_qsphere(res_ghz.statevector(), ax=ax1)
ax1.set_title("GHZ 态 Q-Sphere", fontsize=12)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
plot_qsphere(w_state, ax=ax2)
ax2.set_title("W 态 Q-Sphere", fontsize=12)

plt.suptitle("Q-Sphere：量子态的全局可视化", fontsize=14)
plt.tight_layout()
plt.show()
```

---

## 💾 保存图片到文件

所有基于 Matplotlib 的图形都可以用 `fig.savefig()` 保存为各种格式：

```python
from qsl import QuantumCircuit, plot_histogram
import matplotlib.pyplot as plt

# 创建并执行电路
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(0, 2)
qc.measure_all()
res = qc.execute(shots=8192)

# 1. 保存电路图
fig1, ax1 = qc.draw(output="mpl", style="iqp")
fig1.savefig(
    "circuit.png",
    dpi=150,              # 分辨率：150 DPI
    bbox_inches="tight",  # 自动裁剪白边
    facecolor="white",    # 背景色
    edgecolor="none",
)
print("电路图已保存到 circuit.png")

# 2. 保存直方图
fig2, ax2 = plt.subplots(figsize=(10, 6))
plot_histogram(res.counts, title="GHZ 态测量结果", ax=ax2)
fig2.savefig(
    "histogram.pdf",     # 也可以保存为 PDF/SVG 矢量图
    bbox_inches="tight",
)
print("直方图已保存到 histogram.pdf")

# 3. 高分辨率 TIFF（用于印刷）
fig2.savefig(
    "histogram_300dpi.tiff",
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"},
)
print("高分辨率 TIFF 已保存")
```

### 支持的输出格式

| 格式 | 扩展名 | 特点 |
|------|--------|------|
| PNG | `.png` | 位图，通用，文件小 |
| PDF | `.pdf` | 矢量图，适合 LaTeX 论文 |
| SVG | `.svg` | 矢量图，适合网页/Inkscape编辑 |
| EPS | `.eps` | 矢量图，适合印刷期刊 |
| TIFF | `.tiff` | 高分辨率位图，无压缩损失 |

---

## 📝 完整示例：Grover 算法全流程可视化

让我们用一个完整示例串联所有可视化功能：

```python
from qsl import (
    QuantumCircuit,
    plot_histogram,
    plot_bloch_sphere,
    plot_state_city,
    plot_amplitudes,
)
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("  Grover 搜索算法全流程可视化")
print("=" * 60)

# 1. 构建 3 比特 Grover 电路，搜索 |101⟩
n = 3
target = "101"
qc = QuantumCircuit(n)
qc.h(range(n))

# Oracle：标记 |101⟩
qc.x(1)
qc.h(2)
qc.ccx(0, 1, 2)
qc.h(2)
qc.x(1)

# Diffusion
qc.h(range(n))
qc.x(range(n))
qc.h(2)
qc.ccx(0, 1, 2)
qc.h(2)
qc.x(range(n))
qc.h(range(n))

# 测量
qc.measure_all()

print("\n[1] 电路结构（ASCII）：")
print(qc.draw())

# 2. 执行模拟
res = qc.execute(shots=8192)
sv = res.statevector()

print(f"\n[2] 执行完成，shots=8192")
print(f"    目标态 {target} 出现次数: {res.counts.get(int(target, 2), 0)}")

# 3. 创建大图布局
fig = plt.figure(figsize=(18, 12))

# a. 电路图
fig_dummy, ax_circuit = plt.subplots(figsize=(12, 4))
qc.draw(output="mpl", style="iqp", ax=ax_circuit)
ax_circuit.set_title("(a) Grover 搜索电路（目标 |101⟩）", fontsize=12, pad=15)
fig_dummy.savefig("grover_circuit.png", dpi=150, bbox_inches="tight")
plt.close(fig_dummy)
print("\n[3] 电路图已保存: grover_circuit.png")

# b. 直方图（独立保存）
fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
plot_histogram(res.counts, title="(b) 测量结果统计（8192 次采样）", highlight=target, ax=ax_hist)
fig_hist.savefig("grover_histogram.png", dpi=150, bbox_inches="tight")
plt.close(fig_hist)
print("[4] 直方图已保存: grover_histogram.png")

# c. 态城市图
fig_city = plt.figure(figsize=(10, 8))
ax_city = fig_city.add_subplot(111, projection='3d')
rho = np.outer(sv, sv.conj())
plot_state_city(rho, ax=ax_city, title="(c) 密度矩阵态城市图")
fig_city.savefig("grover_state_city.png", dpi=150, bbox_inches="tight")
plt.close(fig_city)
print("[5] 态城市图已保存: grover_state_city.png")

print("\n" + "=" * 60)
print("  所有可视化图片已保存到当前目录！")
print("=" * 60)
```

---

## ❓ 常见问题解答

### Q1: 为什么 `qc.draw(output="mpl")` 报错？

**A**：请确保已安装可视化依赖：
```bash
pip install "qsl-quantum[viz]"
```
如果仍有问题，请检查 matplotlib 是否正确安装：
```bash
python -c "import matplotlib; print(matplotlib.__version__)"
```

### Q2: 在 Jupyter Notebook 中不显示图片怎么办？

**A**：在 Notebook 开头添加：
```python
%matplotlib inline
```
或者使用：
```python
%matplotlib widget  # 交互式图片
```

### Q3: 中文显示为方块/乱码怎么办？

**A**：这是 matplotlib 中文字体问题。你可以：
1. 设置中文字体：
```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
```
2. 或者使用英文标题/标签避免问题。

### Q4: 最多可以画多少个量子比特的电路图？

**A**：Matplotlib 模式下，15 比特以内可以清晰显示。更多比特时建议使用：
- `qc.draw(output="text")` 文本模式
- 或者裁剪只看你关心的部分电路

### Q5: 如何在无 GUI 的服务器上保存图片？

**A**：使用 Agg 后端：
```python
import matplotlib
matplotlib.use('Agg')  # 必须在 import pyplot 之前
import matplotlib.pyplot as plt

# ... 你的绘图代码 ...

fig.savefig("output.png")  # 不需要 plt.show()
```

---

## 🎯 下一步

恭喜你掌握了 QSL 的可视化功能！接下来你可以：

1. **结合 AI 模块**：在 `07_ai_scientist.md` 中学习如何让 AI 自动求解问题并高亮结果
2. **学习 CLI 工具**：`09_cli_guide.md` 了解命令行快捷操作
3. **探索更多例子**：查看项目 `examples/` 目录下的可视化示例
4. **自定义样式**：尝试创建自己的风格字典，打造专属电路图风格

---

**可视化是理解量子计算的重要窗口！** 📊✨

一张清晰的图胜过千言万语。无论是调试电路、验证结果，还是撰写论文报告，QSL 的可视化工具都能助你一臂之力。
