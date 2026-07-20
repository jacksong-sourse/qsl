[← 返回文档中心](../index.md)

# 可视化 API 参考

qsl.viz 模块提供基于 matplotlib 的出版级量子电路和量子态可视化功能。所有函数延迟导入 matplotlib，未安装时会给出友好安装提示。

---

## 安装依赖

可视化功能需要 matplotlib：

```bash
pip install matplotlib
# 或完整安装
pip install "qsl-quantum[full]"
```

---

## draw_circuit_mpl

用 matplotlib 绘制出版级量子电路图。

```python
draw_circuit_mpl(
    circuit: QuantumCircuit,
    ax=None,
    style: Optional[Dict] = None,
    fold: int = -1
) -> Tuple
```

**参数：**
- `circuit` (QuantumCircuit)：要绘制的量子电路
- `ax`：可选的 matplotlib Axes 对象；不传则自动创建 figure
- `style`：可选 dict 或字符串，自定义绘图风格
  - 字符串可选：`"iqp"` / `"default"` / `"bw"` / `"clifford"` / `"textbook"`
  - dict 可覆盖键：`line_color` / `gate_facecolor` / `gate_edgecolor` / `text_color` / `background` / `fontsize`
- `fold` (int)：>0 时每行最多 fold 列后换行；-1（默认）不换行

**返回值：**
- `(fig, ax)`：matplotlib Figure 和 Axes 对象

**预设风格：**

| 风格名 | 说明 | 门填充色 |
|--------|------|----------|
| `default` | 默认学术风格 | 淡蓝 #DCE9F7 |
| `iqp` | IQP/IBM 风格（米黄） | #FAF8D3 |
| `bw` | 黑白风格 | 白色 |
| `clifford` | Clifford 风格（绿色） | #D4FFD4 |
| `textbook` | 教材风格（紫色） | #E6E6FA |

**示例：**
```python
from qsl import QuantumCircuit
from qsl.viz import draw_circuit_mpl
import matplotlib.pyplot as plt

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.ccx(0, 1, 2)
qc.measure_all()

# 使用默认风格
fig, ax = draw_circuit_mpl(qc)
plt.savefig("circuit_default.png", dpi=150, bbox_inches="tight")

# 使用 iqp 风格
fig2, ax2 = draw_circuit_mpl(qc, style="iqp")
plt.savefig("circuit_iqp.png", dpi=150, bbox_inches="tight")

# 自定义风格
custom_style = {
    "gate_facecolor": "#FFE4E1",
    "gate_edgecolor": "#8B0000",
    "fontsize": 12,
}
fig3, ax3 = draw_circuit_mpl(qc, style=custom_style, fold=5)
plt.savefig("circuit_custom.png", dpi=150, bbox_inches="tight")
```

---

## plot_bloch_sphere

绘制单比特 Bloch 球，展示态矢量在 3D 单位球上的位置。

```python
plot_bloch_sphere(state_or_vector, ax=None) -> Tuple
```

**参数：**
- `state_or_vector`：支持三种输入格式：
  - `QuantumState`（任意比特数，取 qubit=0 的 Bloch 向量）
  - `(x, y, z)` 三元组（Bloch 向量坐标）
  - 2 维复向量（单比特态向量）
- `ax`：可选的 3D Axes；不传则自动创建

**返回值：**
- `(fig, ax)`：matplotlib Figure 和 3D Axes 对象

**Bloch 球说明：**
- 北极：|0⟩ (z=+1)
- 南极：|1⟩ (z=-1)
- x 轴：(|0⟩+|1⟩)/√2
- y 轴：(|0⟩+i|1⟩)/√2

**示例：**
```python
from qsl import QuantumCircuit
from qsl.viz import plot_bloch_sphere
import matplotlib.pyplot as plt

# 方式1：从 QuantumState 绘制
qc = QuantumCircuit(1)
qc.h(0)  # |+⟩ 态
result = qc.execute(shots=1)
fig, ax = plot_bloch_sphere(result.state)
plt.savefig("bloch_hadamard.png", dpi=150)

# 方式2：从态向量绘制
import numpy as np
state = np.array([1, 1j], dtype=complex) / np.sqrt(2)  # |i+⟩
fig2, ax2 = plot_bloch_sphere(state)
plt.savefig("bloch_y.png", dpi=150)

# 方式3：直接指定 Bloch 向量坐标
fig3, ax3 = plot_bloch_sphere((0, 0, 1))  # |0⟩
plt.savefig("bloch_0.png", dpi=150)
```

---

## plot_state_city

绘制密度矩阵 3D 城市图（实部和虚部两个子图）。

```python
plot_state_city(rho_or_state, ax=None) -> Tuple
```

**参数：**
- `rho_or_state`：支持多种输入格式：
  - `QuantumState`（自动转换为 ρ = |ψ⟩⟨ψ|）
  - `DensityMatrix`
  - 密度矩阵 numpy 数组
  - 态向量（自动转换为外积）
- `ax`：可选的两个 3D Axes 组成的元组 `(ax_re, ax_im)`；不传则自动创建

**返回值：**
- `(fig, (ax_re, ax_im))`：Figure 和两个 3D Axes（实部/虚部）

**示例：**
```python
from qsl import QuantumCircuit
from qsl.viz import plot_state_city
import matplotlib.pyplot as plt

# Bell 态密度矩阵
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
result = qc.execute(shots=1)

fig, (ax_re, ax_im) = plot_state_city(result.state)
plt.suptitle("Bell State |Φ+⟩ Density Matrix", fontsize=14)
plt.savefig("state_city_bell.png", dpi=150, bbox_inches="tight")
```

---

## plot_amplitudes

绘制基态振幅概率柱状图。

```python
plot_amplitudes(state, ax=None) -> Tuple
```

**参数：**
- `state`：`QuantumState` 或长度为 2^n 的复向量
- `ax`：可选的 Axes；不传则自动创建

**返回值：**
- `(fig, ax)`：matplotlib Figure 和 Axes 对象

**示例：**
```python
from qsl import QuantumCircuit
from qsl.circuit.library import ghz_state
from qsl.viz import plot_amplitudes
import matplotlib.pyplot as plt

# GHZ 态的振幅分布
qc = ghz_state(3)
result = qc.execute(shots=1)

fig, ax = plot_amplitudes(result.state)
ax.set_title("GHZ State (n=3) Probabilities")
plt.savefig("amplitudes_ghz.png", dpi=150, bbox_inches="tight")
```

---

## plot_qsphere

绘制 Q 球（简化版）：3D 球面上每个基态一个点，按 Hamming 权重分环。

```python
plot_qsphere(state, ax=None) -> Tuple
```

**参数：**
- `state`：`QuantumState` 或长度为 2^n 的复向量
- `ax`：可选的 3D Axes；不传则自动创建

**返回值：**
- `(fig, ax)`：matplotlib Figure 和 3D Axes 对象

**Q 球说明：**
- 点的大小 = 该基态的概率
- 点的颜色 = 相位（HSV 色轮）
- 按 Hamming 权重（比特串中 1 的个数）分纬度环

**示例：**
```python
from qsl import QuantumCircuit
from qsl.viz import plot_qsphere
import matplotlib.pyplot as plt

# 5 比特随机电路
from qsl.circuit.library import random_circuit
qc = random_circuit(4, depth=5, seed=42)
result = qc.execute(shots=1)

fig, ax = plot_qsphere(result.state)
plt.savefig("qsphere_random.png", dpi=150)
```

---

## plot_histogram

绘制测量计数直方图。

```python
plot_histogram(
    counts: Dict,
    ax=None,
    highlight=None,
    title: Optional[str] = None,
    sort: bool = True
) -> Tuple
```

**参数：**
- `counts` (Dict)：`{比特串/整数: 次数}` 字典（如 `ExecutionResult.get_counts()` 返回值）
- `ax`：可选的 Axes；不传则自动创建
- `highlight`：需高亮的比特串集合（高亮为橙色 #E8862E，其余为淡蓝）
- `title` (Optional[str])：图标题
- `sort` (bool)：True（默认）时按计数降序排列

**返回值：**
- `(fig, ax)`：matplotlib Figure 和 Axes 对象

**示例：**
```python
from qsl import QuantumCircuit
from qsl.viz import plot_histogram
import matplotlib.pyplot as plt

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
result = qc.execute(shots=1024)

# 基本直方图
counts = result.get_counts(binary=True)
fig, ax = plot_histogram(counts, title="GHZ State Measurement (1024 shots)")
plt.savefig("histogram_ghz.png", dpi=150, bbox_inches="tight")

# 高亮特定结果
fig2, ax2 = plot_histogram(
    result.counts,
    highlight={0b000, 0b111},
    title="GHZ State (highlighted: |000⟩, |111⟩)"
)
plt.savefig("histogram_highlight.png", dpi=150, bbox_inches="tight")
```

---

## QuantumCircuit.draw() 快捷方法

QuantumCircuit 类本身提供了 draw() 方法，支持文本和 matplotlib 两种输出：

```python
# 文本绘制（ASCII 艺术，无需 matplotlib）
text_str = qc.draw(output="text")
print(text_str)

# matplotlib 绘制（等价于 draw_circuit_mpl）
fig, ax = qc.draw(output="mpl", style="iqp", fold=10)
```

**示例：**
```python
from qsl import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# 文本输出
print(qc.draw())
# 输出:
#      ┌───┐
# q_0: ┤ H ├──■──
#      └───┘┌─┴─┐
# q_1: ─────┤ X ├
#           └───┘
```

---

## 完整可视化示例

```python
from qsl import QuantumCircuit
from qsl.viz import (
    draw_circuit_mpl,
    plot_bloch_sphere,
    plot_state_city,
    plot_amplitudes,
    plot_qsphere,
    plot_histogram,
)
import matplotlib.pyplot as plt
import numpy as np

# 构建电路
qc = QuantumCircuit(3, name="Example")
qc.h(0)
qc.cx(0, 1)
qc.ccx(0, 1, 2)
result = qc.execute(shots=2048, seed=42)

# 创建 2x3 子图组合
fig = plt.figure(figsize=(18, 10))

# 1. 电路图
ax1 = fig.add_subplot(2, 3, 1)
draw_circuit_mpl(qc, ax=ax1, style="iqp")
ax1.set_title("Quantum Circuit")

# 2. 振幅柱状图
ax2 = fig.add_subplot(2, 3, 2)
plot_amplitudes(result.state, ax=ax2)

# 3. 计数直方图
ax3 = fig.add_subplot(2, 3, 3)
plot_histogram(result.get_counts(binary=True), ax=ax3, title="Measurements")

# 4. Bloch 球 (qubit 0)
ax4 = fig.add_subplot(2, 3, 4, projection="3d")
plot_bloch_sphere(result.state, ax=ax4)

# 5. 态城市图（单独大图）
fig2, (ax_re, ax_im) = plot_state_city(result.state)
fig2.suptitle("Density Matrix", fontsize=14)

plt.tight_layout()
plt.show()
```

---

## 无头环境运行

在无 GUI 的服务器/CI 环境下，matplotlib 会自动使用 Agg 后端：

```python
import matplotlib
matplotlib.use("Agg")  # 显式设置无头后端

from qsl.viz import draw_circuit_mpl
# ... 绘图代码 ...
plt.savefig("output.png")  # 仅保存文件，不弹窗显示
```

---

## 颜色与样式常量

| 常量名 | 色值 | 用途 |
|--------|------|------|
| `_FACE_BLUE` | #4C86C6 | 柱状图主色、密度矩阵实部 |
| `_LIGHT_BLUE` | #9FC5E8 | 直方图非高亮色、Q 球面 |
| `_ORANGE` | #E8862E | 高亮色、密度矩阵虚部 |
| `_EDGE` | #2E5A87 | 边框色 |
