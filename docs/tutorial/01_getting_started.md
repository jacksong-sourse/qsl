[← 返回文档中心](../index.md)

# QSL 入门教程：从零开始你的第一个量子程序

欢迎使用 QSL（Quantum Search Language）量子计算框架！本教程将带你从零开始，一步步完成 QSL 的安装、验证，并编写你的第一个量子电路——Bell态（纠缠态）。

---

## 📋 前置要求

在开始之前，请确保你的系统满足以下条件：

- **Python 版本**：Python 3.9 或更高版本
- **包管理器**：pip（Python 自带）
- **操作系统**：Windows / macOS / Linux 均可
- **内存**：建议 4GB 以上（模拟更多量子比特需要更多内存）

> 💡 **提示**：可以通过在终端运行 `python --version` 或 `python3 --version` 来检查你的 Python 版本。

---

## 📦 第一步：安装 QSL

QSL 的核心依赖仅为 `numpy`，安装非常快速。打开你的终端（命令提示符、PowerShell 或终端），运行以下命令：

### 最小安装（推荐新手）

```bash
pip install qsl-quantum
```

这个安装包包含了量子电路模拟、Grover 搜索、Shor 算法等核心功能，仅依赖 numpy，5 秒即可完成导入。

### 可选安装（按需要选择）

如果你需要可视化、量子机器学习或真机后端支持，可以选择安装额外依赖：

```bash
# 可视化支持（matplotlib，用于绘制电路图、直方图等）
pip install "qsl-quantum[viz]"

# 量子算法完整支持（scipy，QAOA/VQE/Shor 需要）
pip install "qsl-quantum[algorithms]"

# 量子机器学习（torch + sklearn）
pip install "qsl-quantum[qml]"

# 全部可选依赖（推荐进阶用户）
pip install "qsl-quantum[full]"
```

---

## ✅ 第二步：验证安装

安装完成后，让我们验证 QSL 是否正确安装。

### 方法一：命令行验证

在终端运行以下命令：

```bash
python -c "import qsl; print('QSL 版本:', qsl.__version__)"
```

如果安装成功，你应该看到类似以下输出：

```
QSL 版本: 0.6.3
```

你也可以直接查看版本号：

```bash
python -m qsl --version
```

### 方法二：Python 交互式验证

打开 Python 解释器，尝试导入 QSL：

```python
>>> from qsl import QuantumCircuit
>>> print("QSL 导入成功！")
QSL 导入成功！
```

> 🎉 **恭喜！** 如果以上步骤都没有报错，说明 QSL 已经成功安装在你的系统中了。

---

## 🔬 第三步：你的第一个量子电路——Bell态

现在让我们来编写第一个量子程序：创建一个 **Bell态**（也叫 EPR 对）。Bell态是量子纠缠的经典例子，两个量子比特处于 |00⟩ 和 |11⟩ 的叠加态，测量时两个比特的结果总是相关的。

### Bell态原理

制备Bell态只需要两个量子门：
1. **Hadamard门（H门）**：将第一个量子比特置于叠加态
2. **CNOT门（受控非门）**：以第一个比特为控制位，第二个比特为目标位，产生纠缠

最终得到的量子态为：|Φ⁺⟩ = (|00⟩ + |11⟩)/√2

### 完整代码示例

创建一个名为 `bell_state.py` 的文件，输入以下代码：

```python
from qsl import QuantumCircuit

print("=" * 50)
print("  QSL 第一个量子程序：Bell 态制备")
print("=" * 50)

# 1. 创建一个 2 量子比特的量子电路
qc = QuantumCircuit(2)
print("\n[1] 创建 2 量子比特电路完成")

# 2. 在第 0 个量子比特上施加 Hadamard 门，创建叠加态
qc.h(0)
print("[2] 在 qubit 0 上施加 H 门")

# 3. 施加 CNOT 门，控制位为 0，目标位为 1，创建纠缠
qc.cx(0, 1)
print("[3] 在 (0,1) 上施加 CNOT 门，创建纠缠")

# 4. 打印 ASCII 电路图
print("\n电路结构：")
print(qc.draw())

# 5. 执行电路，模拟 1024 次测量
print("\n[4] 执行模拟（shots=1024）...")
res = qc.execute(shots=1024)

# 6. 查看测量结果统计
print("\n测量结果统计（counts）：")
print(res.counts)

# 7. 查看态向量
print("\n态向量（statevector）：")
sv = res.statevector()
print(sv)

# 8. 以 Dirac 记号美观打印量子态
print("\n量子态（Dirac 记号）：")
res.state.pretty_print()

print("\n" + "=" * 50)
print("  程序执行完成！")
print("=" * 50)
```

### 运行程序

在终端运行这个脚本：

```bash
python bell_state.py
```

### 预期输出

你应该看到类似以下的输出（counts 数值会有微小波动，这是量子测量的概率特性导致的）：

```
==================================================
  QSL 第一个量子程序：Bell 态制备
==================================================

[1] 创建 2 量子比特电路完成
[2] 在 qubit 0 上施加 H 门
[3] 在 (0,1) 上施加 CNOT 门，创建纠缠

电路结构：
     ┌───┐
q_0: ┤ H ├──■──
     └───┘┌─┴─┐
q_1: ─────┤ X ├
          └───┘

[4] 执行模拟（shots=1024）...

测量结果统计（counts）：
{0: 518, 3: 506}

态向量（statevector）：
[0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]

量子态（Dirac 记号）：
0.7071|00⟩ + 0.7071|11⟩

==================================================
  程序执行完成！
==================================================
```

### 结果解释

让我们来解读一下输出结果：

1. **`res.counts`**：这是一个字典，键是测量结果的整数表示，值是出现次数。
   - `0` 对应二进制 `00`，出现了约 512 次
   - `3` 对应二进制 `11`，出现了约 512 次
   - 注意：结果 `01` (1) 和 `10` (2) 几乎不会出现！这就是量子纠缠的体现。

2. **`res.statevector()`**：这是量子态的向量表示，四个复数分别对应 |00⟩, |01⟩, |10⟩, |11⟩ 的概率幅。
   - |00⟩ 和 |11⟩ 的振幅都是 1/√2 ≈ 0.7071
   - 概率是振幅模的平方：(1/√2)² = 0.5，所以各占约 50%

3. **`res.state.pretty_print()`**：以物理学家熟悉的 Dirac 记号美观打印量子态。

---

## 💻 第四步：使用命令行界面（CLI）

QSL 提供了方便的命令行工具，让你无需编写 Python 脚本就能快速体验量子计算。

### 查看 CLI 帮助

```bash
python -m qsl --help
```

### 查看版本

```bash
python -m qsl --version
```

### 运行内置演示

QSL 内置了多个演示程序，包括 Grover 搜索和 AI 量子科学家演示：

#### 1. Grover 搜索演示

```bash
# 交互式选择演示
python -m qsl --demo

# 直接运行指定演示（1-4）
python -m qsl --demo 1
```

演示包括：
- 1: SAT 求解（3 变量）
- 2: 图着色（3 顶点 2 色）
- 3: 恰好一个 1（n=3）
- 4: 大空间搜索（n=6）
- 0: 运行所有演示

#### 2. 中文 AI 演示（无需 API Key）

QSL 内置了 10 个中文 AI 演示模板，无需配置 LLM API Key 即可运行：

```bash
# 列出所有可用的中文演示
python -m qsl --list-demos

# 运行指定编号的演示
python -m qsl --ai-demo 1
```

这些演示涵盖了 Shor 分解、Grover 搜索、GHZ 态制备、量子随机数生成、BB84 量子密钥分发等经典量子计算场景。

### 命令行求解 SAT 问题

你还可以直接在命令行求解布尔可满足性问题：

```bash
python -m qsl --solve 3 "x0 | ~x1" "x1 | x2" "~x0 | ~x2"
```

---

## 📊 进阶：可视化结果（可选）

如果你安装了可视化依赖（`pip install "qsl-quantum[viz]"`），还可以绘制漂亮的电路图和测量直方图：

```python
from qsl import QuantumCircuit
from qsl import plot_histogram
import matplotlib.pyplot as plt

# 创建并执行 Bell 态电路
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
res = qc.execute(shots=4096)

# 绘制 matplotlib 风格电路图
fig1, ax1 = qc.draw(output="mpl", style="iqp")
fig1.suptitle("Bell 态电路", fontsize=14)

# 绘制测量直方图
fig2, ax2 = plt.subplots()
plot_histogram(res.counts, title="Bell 态测量结果（4096 次采样）", ax=ax2)

# 显示所有图形
plt.show()
```

---

## ❓ 常见问题解答（FAQ）

### Q1: 安装时出现 "pip 不是内部或外部命令" 怎么办？

**A**：这说明 Python 的 Scripts 目录没有添加到系统 PATH 中。你可以尝试：
- 使用 `python -m pip install qsl-quantum` 代替 `pip install`
- 或者重新安装 Python，并勾选 "Add Python to PATH" 选项

### Q2: 导入 qsl 时出现 RuntimeWarning 关于真机后端 SDK？

**A**：这是正常现象！QSL 会在你首次访问 IBM 或 AWS 后端时提示缺少对应的 SDK，但本地模拟器完全不受影响。你可以：
- 忽略这个警告，本地模拟完全正常
- 或者设置环境变量 `QSL_SILENT_BACKEND_CHECK=1` 来静默这个警告
- 如果需要使用真机后端，再安装对应的 SDK：`pip install qiskit qiskit-ibm-runtime` 或 `pip install boto3 amazon-braket-sdk`

### Q3: 为什么测量结果每次都不完全一样？

**A**：这是量子力学的概率特性！量子测量是概率性的，就像抛硬币一样。增加 `shots` 参数（比如从 1024 增加到 8192）可以让结果更接近理论概率 50%/50%。

### Q4: 最多可以模拟多少个量子比特？

**A**：
- 态向量模拟：普通笔记本可以模拟 26-28 个量子比特（因为 2^28 ≈ 2.68 亿个复数，约占 4GB 内存）
- 密度矩阵模拟：通常用于 12 比特以下的含噪声模拟
- 如果需要更多比特，可以考虑使用真机后端或近似模拟方法

### Q5: QSL 和 Qiskit 是什么关系？代码可以互通吗？

**A**：
- QSL 是独立实现的全栈量子计算框架，API 设计与 Qiskit 高度相似，降低学习迁移成本
- QSL 内置了与 Qiskit/Cirq 的双向转换器：
  ```python
  from qsl import to_qiskit, from_qiskit
  qiskit_circuit = to_qiskit(qc)  # QSL → Qiskit
  qsl_circuit = from_qiskit(qk)   # Qiskit → QSL
  ```
- 你可以混合使用两个框架的优势

### Q6: 如何在 Jupyter Notebook 中使用 QSL？

**A**：完全支持！和普通 Python 脚本一样导入和使用即可。`qc.draw()` 和 `res.state.pretty_print()` 在 Notebook 中会有更好的显示效果。如果安装了 matplotlib，`draw(output="mpl")` 会直接在 Notebook 中显示电路图。

### Q7: 运行 `python -m qsl --ai-demo` 报错怎么办？

**A**：AI 演示需要 `qsl-quantum[ai]` 依赖。尝试运行：
```bash
pip install "qsl-quantum[ai]"
```
即使没有安装 AI 依赖，核心电路模拟功能也完全正常。

---

## 🎯 下一步

恭喜你完成了 QSL 的入门教程！接下来你可以：

1. **学习更多量子门**：尝试 X、Y、Z、S、T、RX、RY、RZ、SWAP、Toffoli 等门
2. **探索标准电路库**：`from qsl.circuit import library`，里面有 GHZ态、QFT、Grover扩散算子、量子隐形传态等预制电路
3. **尝试量子算法**：Grover 搜索、Shor 分解、QAOA、VQE
4. **阅读更多教程**：继续查看本目录下的其他教程文件
5. **查看示例代码**：访问项目的 README.md 和 tests/ 目录，里面有大量可运行的示例

---

## 📚 快速参考卡

```python
# 基本导入
from qsl import QuantumCircuit

# 创建电路
qc = QuantumCircuit(n_qubits)

# 常用单比特门
qc.h(qubit)      # Hadamard
qc.x(qubit)      # Pauli-X (NOT)
qc.y(qubit)      # Pauli-Y
qc.z(qubit)      # Pauli-Z
qc.s(qubit)      # S 门 (π/2 相位)
qc.t(qubit)      # T 门 (π/4 相位)
qc.rx(theta, qubit)  # 绕 X 轴旋转
qc.ry(theta, qubit)  # 绕 Y 轴旋转
qc.rz(theta, qubit)  # 绕 Z 轴旋转

# 常用两比特门
qc.cx(control, target)    # CNOT (受控非)
qc.cz(control, target)    # 受控 Z
qc.swap(q1, q2)           # SWAP 门

# 执行与结果
res = qc.execute(shots=1024)
res.counts                # 测量统计 {int: int}
res.statevector()         # 态向量 numpy 数组
res.state.pretty_print()  # Dirac 记号打印
qc.draw()                 # ASCII 电路图
```

---

**祝你量子计算之旅愉快！** ⚛️

如果遇到问题，可以：
- 查看项目 README.md
- 检查 tests/ 目录下的测试用例
- 运行 `python -m qsl --help` 查看 CLI 帮助
