<div align="center">

# QSL — Quantum Search Language

**对标 Qiskit 的全栈量子计算框架 · 中文友好 · AI 驱动**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/qsl-quantum?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/qsl-quantum/)
[![License](https://img.shields.io/badge/License-MIT-4CC61E)](./LICENSE)
[![Tests](https://img.shields.io/badge/Tests-731%2B-00C851)](https://github.com/qsl-quantum/qsl/actions)
[![Minimal Deps](https://img.shields.io/badge/min%20deps-numpy%20only-important)](#-安装)

[特性](#-特性) ·
[安装](#-安装) ·
[快速开始](#-快速开始) ·
[电路 API](#-电路-api) ·
[算法](#-量子算法) ·
[AI 科学家](#-ai-量子科学家) ·
[可视化](#-可视化) ·
[CLI](#-命令行) ·
[Qiskit 互通](#-qiskit-互通)

</div>

---

## ✨ 特性

QSL (Quantum Search Language) 是一个面向研究与教学的全栈量子计算框架。核心依赖 **仅 numpy**，装完即可用。

- **🔧 电路层（对标 Qiskit）**
  - `QuantumCircuit` 类：门追加/插入/删除、`inverse()` 逆置、`compose()` 拼接、`decompose()` 分解、`transpile()` 优化
  - 符号参数 `Parameter` + `bind_parameters()` / `assign_parameters()`，VQE/QAOA/QML 基础
  - 通用受控操作 `gate.control(n)`、门幂次 `gate.power(k)`、门逆 `gate.inverse()`
  - 完整门库：Pauli/H/S/T/SX 族、RX/RY/RZ/RXX/RYY/RZZ、CRX/CRY/CRZ/CU/CP、CH/CS/CSdg/CT/CTdg、iSWAP/ECR/DCX/CSWAP/MCMT
- **📚 标准电路库**：Bell、GHZ、W 态、QFT、QPE、Grover 扩散算子、量子隐形传态、随机电路、量子游走
- **💻 高性能模拟**
  - 全振幅向量化态向量模拟（上限 26–28 比特）
  - 密度矩阵路径，内置噪声模型：退极化、振幅阻尼、相位阻尼、读出误差
  - 可选 cupy GPU 加速
  - 二分查找采样、Pauli 串直接解析期望值（免采样）
- **🔁 生态互通**：OpenQASM 2.0 导入导出、QASM 3.0 导出；`to_qiskit()` / `from_qiskit()` / `to_cirq()` 双向转换
- **📐 可视化**：matplotlib 出版级电路图、Bloch 球、态城市图（city plot）、Q 球、振幅柱状图、直方图（标记正确解）
- **🧮 核心算法**：QFT、Shor 大数分解、Grover（BBHT 未知解数搜索、布尔电路 Oracle）、QAOA（Max-Cut 等组合优化）、VQE（变分量子本征求解）
- **🤖 AI 量子科学家**
  - `LLMProvider` 抽象层：OpenAI / DeepSeek / Kimi / 通义千问 / Ollama 一处配置全局切换；国内默认走 DeepSeek/Kimi
  - 自然语言问题 → 量子算法自动选择、参数抽取、电路编译、执行、验证、中文解释
  - 自动验证器：Shor 结果回乘、SAT 解代回、QAOA 对比经典基线、Grover 解校验；失败自动重规划
  - 10 个免 Key 中文演示模板：分解、3-SAT、数独、最大割、TSP、图着色、Grover、GHZ、QRNG、BB84
- **⚙️ 工程化**：pytest 731+ 测试、GitHub Actions CI（3.9–3.12）、ruff 代码检查、wheel 构建验证

---

## 📦 安装

**最小安装（仅 numpy，5 秒可 import）：**

```bash
pip install qsl-quantum
```

**按场景安装可选依赖：**

```bash
pip install "qsl-quantum[viz]"          # matplotlib 可视化
pip install "qsl-quantum[algorithms]"   # scipy（QAOA/VQE/Shor 需要）
pip install "qsl-quantum[qml]"          # torch + sklearn（量子机器学习）
pip install "qsl-quantum[cross]"        # qiskit + cirq（转换器/交叉验证）
pip install "qsl-quantum[ai]"           # openai + langchain（AI 科学家）
pip install "qsl-quantum[full]"         # 全部可选依赖
pip install "qsl-quantum[dev]"          # 开发测试工具
```

**验证安装：**

```bash
python -c "import qsl; print(qsl.__version__)"
# 0.6.1
python -m qsl --version
```

---

## 🚀 快速开始

### 1. 第一个量子电路：Bell 态

```python
from qsl import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)       # Hadamard 叠加
qc.cx(0, 1)   # CNOT 纠缠

# 执行并查看结果
res = qc.execute(shots=1024)
print(res.counts)           # → {0: ~512, 3: ~512}
print(res.statevector())    # → [0.707+0j 0 0 0.707+0j]
res.state.pretty_print()    # → 0.7071|00⟩ + 0.7071|11⟩
```

### 2. Grover 搜索（解 SAT 问题）

```python
from qsl import solve_sat

# 求解 3-SAT: (x0 ∨ ¬x1) ∧ (x1 ∨ x2) ∧ (¬x0 ∨ ¬x2)
# CNF 格式: 每个子句是文字列表 (变量从 1 开始, 负数表示取反)
result = solve_sat(
    cnf_clauses=[[1, -2], [2, 3], [-1, -3]],
    n_qubits=3,
    shots=10
)
print(result.get_solutions())   # → 满足条件的解列表
# 也可以直接用布尔字符串（通过核心 GroverSearch.search_expressions）
```

### 3. Shor 大数分解

```python
from qsl.algorithms import ShorSolver

factors = ShorSolver(15).factor()   # → [3, 5]
print(f"15 = {factors[0]} × {factors[1]}")
```

### 4. 参数化电路（QAOA/VQE 基础）

```python
from qsl import QuantumCircuit, Parameter
import numpy as np

theta = Parameter("θ")
qc = QuantumCircuit(2)
qc.h(0); qc.h(1)
qc.rzz(theta, 0, 1)     # 注意：角度在前，比特在后（与 Qiskit 一致）
qc.rx(0.3, 0); qc.rx(0.3, 1)

bound = qc.assign({"θ": np.pi/2})   # 绑定参数
counts = bound.measure_all(shots=1000)
```

### 5. AI 量子科学家（中文自然语言）

```python
from qsl import QuantumAgent

agent = QuantumAgent(verbose=True)
report = agent.run("把 15 分解质因数")
print(report.to_markdown())    # 任务 → 算法 → 电路 → 结果 → 验证状态
```

无 LLM Key 时自动回退到规则路由与参数抽取，开箱可用。

---

## 🧩 电路 API

### 构建电路

```python
from qsl import QuantumCircuit, Parameter

qc = QuantumCircuit(3, name="demo", global_phase=0.0)

# 单比特门
qc.h(0); qc.x(1); qc.y(2); qc.z(0)
qc.s(0); qc.t(1); qc.sx(2)          # S / T / √X
qc.sdg(0); qc.tdg(1); qc.sxdg(2)    # 共轭转置

# 参数化门（角度 / 控制位 / 目标位，对齐 Qiskit）
qc.rx(0.5, 0); qc.ry(0.5, 1); qc.rz(0.5, 2)
qc.p(0.3, 0); qc.u(1.0, 0.2, 0.3, 0)    # θ, φ, λ

# 两比特门
qc.cx(0, 1); qc.cy(0, 1); qc.cz(0, 1)
qc.ch(0, 1); qc.cs(0, 1); qc.ct(0, 1)
qc.crx(0.7, 0, 1); qc.cry(0.7, 0, 1); qc.crz(0.7, 0, 1)
qc.cp(0.5, 0, 1); qc.cu(0.3, 0.4, 0.5, 0, 1)   # θ, φ, λ, c, t [, γ]
qc.swap(0, 1); qc.iswap(0, 1); qc.ecr(0, 1); qc.dcx(0, 1)
qc.rxx(0.5, 0, 1); qc.ryy(0.5, 0, 1); qc.rzz(0.5, 0, 1)

# 三比特 / 多比特
qc.ccx(0, 1, 2)             # Toffoli
qc.cswap(0, 1, 2)           # Fredkin
qc.mcx([0,1], 2)            # 多控制 X
qc.mcz([0,1,2])             # 多控制 Z
qc.barrier()
```

### 电路变换

```python
qc_inv  = qc.inverse()           # 电路逆
qc2     = qc.compose(qc_inv)     # 电路拼接
qc_dec  = qc.decompose()         # 分解到基础门集
qc_t    = qc.transpile(optimization_level=2)  # 编译优化
qc_rev  = qc.reverse_bits()      # 比特序反转

# 门级变换
from qsl.quantum_gates import H
cc_h = H.control(2)              # 2 个控制位的 H 门
h2   = H.power(0.5)              # H 的 1/2 幂
hdag = H.inverse()               # H†
```

### 执行与测量

```python
# 态向量模拟
res = qc.execute(shots=1024, seed=42)
counts = res.counts                  # dict[int,int]
sv     = res.statevector()           # 复数态向量
probs  = res.probabilities_dict()    # {bitstring: prob}

# 便捷方法
counts = qc.measure_all(shots=1000)

# 密度矩阵 + 噪声模拟
from qsl import NoiseModel
noise = NoiseModel(
    depolarizing=0.01,       # 退极化 1%
    amplitude_damping=0.005, # T1 振幅阻尼
    phase_damping=0.01,      # T2 相位阻尼
    readout_error=0.02       # 读出误码 2%
)
res_noisy = qc.execute_density(shots=1024, noise=noise)

# 解析期望值（无需采样）
ev_z  = qc.expectation("IZ")                # ⟨Z1⟩（第 0 位是 Pauli I，第 1 位是 Z；左=低比特）
ev_zz = qc.expectation("ZZ")                # ⟨Z0 Z1⟩
ev_xx_zz = qc.expectation([(0.5, "ZZ"), (-0.3, "XX")])
```

### 导入导出

```python
from qsl import (QuantumCircuit,
    dumps_qasm2, loads_qasm2, dumps_qasm3,
    to_qiskit, from_qiskit, to_cirq)

# QASM 互通
qasm_str = dumps_qasm2(qc)
qc2 = loads_qasm2(qasm_str)
print(dumps_qasm3(qc))

# Qiskit / Cirq 双向转换
qk = to_qiskit(qc)
qc_back = from_qiskit(qk)
cq = to_cirq(qc)
```

### 标准电路库

```python
from qsl.circuit import library

qc_bell   = library.bell_state("phi+")     # |Φ+⟩
qc_ghz    = library.ghz_state(4)           # 4 比特 GHZ
qc_w      = library.w_state(4)             # 4 比特 W 态
qc_qft    = library.qft(4)                 # 4 比特 QFT
qc_iqft   = library.qft(4, inverse=True)   # IQFT
qc_qpe    = library.qpe(U_gate, n_counting=4)  # 量子相位估计（U_gate 为要估计本征相位的酉门）
qc_diff   = library.grover_diffusion(4)    # Grover 扩散算子
qc_tp     = library.teleportation()        # 量子隐形传态
qc_rand   = library.random_circuit(5, depth=10, seed=0)
qc_walk   = library.quantum_walk_cycle(8)  # 循环图量子游走
```

---

## 🧮 量子算法

### QFT — 量子傅里叶变换

```python
from qsl.circuit import library

qc = library.qft(4)     # 4 比特 QFT 电路 (QuantumCircuit 对象)
print(qc.draw())        # ASCII 电路图
res = qc.execute()
```

### QAOA — 量子近似优化（以 Max-Cut 为例）

```python
from qsl.algorithms import QAOA
import numpy as np

# 4 节点环图 Max-Cut 的邻接矩阵
adj = np.array([
    [0,1,0,1],
    [1,0,1,0],
    [0,1,0,1],
    [1,0,1,0],
], dtype=float)
cost = QAOA.maxcut_cost_matrix(adj)

qaoa = QAOA(n_qubits=4, cost_matrix=cost, p=2)
qaoa.optimize(maxiter=200)
print("最优切割值:", qaoa.optimal_value)
print("最优比特串:", bin(qaoa.optimal_bitstring))
```

### VQE — 变分量子本征求解

```python
from qsl.algorithms import VQE

# 求 H = 0.5*Z0 - 0.2*X0*X1 的基态能量（Pauli 串长度必须等于 n_qubits）
vqe = VQE(
    n_qubits=2,
    hamiltonian_pauli_terms=[(0.5, "IZ"), (-0.2, "XX")],
    n_layers=2,
)
vqe.optimize(maxiter=200)
print("基态能量:", vqe.ground_energy)
```

---

## 🤖 AI 量子科学家

QSL 内置中文 AI 科学家，支持自然语言驱动的量子计算。零 Key 时走规则引擎（支持中文参数抽取），有 Key 时调用 LLM 处理复杂任务。

```python
from qsl import QuantumAgent, create_provider, set_default_provider

# 配置 LLM（可选，不配置则走规则回退）
# 自动探测环境变量: DEEPSEEK_API_KEY / MOONSHOT_API_KEY / OPENAI_API_KEY / DASHSCOPE_API_KEY
provider = create_provider()
if provider is not None:
    set_default_provider(provider)

agent = QuantumAgent(verbose=True)
report = agent.run("把 15 分解质因数")
# → 自动选 Shor、构造电路、执行、验证 (3×5=15)、输出结构化报告
print(report.to_markdown())
```

**10 个内置中文演示（无需 Key）：**

```bash
python -m qsl --list-demos       # 列出演示
python -m qsl --ai-demo 1        # 运行第 1 个演示
```

代码中运行：

```python
from qsl import run_demo, list_demos
for d in list_demos():
    print(d['id'], d['name'], d['desc'])
report = run_demo(1, verbose=True)
print(report.to_markdown())
```

**配置 LLM（国内推荐 DeepSeek / Kimi）：**

```bash
# 任选其一
export DEEPSEEK_API_KEY="sk-..."
export MOONSHOT_API_KEY="sk-..."      # Kimi
export OPENAI_API_KEY="sk-..."
export DASHSCOPE_API_KEY="sk-..."     # 通义千问

# 可选：显式指定
export QSL_LLM=deepseek               # deepseek / kimi / openai / qwen / ollama
export QSL_LLM_MODEL=deepseek-chat
```

---

## 📊 可视化

需要安装 `pip install "qsl-quantum[viz]"`。

```python
import matplotlib.pyplot as plt
from qsl import QuantumCircuit
from qsl import plot_histogram, plot_bloch_sphere, plot_state_city

qc = QuantumCircuit(2); qc.h(0); qc.cx(0,1)
res = qc.execute(shots=4096)

# 1. 电路图（matplotlib 出版级）
fig, ax = qc.draw(output="mpl", style="iqp")

# 2. 测量直方图（支持标记正确解，Grover 演示刚需）
plot_histogram(res.counts, title="Bell 态测量")

# 3. 态可视化
# plot_bloch_sphere(state)        # 单比特 Bloch 球
# plot_state_city(density_matrix) # 密度矩阵 3D 城市图
# plot_amplitudes(sv)             # 振幅柱状图
# plot_qsphere(sv)                # Q 球
plt.show()
```

---

## 💻 命令行

```bash
python -m qsl                      # 交互式启动
python -m qsl --version            # 版本号
python -m qsl --help               # 帮助
python -m qsl --demo               # 列出并运行 Grover 演示
python -m qsl --demo 1             # 直接运行第 1 个 Grover 演示
python -m qsl --solve 3 "x0|~x1" "x1|x2" "~x0|~x2"   # 命令行 SAT
python -m qsl --file test.qsl      # 运行 .qsl DSL 文件
python -m qsl --list-demos         # 列出 10 个中文 AI 演示
python -m qsl --ai-demo 1          # 运行中文 AI 演示
```

---

## 🔁 Qiskit 互通

QSL 的门参数顺序和全局相位约定与 Qiskit 保持一致（角度在前、比特在后），保证逐位数值对比通过。

```python
from qsl import QuantumCircuit, to_qiskit, from_qiskit
import numpy as np

# qsl → qiskit
qc = QuantumCircuit(2); qc.h(0); qc.crx(0.5, 0, 1)
qk = to_qiskit(qc)

# qiskit → qsl
from qiskit.circuit.library import QFT
qc_back = from_qiskit(QFT(4))
```

交叉验证测试覆盖所有标准门，逐振幅误差 < 1e-10。

---

## 📁 项目结构

```
qsl/
├── circuit/        # 电路对象模型（QuantumCircuit/Gate/Parameter/QASM/转换器/可视化/标准库）
├── core/           # 状态向量/密度矩阵模拟器、Grover、Oracle、布尔解析器
├── algorithms/     # QFT、Shor、QAOA、VQE
├── qml/            # QuantumLayer、QNN、量子核、QSVM、QGAN
├── backends/       # 本地模拟器、IBM/AWS Braket 真机后端
├── compiler/       # DSL 解析器、编译器、电路优化器、布局映射、误差缓解
├── viz/            # matplotlib 可视化（电路图 / Bloch 球 / 城市图 / 直方图）
├── ai/             # LLMProvider、自然语言翻译器、智能体、自动验证、中文演示、解释器
├── meta/           # 算法搜索、AI 编译器、定理猜想
├── network/        # 分布式节点、量子区块链（演示用）
├── pipelines/      # 药物发现 / 密码分析 / 投资组合应用示例
└── utils/          # 异常与参数校验
```

---

## 🛠️ 开发

```bash
git clone https://gitee.com/song-jack/qsl.git
cd qsl
pip install -e ".[dev,viz,algorithms]"

pytest                                # 运行全部 731+ 测试
pytest --cov=qsl --cov-report=term    # 覆盖率
ruff check qsl                        # 代码检查
```

---

## 📜 变更日志

参见 [CHANGELOG.md](./CHANGELOG.md)，遵循 [Keep a Changelog](https://keepachangelog.com/)。

- **v0.6.1**（2026-07-19）：修复参数门顺序、CLI 增强、Qiskit 兼容 API、Dirac 记号打印、BBHT 重启、移除过时 setup.py
- **v0.6.0**（2026-07-19）：电路层、QASM、转换器、可视化、噪声模拟、LLMProvider、自动验证、中文演示

---

## 📄 许可证

MIT License © 2026 Song Ziming

---

## 🙋 FAQ

**Q：最小依赖到底是什么？**
A：仅 numpy。`pip install qsl-quantum` 装完即可运行模拟器、Grover、Shor（Shor/QAOA/VQE 大整数部分需要 scipy，用 `[algorithms]`）。

**Q：与 Qiskit 的关系？**
A：QSL 是独立实现的量子计算框架，API 与 Qiskit 高度相似以降低迁移成本；通过 `to_qiskit()`/`from_qiskit()` 双向互通，可以混用。

**Q：模拟上限多少比特？**
A：向量化态向量路径在普通笔记本上可模拟到 26–28 比特（内存限制）；密度矩阵路径用于小比特数含噪声模拟。

**Q：不用 OpenAI Key 能用 AI 功能吗？**
A：可以。零 Key 时自动启用中文规则引擎：支持分解 / 搜索 / 优化 / GHZ 制备等常见任务的中文意图识别和参数抽取。DeepSeek/Kimi 国内可直接使用。
