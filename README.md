<div align="center">

# 🚀 QSL — Quantum Search Language v0.5.0

**用一句话描述你想解决什么问题，剩下的交给量子计算。**

<p align="center">
  <a href="#-安装"><img src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-4CC61E" alt="License"></a>
  <img src="https://img.shields.io/badge/tests-398%20passed-00C851" alt="Tests">
  <img src="https://img.shields.io/badge/version-0.5.0-268BD2" alt="Version">
  <a href="https://pypi.org/project/qsl-quantum/"><img src="https://img.shields.io/badge/pypi-qsl--quantum-FFD43B?logo=pypi&logoColor=black" alt="PyPI"></a>
</p>

</div>

---

## ✨ 特性总览

一个 **全栈量子计算框架**，从声明式量子搜索到AI驱动的量子科学家：

| 层级 | 模块 | 功能 |
|:----:|:-----|:-----|
| **1️⃣** | **量子门 & 算法** | 50+ 量子门 · QFT · **Shor 量子相位估计** · **QAOA** · **VQE (parameter-shift)** |
| **2️⃣** | **量子机器学习** | **向量化 QuantumLayer** · QNN · 量子核 · QSVM · **可微 QGAN (Straight-Through)** |
| **3️⃣** | **后端 & 编译器** | 高性能模拟器 · IBM/AWS 真机 · 门融合 · 布局映射 · 零噪声外推 |
| **4️⃣** | **AI 量子科学家** | 自然语言→量子程序 · 自主智能体 · 假设检验 · 自动发现 |
| **5️⃣** | **元系统 & 网络** | 遗传电路搜索 · 量子定理证明 (Grover) · 分布式节点 · 量子区块链 |

---

## 🎯 5 秒上手

```python
from qsl import QSLProgram, compile_and_run

# 声明你想找什么：解一个 3-SAT 问题
program = QSLProgram(
    name="3-SAT",
    n_qubits=3,
    premises=["x0 | ~x1", "x1 | x2", "~x0 | ~x2"],
    shots=10
)
result = compile_and_run(program)
print(result.get_solutions())  # → [3, 4] 即 |011⟩ 和 |100⟩
```

**不需要懂量子力学。** 框架会自动编译最优量子电路并执行。

---

## 📦 核心功能演示

### 🔢 Shor 算法 — 整数因子分解

```python
from qsl import ShorSolver

# 量子相位估计实现周期查找 (支持超过旧12-qubit阈值)
solver = ShorSolver(21, max_control_qubits=12)
factors = solver.factor()
print(f"21 = {' × '.join(map(str, factors))}")  # → 21 = 3 × 7
```

### 🧬 VQE — 分子基态能量计算

```python
from qsl import VQE
import numpy as np

# 计算氢分子 H₂ 基态能量 (parameter-shift 梯度)
vqe = VQE(4, VQE.h2_hamiltonian(), n_layers=2)
energy, ground_state = vqe.optimize(maxiter=100)
print(f"H₂ ground energy: {energy:.4f} Hartree")
```

### 📊 QAOA — 组合优化 (MaxCut / 投资组合)

```python
from qsl import QAOA
import numpy as np

# MaxCut 问题
adj = np.array([[0,1,0],[1,0,1],[0,1,0]])
Q = QAOA.maxcut_cost_matrix(adj)
qaoa = QAOA(3, Q, p=2, encoding="qubo")
params, cost = qaoa.optimize()
bitstring, value = qaoa.get_optimal_bitstring()
```

### 🔍 Grover — 布尔表达式量子搜索

```python
from qsl import GroverSearch, solve_sat
from qsl.core.parser import parse_bool

# 直接从布尔表达式构建量子 Oracle (无经典枚举!)
expr = parse_bool("x0 & x1 & ~x2")
grover = GroverSearch(n_qubits=4, verbose=False)
result = grover.search_expressions([expr], num_solutions=1)
```

### 🤖 量子机器学习层

```python
from qsl import QuantumLayer
import torch

# 完全向量化的 PyTorch 层 (无 Python for 循环)
layer = QuantumLayer(n_qubits=4, n_features=4, encoding="angle")
x = torch.randn(8, 4)  # batch of 8
out = layer(x)          # shape: (8, 4) — 可端到端训练
```

---

## 🛠 安装

```bash
# 核心 (仅依赖 numpy, ~100KB)
pip install qsl-quantum

# 量子算法 (scipy)
pip install qsl-quantum[algorithms]

# 量子机器学习 (torch, scikit-learn)
pip install qsl-quantum[qml]

# 真实量子硬件
pip install qsl-quantum[ibm]      # IBM Quantum
pip install qsl-quantum[aws]      # AWS Braket

# 全部依赖
pip install qsl-quantum[full]
```

> ⚠️ 导入时会提示未安装 SDK 的后端不可用，本地模拟器始终可用。

---

## 📁 项目结构

```
qsl/
├── core/           量子态 · 布尔解析器 · Grover (真正量子Oracle)
├── compiler/       DSL · 编译器 · 门融合/交换 · 错误缓解
├── backends/       模拟器 · IBM · AWS Braket · 自动选择
├── algorithms/     QFT · Shor (量子相位估计) · QAOA · VQE (parameter-shift)
├── qml/            QuantumLayer (向量化) · QNN · QSVM · QGAN (可微)
├── ai/             LLM 翻译器 · 量子智能体 · 假设检验
├── pipelines/      药物发现 · 密码分析 · 投资组合优化 (真正QAOA)
├── meta/           遗传电路搜索 · 量子定理证明 (Grover)
├── network/        分布式节点 · 量子区块链
└── utils/          异常体系 · 输入验证
tests/              398 个单元测试
```

---

## ✅ 运行测试

```bash
pip install -e ".[dev]"
pytest tests/ -v
# ======= 398 passed in ~9s =======
```

---

## 🔬 v0.5.0 重大修复 (相比 v0.4.1)

| 问题 | 修复 |
|:-----|:-----|
| Grover Oracle 经典全枚举 2ⁿ 态 | ✅ 从布尔表达式直接构建量子电路 |
| Shor >12 qubit 退回经典 | ✅ 正确量子相位估计 + 逆QFT |
| QuantumLayer Python for 循环 | ✅ 全部改为 numpy/torch 批量运算 |
| QGAN torch.bernoulli 不可微 | ✅ Straight-Through Estimator |
| DensityMatrix 转 list-of-lists | ✅ 全程保持 numpy ndarray |
| 药物发现随机哈密顿量 | ✅ 支持 OpenFermion+PySCF 真实计算 |
| IBM 后端经典枚举Oracle | ✅ 量子Oracle电路构建 |
| 投资组合经典线性求解 | ✅ 逐点运行QAOA生成前沿 |
| VQE 有限差分梯度 O(n_params×2ⁿ) | ✅ Parameter-shift 规则 |
| 定理证明器经典枚举 | ✅ Grover 量子搜索证明空间 |
| QFT apply/matrix 不一致 | ✅ 受控相位门逻辑修正 |
| DensityMatrix amplitude damping 仅作用于 qubit 0 | ✅ 所有qubit循环施加 |
| QAOA Ising/QUBO 编码不匹配 | ✅ 统一变量转换 |
| QuantumLayer CNOT 优先级bug | ✅ 运算符逻辑修正 |
| 解析器不支持下划线开头变量 | ✅ `_` 标识符支持 |
| kron 遮蔽 numpy.kron | ✅ 重命名为 kronecker_prod |
| VQE 非H₂分子静默替换 | ✅ 明确报错提示 |
| IBM JobStatus 路径问题 | ✅ try/except 兼容Qiskit 1.0+ |

---

## 👤 作者

宋梓铭 · [Gitee](https://gitee.com/song-jack/qsl) · 15011462616@163.com

---

## 📄 许可证

MIT License — 可自由使用、修改、分发。
