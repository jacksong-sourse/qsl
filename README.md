<h1 align="center">QSL &mdash; 量子搜索语言 v0.4.1</h1>

<p align="center">
  <b>用一句话描述你想找什么，剩下的交给量子计算。</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/tests-365%20passed-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/version-0.4.1-blue" alt="Version">
  <img src="https://img.shields.io/badge/pypi-qsl--quantum-orange?logo=pypi" alt="PyPI">
</p>

---

## v0.4.1 — 五级量子计算框架

一个完整的量子计算 Python 框架，从声明式搜索到自进化 AI 系统：

| 级别 | 模块 | 功能 |
|------|------|------|
| **一级** | Gates + Algorithms | 量子门 / QFT / Shor 因子分解 / QAOA / VQE |
| **二级** | QML | 量子神经网络 / 量子 SVM / 量子核函数 / QGAN |
| **三级** | Backends + Compiler | IBM / AWS Braket / 自动后端选择 / 门融合 / 布局映射 / 错误缓解 |
| **四级** | AI Scientist | LLM 问题翻译 / 量子自主智能体 / 假设测试 / 发现流水线 |
| **五级** | Meta + Network | 遗传算法搜索电路 / RL 编译优化 / 量子区块链 / 分布式节点 |

---

## 5 秒看懂

```python
from qsl import QSLProgram, compile_and_run

program = QSLProgram(
    name="3-SAT",
    n_qubits=3,
    premises=["x0 | ~x1", "x1 | x2", "~x0 | ~x2"],
    shots=10
)
result = compile_and_run(program)
print(result.get_solutions())  # [3, 4] = 011 和 100
```

不用懂量子力学。

---

## 快速体验各模块

### 量子算法

```python
from qsl import ShorSolver, QAOA, VQE, QuantumFourierTransform

# 量子傅里叶变换
qft = QuantumFourierTransform(3)
circuit = qft.build_circuit()

# Shor 算法（经典模拟器）分解 15 = 3 × 5
solver = ShorSolver(15)
factors = solver.factor()
print(factors)  # [3, 5]

# VQE 计算 H2 基态能量
vqe = VQE(2, VQE.h2_hamiltonian())
energy, state = vqe.optimize()

# QAOA 求解 MaxCut 优化问题
import numpy as np
adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # 3节点路径图
qaoa = QAOA(3, QAOA.maxcut_cost_matrix(adj), p=2)
params, energy = qaoa.optimize()
```

### 量子机器学习

```python
from qsl import QuantumLayer, QNN, QuantumSVM

# 量子神经网络
layer = QuantumLayer(n_qubits=3, n_features=4, encoding="angle")
qnn = QNN(n_qubits=3, n_features=4, n_outputs=2)
qnn.fit(X_train, y_train, epochs=50)
preds = qnn.predict(X_test)

# 量子 SVM（兼容 sklearn API）
qsvm = QuantumSVM().fit(X_train, y_train)
print(qsvm.score(X_test, y_test))
```

### 硬件后端

```python
from qsl import AutoBackend

# 自动选择最优后端
backend = AutoBackend(max_qubits=50)
best, backend_type = backend.select()
print(f"Selected: {best} ({backend_type})")
```

### AI 量子科学家

```python
from qsl import ProblemTranslator, QuantumAgent

# 自然语言 → 量子程序
translator = ProblemTranslator()
program = translator.translate("破解 RSA-15 加密")

# 自主量子智能体
agent = QuantumAgent("寻找最优投资组合")
report = agent.run()
```

---

## 安装

```bash
# 核心（仅依赖 numpy）
pip install qsl-quantum

# 带量子算法支持
pip install qsl-quantum[algorithms]

# 量子机器学习
pip install qsl-quantum[qml]

# 真实硬件 (IBM / AWS)
pip install qsl-quantum[ibm]     # IBM Quantum
pip install qsl-quantum[aws]     # AWS Braket

# 全部安装
pip install qsl-quantum[full]
```

---

## 项目结构

```
qsl/
├── core/              量子态、布尔解析、Grover 搜索
├── compiler/          编译器、DSL 解析、门优化、错误缓解
├── backends/          模拟器 + IBM + AWS Braket + 自动选择
├── algorithms/        QFT / Shor / QAOA / VQE
├── qml/               量子层 / QNN / 量子核 / QSVM / QGAN
├── ai/                ⚠ 演示: LLM 翻译器 / 量子智能体 / 假设测试
├── pipelines/         ⚠ 演示: 药物发现 / 密码分析 / 投资组合
├── meta/              ⚠ 演示: 遗传电路搜索 / RL 编译 / 定理证明
├── network/           ⚠ 演示: 分布式节点 / 量子区块链
└── utils/             异常体系、输入验证
tests/                 365 个测试用例
```

---

## 运行测试

```bash
pip install -e ".[dev]"
pytest tests/ -v        # 365 passed
```

---

## 作者

宋梓铭 &middot; [Gitee](https://gitee.com/song-jack/qsl) &middot; 15011462616@163.com

---

## 许可证

MIT &mdash; 随意使用、修改、分发。
