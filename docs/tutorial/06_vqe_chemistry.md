[← 返回文档中心](../index.md)

# 06 - VQE 变分量子本征求解器

> **难度**：⭐⭐⭐ | **依赖**：`pip install qsl-quantum[algorithms]`（需要 scipy）

VQE（Variational Quantum Eigensolver，变分量子本征求解器）由 Peruzzo 等人于 2014 年提出，是量子-经典混合算法的代表。它利用**参数化量子电路（Ansatz）**制备试探波函数，在量子硬件上计算能量期望值，再通过经典优化器调整参数，最终逼近分子哈密顿量的**基态能量**。

VQE 是近期量子硬件（NISQ 时代）最有希望实现实用化量子优势的算法之一，广泛应用于**量子化学**、**材料科学**等领域。

---

## 📋 目录

1. [算法原理](#算法原理)
2. [安装依赖](#安装依赖)
3. [快速开始：H₂ 分子基态能量](#快速开始h₂-分子基态能量)
4. [VQE 参数详解](#vqe-参数详解)
5. [自定义哈密顿量](#自定义哈密顿量)
6. [Ansatz 类型：HE vs UCCSD](#ansatz-类型he-vs-uccsd)
7. [结果分析与观测量](#结果分析与观测量)
8. [完整可运行脚本](#完整可运行脚本)

---

## 算法原理

### 变分原理

对于任意量子态 $|\psi(\boldsymbol{\theta})\rangle$，其能量期望值满足：

$$\langle \psi(\boldsymbol{\theta}) | H | \psi(\boldsymbol{\theta}) \rangle \ge E_0$$

其中 $E_0$ 是哈密顿量 $H$ 的基态能量（最小本征值）。VQE 通过调整参数 $\boldsymbol{\theta}$ 不断逼近这个下界。

### VQE 工作流程

1. **哈密顿量映射**：将分子的电子哈密顿量通过 Jordan-Wigner 或 Bravyi-Kitaev 变换编码为泡利字符串形式：
   $$H = \sum_i c_i P_i, \quad P_i \in \{I, X, Y, Z\}^{\otimes n}$$

2. **Ansatz 制备**：参数化量子电路 $U(\boldsymbol{\theta})$ 将初始态 $|0\rangle^{\otimes n}$ 演化为试探态：
   $$|\psi(\boldsymbol{\theta})\rangle = U(\boldsymbol{\theta}) |0\rangle^{\otimes n}$$

3. **能量测量**：在量子计算机上分别测量每个泡利项 $P_i$ 的期望值 $\langle \psi | P_i | \psi \rangle$，按系数求和：
   $$E(\boldsymbol{\theta}) = \sum_i c_i \langle P_i \rangle$$

4. **经典优化**：使用 COBYLA 等经典优化器更新 $\boldsymbol{\theta}$，最小化 $E(\boldsymbol{\theta})$，直到收敛。

---

## 安装依赖

```bash
pip install "qsl-quantum[algorithms]"
# 核心依赖：numpy + scipy（经典优化器）
pip install numpy scipy
```

---

## 快速开始：H₂ 分子基态能量

氢分子（H₂）是量子化学的"Hello World"。在 STO-3G 基组下，H₂ 哈密顿量可映射为 4 个量子比特的泡利算符之和。QSL 内置了 H₂ 平衡键长（约 0.74 Å）的哈密顿量：

```python
import numpy as np
from qsl.algorithms import VQE

# 获取 H₂ 分子哈密顿量（4 量子比特，STO-3G 基组）
h2_terms = VQE.h2_hamiltonian(bond_length=0.74)

print("H₂ 哈密顿量泡利项:")
for coeff, pauli in h2_terms:
    print(f"  {coeff:+.4f} · {pauli}")

# 创建 VQE 实例
# - 4 量子比特
# - 默认硬件高效 ansatz (he)，2 层
vqe = VQE(
    n_qubits=4,
    hamiltonian_pauli_terms=h2_terms,
    ansatz_type="he",
    n_layers=2
)

# 运行优化
np.random.seed(42)
energy, state = vqe.optimize(maxiter=200, verbose=True)

# 查看结果
print(f"\n{'='*50}")
print(f"H₂ 基态能量 (VQE): {energy:.6f} Hartree")
print(f"精确基态能量约为:  -1.1373 Hartree (FCI/STO-3G)")
print(f"化学精度 (1.6 mHa): {abs(energy - (-1.1373)) < 0.0016}")
```

**预期输出**（近似值）：
```
H₂ 基态能量 (VQE): -1.13... Hartree
```

### 关键属性

```python
# 基态能量
print(f"ground_energy: {vqe.ground_energy:.8f}")

# 基态波函数（态向量）
print(f"ground_state shape: {vqe.ground_state.shape}")  # (16,) 复数向量

# 最优 ansatz 参数
print(f"optimal_params shape: {vqe.optimal_params.shape}")
```

---

## VQE 参数详解

### 构造函数

```python
VQE(
    n_qubits: int,
    hamiltonian_pauli_terms: list[tuple[float, str]],
    ansatz_type: str = "he",
    n_layers: int = 2
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_qubits` | int | 必填 | 量子比特数 |
| `hamiltonian_pauli_terms` | list | 必填 | 哈密顿量的泡利分解：`[(coeff, "pauli_string"), ...]` |
| `ansatz_type` | str | `"he"` | 拟设类型：`"he"`（硬件高效）或 `"uccsd"`（酉耦合簇） |
| `n_layers` | int | `2` | HE ansatz 的层数（仅 `ansatz_type="he"` 时生效） |

### `hamiltonian_pauli_terms` 格式要求

- 每个泡利字符串长度必须**严格等于** `n_qubits`；
- 每个字符只能是 `'I'`, `'X'`, `'Y'`, `'Z'`（大写）；
- 系数 `coeff` 为实数。

**示例**：
```python
# 单量子比特哈密顿量 H = -0.5 Z（横向场伊辛模型单个自旋）
terms = [(-0.5, "Z")]
vqe = VQE(n_qubits=1, hamiltonian_pauli_terms=terms)

# 2 量子比特海森堡模型 H = XX + YY + ZZ
terms = [
    (1.0, "XX"),
    (1.0, "YY"),
    (1.0, "ZZ"),
]
vqe = VQE(n_qubits=2, hamiltonian_pauli_terms=terms)
```

### 主要方法

```python
# 运行 VQE 优化
energy, state = vqe.optimize(maxiter=200, verbose=False)
#   maxiter: 经典优化器最大迭代次数
#   verbose: 是否打印优化进度
#   返回: (基态能量, 基态态向量 numpy 数组)

# 获取各泡利项的期望值
expectations = vqe.get_expectations()
#   返回: {pauli_string: (coeff, expectation_value), ...}
```

### 关键属性

| 属性 | 说明 |
|------|------|
| `vqe.ground_energy` | 计算得到的基态能量（优化前为 `None`） |
| `vqe.ground_state` | 基态波函数（numpy 复数数组，维度 $2^{n\_qubits}$） |
| `vqe.optimal_params` | 最优 ansatz 参数数组 |

---

## 自定义哈密顿量

除了内置的 H₂ 分子，你可以构造任意泡利形式的哈密顿量。以下是几个物理示例：

### 示例 1：单量子比特（自旋在磁场中）

$$H = -Z$$

```python
import numpy as np
from qsl.algorithms import VQE

# H = -Z：本征态 |0> 能量 -1，|1> 能量 +1
# 基态是 |0>，基态能量 -1
terms = [(-1.0, "Z")]
vqe = VQE(n_qubits=1, hamiltonian_pauli_terms=terms, n_layers=1)

np.random.seed(42)
energy, state = vqe.optimize(maxiter=100)

print(f"单量子比特 H=-Z 的基态能量: {energy:.6f}")
print(f"理论精确值: -1.0")
print(f"基态: |0> 振幅={state[0]:.4f}, |1> 振幅={state[1]:.4f}")
```

### 示例 2：两量子比特海森堡反铁磁体

$$H = X_0 X_1 + Y_0 Y_1 + Z_0 Z_1$$

```python
import numpy as np
from qsl.algorithms import VQE

# 海森堡模型：单态 (|01> - |10>)/√2 是基态，能量 -3
terms = [
    (1.0, "XX"),
    (1.0, "YY"),
    (1.0, "ZZ"),
]

vqe = VQE(n_qubits=2, hamiltonian_pauli_terms=terms, n_layers=2)
np.random.seed(42)
energy, state = vqe.optimize(maxiter=300)

print(f"海森堡两体基态能量: {energy:.6f}")
print(f"理论基态能量: -3.0")
print(f"态向量:")
for i, amp in enumerate(state):
    bits = format(i, '02b')
    if abs(amp) > 1e-3:
        print(f"  |{bits}>: {amp:+.4f}")
```

### 示例 3：横向场伊辛模型（TFIM）

$$H = -J \sum_{\langle i,j \rangle} Z_i Z_j - h \sum_i X_i$$

```python
import numpy as np
from qsl.algorithms import VQE

# 3 量子比特 TFIM 开边界：H = -(Z0Z1 + Z1Z2) - 0.5(X0 + X1 + X2)
J = 1.0
h = 0.5
terms = [
    (-J, "ZZI"),
    (-J, "IZZ"),
    (-h, "XII"),
    (-h, "IXI"),
    (-h, "IIX"),
]

vqe = VQE(n_qubits=3, hamiltonian_pauli_terms=terms, ansatz_type="he", n_layers=3)
np.random.seed(42)
energy, state = vqe.optimize(maxiter=400)

print(f"3 自旋 TFIM 基态能量: {energy:.6f}")
```

---

## Ansatz 类型：HE vs UCCSD

VQE 的性能高度依赖于 ansatz（拟设电路）的选择。QSL 提供两种 ansatz：

### 1. 硬件高效拟设（Hardware-Efficient, HE）

- **结构**：每层包含单比特旋转门（RY、RZ）+ 邻近 CNOT 纠缠门 + 环形 CNOT；
- **参数数量**：`n_layers * (2 * n_qubits) + n_qubits`；
- **优点**：电路浅、门数少、适配真实硬件连接；
- **缺点**：无物理先验，可能出现"贫瘠高原"（barren plateau）；
- **适用**：通用问题、快速原型、浅层演示。

```python
vqe_he = VQE(
    n_qubits=4,
    hamiltonian_pauli_terms=h2_terms,
    ansatz_type="he",
    n_layers=2  # 层数可调
)
```

### 2. 酉耦合簇单双激发（UCCSD）

- **结构**：基于 Hartree-Fock 参考态（半填充：$|11\ldots00\rangle$），应用单激发（Single）和双激发（Double）算符；
- **参数数量**：$n_{occ} n_{virt} + \binom{n_{occ}}{2}\binom{n_{virt}}{2}$（由量子比特数自动计算）；
- **优点**：包含化学直觉，参数化更高效，更接近化学精确；
- **缺点**：电路更深，需要粒子数守恒的激发算符；
- **适用**：量子化学分子模拟。

```python
vqe_uccsd = VQE(
    n_qubits=4,
    hamiltonian_pauli_terms=h2_terms,
    ansatz_type="uccsd"
    # 注意：UCCSD 不需要 n_layers 参数
)
```

### HE vs UCCSD 对比（H₂ 分子）

```python
import numpy as np
from qsl.algorithms import VQE

h2_terms = VQE.h2_hamiltonian()

# HE ansatz
vqe_he = VQE(n_qubits=4, hamiltonian_pauli_terms=h2_terms, ansatz_type="he", n_layers=2)
np.random.seed(42)
e_he, _ = vqe_he.optimize(maxiter=300)

# UCCSD ansatz
vqe_uccsd = VQE(n_qubits=4, hamiltonian_pauli_terms=h2_terms, ansatz_type="uccsd")
np.random.seed(42)
e_uccsd, _ = vqe_uccsd.optimize(maxiter=300)

exact = -1.1373
print(f"H₂ 基态能量对比 (键长 0.74 Å):")
print(f"  精确 FCI 值:   {exact:.4f} Hartree")
print(f"  HE  (2层):     {e_he:.4f} Hartree  (误差: {abs(e_he - exact)*1000:.2f} mHa)")
print(f"  UCCSD:         {e_uccsd:.4f} Hartree  (误差: {abs(e_uccsd - exact)*1000:.2f} mHa)")
print(f"  化学精度:       1.6 mHa")
```

---

## 结果分析与观测量

优化完成后，可以通过 `get_expectations()` 查看每个泡利项的贡献，这对于分析电子相关效应很有帮助：

```python
import numpy as np
from qsl.algorithms import VQE

h2_terms = VQE.h2_hamiltonian()
vqe = VQE(n_qubits=4, hamiltonian_pauli_terms=h2_terms, ansatz_type="uccsd")

np.random.seed(42)
energy, state = vqe.optimize(maxiter=300)

print(f"总基态能量: {energy:.6f} Hartree\n")
print(f"各泡利项贡献分析:")
print(f"{'泡利项':<10} {'系数':>8} {'期望值':>10} {'贡献':>10}")
print("-" * 45)

expectations = vqe.get_expectations()
total = 0.0
for pauli, (coeff, exp_val) in sorted(expectations.items()):
    contribution = coeff * exp_val
    total += contribution
    print(f"{pauli:<10} {coeff:>+8.4f} {exp_val:>+10.6f} {contribution:>+10.6f}")

print("-" * 45)
print(f"{'合计':<10} {'':>8} {'':>10} {total:>+10.6f}")
```

### 基态概率分布

```python
import numpy as np
from qsl.algorithms import VQE

h2_terms = VQE.h2_hamiltonian()
vqe = VQE(n_qubits=4, hamiltonian_pauli_terms=h2_terms, n_layers=2)

np.random.seed(42)
vqe.optimize(maxiter=200)

state = vqe.ground_state
probs = np.abs(state) ** 2

print("基态主要计算基态分量（概率 > 1%）:")
for i in range(len(probs)):
    if probs[i] > 0.01:
        bits = format(i, '04b')
        print(f"  |{bits}>: 概率 = {probs[i]*100:.2f}%")
```

---

## H₂ 键长解离曲线扫描

改变 H₂ 的键长，扫描势能面（Potential Energy Surface, PES）：

```python
import numpy as np
from qsl.algorithms import VQE

print("H₂ 势能面扫描 (HE ansatz, n_layers=2)")
print(f"{'键长(Å)':<10} {'基态能量(Hartree)':<20}")
print("-" * 35)

# 不同键长（注意：内置 h2_hamiltonian 当前使用固定系数，
# 实际应用中需根据键长重新计算积分；此处演示框架用法）
bond_lengths = [0.74]

for r in bond_lengths:
    terms = VQE.h2_hamiltonian(bond_length=r)
    vqe = VQE(n_qubits=4, hamiltonian_pauli_terms=terms, n_layers=3)
    np.random.seed(42)
    energy, _ = vqe.optimize(maxiter=400)
    print(f"{r:<10.2f} {energy:<20.6f}")
```

---

## 完整可运行脚本

```python
"""
VQE 变分量子本征求解器完整示例
包含 H2 分子、海森堡模型、TFIM，以及 HE/UCCSD ansatz 对比
"""
import numpy as np
from qsl.algorithms import VQE


def h2_demo():
    """H₂ 分子基态能量计算"""
    print("=" * 60)
    print("示例 1: H₂ 分子基态能量 (STO-3G, 键长 0.74 Å)")
    print("=" * 60)
    
    h2_terms = VQE.h2_hamiltonian()
    print(f"哈密顿量包含 {len(h2_terms)} 个泡利项")
    
    vqe = VQE(n_qubits=4, hamiltonian_pauli_terms=h2_terms,
              ansatz_type="he", n_layers=2)
    
    np.random.seed(42)
    energy, state = vqe.optimize(maxiter=300, verbose=False)
    
    exact_fci = -1.1373
    print(f"  VQE 基态能量:  {energy:.6f} Hartree")
    print(f"  精确 FCI 能量: {exact_fci:.4f} Hartree")
    print(f"  误差:          {abs(energy - exact_fci)*1000:.2f} mHa")
    print(f"  化学精度(1.6mHa): {'✅ 达到' if abs(energy - exact_fci) < 0.0016 else '❌ 未达到'}")
    return energy


def heisenberg_demo():
    """两量子比特海森堡反铁磁模型"""
    print("\n" + "=" * 60)
    print("示例 2: 两量子比特海森堡模型 H = XX + YY + ZZ")
    print("=" * 60)
    
    terms = [
        (1.0, "XX"),
        (1.0, "YY"),
        (1.0, "ZZ"),
    ]
    
    vqe = VQE(n_qubits=2, hamiltonian_pauli_terms=terms, n_layers=2)
    np.random.seed(123)
    energy, state = vqe.optimize(maxiter=300)
    
    print(f"  VQE 基态能量: {energy:.6f}")
    print(f"  理论基态能量: -3.0 (单态)")
    
    print(f"  态向量 (显著分量):")
    for i, amp in enumerate(state):
        if abs(amp) > 1e-3:
            print(f"    |{format(i, '02b')}>: {amp.real:+.4f}{amp.imag:+.4f}j")
    return energy


def ansatz_comparison_demo():
    """HE vs UCCSD ansatz 对比"""
    print("\n" + "=" * 60)
    print("示例 3: HE vs UCCSD Ansatz 对比 (H₂ 分子)")
    print("=" * 60)
    
    h2_terms = VQE.h2_hamiltonian()
    exact = -1.1373
    
    # HE
    vqe_he = VQE(n_qubits=4, hamiltonian_pauli_terms=h2_terms,
                 ansatz_type="he", n_layers=2)
    np.random.seed(42)
    e_he, _ = vqe_he.optimize(maxiter=400)
    
    # UCCSD
    vqe_uccsd = VQE(n_qubits=4, hamiltonian_pauli_terms=h2_terms,
                    ansatz_type="uccsd")
    np.random.seed(42)
    e_uccsd, _ = vqe_uccsd.optimize(maxiter=400)
    
    print(f"  精确 FCI:    {exact:.4f} Hartree")
    print(f"  HE  (2层):   {e_he:.4f} Hartree  误差: {abs(e_he-exact)*1000:.2f} mHa  "
          f"参数数: {len(vqe_he.optimal_params)}")
    print(f"  UCCSD:       {e_uccsd:.4f} Hartree  误差: {abs(e_uccsd-exact)*1000:.2f} mHa  "
          f"参数数: {len(vqe_uccsd.optimal_params)}")


def tfim_demo():
    """横向场伊辛模型"""
    print("\n" + "=" * 60)
    print("示例 4: 3 自旋横向场伊辛模型 (TFIM)")
    print("=" * 60)
    
    J = 1.0
    h = 0.8
    terms = [
        (-J, "ZZI"),
        (-J, "IZZ"),
        (-h, "XII"),
        (-h, "IXI"),
        (-h, "IIX"),
    ]
    
    vqe = VQE(n_qubits=3, hamiltonian_pauli_terms=terms, n_layers=3)
    np.random.seed(42)
    energy, state = vqe.optimize(maxiter=500)
    
    print(f"  H = -J(ZZI + IZZ) - h(XII + IXI + IIX), J={J}, h={h}")
    print(f"  VQE 基态能量: {energy:.6f}")
    
    probs = np.abs(state) ** 2
    print(f"  主要基态分量:")
    for i in range(len(probs)):
        if probs[i] > 0.05:
            print(f"    |{format(i, '03b')}>: {probs[i]*100:.1f}%")


def expectations_demo():
    """各泡利项期望值分析"""
    print("\n" + "=" * 60)
    print("示例 5: H₂ 各泡利项期望值分析")
    print("=" * 60)
    
    h2_terms = VQE.h2_hamiltonian()
    vqe = VQE(n_qubits=4, hamiltonian_pauli_terms=h2_terms, ansatz_type="uccsd")
    np.random.seed(42)
    vqe.optimize(maxiter=300)
    
    expectations = vqe.get_expectations()
    print(f"  总能量: {vqe.ground_energy:.6f} Hartree\n")
    print(f"  {'泡利项':<8} {'系数':>8} {'期望值':>10} {'贡献':>10}")
    print(f"  {'-'*40}")
    total = 0.0
    for pauli in sorted(expectations.keys()):
        coeff, exp_val = expectations[pauli]
        contrib = coeff * exp_val
        total += contrib
        print(f"  {pauli:<8} {coeff:>+8.4f} {exp_val:>+10.6f} {contrib:>+10.6f}")
    print(f"  {'-'*40}")
    print(f"  {'合计':<8} {'':>8} {'':>10} {total:>+10.6f}")


if __name__ == "__main__":
    h2_demo()
    heisenberg_demo()
    ansatz_comparison_demo()
    tfim_demo()
    expectations_demo()
```

---

## ⚠️ 注意事项

1. **scipy 依赖**：VQE 默认使用 COBYLA 优化器，需要安装 scipy。如未安装，将自动回退到基于 parameter-shift 规则的梯度下降（收敛速度较慢）。

2. **随机初始化**：参数初始值随机生成，可能导致结果波动。建议设置随机种子（`np.random.seed(42)`）获得可复现结果，或多次运行取最优。

3. **层数选择**：
   - 层数少（`n_layers=1,2`）：训练快，但表达能力有限，可能欠拟合；
   - 层数多（`n_layers≥4`）：表达能力强，但更难优化，可能遭遇贫瘠高原；
   - 对于 H₂ 这类小分子，`n_layers=2~3` 的 HE ansatz 通常足够。

4. **UCCSD 的参考态**：UCCSD ansatz 默认从 Hartree-Fock 态 $|11\ldots00\rangle$ 出发（半填充），适用于电子数等于量子比特数一半的情况（如 H₂ 的 4 量子比特对应 2 个电子）。

5. **泡利字符串顺序**：泡利字符串遵循 QSL 约定——第 0 位字符对应 q0（最低位量子比特），例如 `"ZIIZ"` 表示 $Z_0 \otimes I_1 \otimes I_2 \otimes Z_3$。

---

## 🔗 相关阅读

- [05 - QAOA 组合优化](05_qaoa_optimization.md)
- [04 - Shor 大数分解](04_shor_factorization.md)
- [算法 API 参考](../api/algorithms.md)
