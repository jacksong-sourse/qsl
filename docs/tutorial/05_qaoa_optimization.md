[← 返回文档中心](../index.md)

# 05 - QAOA 组合优化算法

> **难度**：⭐⭐⭐ | **依赖**：`pip install qsl-quantum[algorithms]`（需要 scipy）

QAOA（Quantum Approximate Optimization Algorithm，量子近似优化算法）由 Edward Farhi 等人于 2014 年提出，是一种用于求解组合优化问题的变分量子算法。它通过参数化量子电路交替演化**问题哈密顿量** $H_C$ 和**混合哈密顿量** $H_M$，在量子硬件上寻找近似最优解。

---

## 📋 目录

1. [算法原理](#算法原理)
2. [安装依赖](#安装依赖)
3. [MaxCut 问题（4 节点环图）](#maxcut-问题4-节点环图)
4. [QAOA 参数详解](#qaoa-参数详解)
5. [Ising 与 QUBO 编码](#ising-与-qubo-编码)
6. [TSP 旅行商问题示例](#tsp-旅行商问题示例)
7. [采样与结果分析](#采样与结果分析)
8. [完整可运行脚本](#完整可运行脚本)

---

## 算法原理

QAOA 使用 $p$ 层交替演化的量子电路：

$$|\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})\rangle = \left( \prod_{k=1}^{p} e^{-i\beta_k H_M} e^{-i\gamma_k H_C} \right) |+\rangle^{\otimes n}$$

其中：

- $|+\rangle^{\otimes n} = H^{\otimes n}|0\rangle^{\otimes n}$：均匀叠加态作为初始态；
- $H_C$：**代价哈密顿量**（Cost Hamiltonian），编码优化问题的目标函数；
- $H_M = \sum_i X_i$：**混合哈密顿量**（Mixer Hamiltonian），由泡利 X 门组成，用于探索解空间；
- $\gamma_k, \beta_k$：可训练参数，通过经典优化器（如 COBYLA）调整以最小化 $\langle H_C \rangle$。

随着层数 $p \to \infty$，QAOA 能收敛到精确最优解；在浅层电路下也能给出高质量的近似解。

---

## 安装依赖

```bash
pip install "qsl-quantum[algorithms]"
# 或单独安装 scipy（用于经典优化器）
pip install scipy numpy
```

---

## MaxCut 问题（4 节点环图）

MaxCut 是最经典的 NP-hard 组合优化问题之一：将图的顶点划分为两个集合，使得**跨集合的边数（割值）最大**。

### 问题定义

4 节点环图：顶点 0-1-2-3-0 连成正方形，共 4 条边。

邻接矩阵：
```
    0  1  2  3
0 [ 0, 1, 0, 1 ]
1 [ 1, 0, 1, 0 ]
2 [ 0, 1, 0, 1 ]
3 [ 1, 0, 1, 0 ]
```

最优解：交替划分（如 {0,2} / {1,3}），割值 = 4。

### 代码示例

```python
import numpy as np
from qsl.algorithms import QAOA

# 1. 定义 4 节点环图的邻接矩阵
adj = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
], dtype=float)

# 2. 将邻接矩阵转换为 QAOA 代价矩阵
cost_matrix = QAOA.maxcut_cost_matrix(adj)

# 3. 创建 QAOA 实例（4 量子比特，p=1 层，默认 Ising 编码）
qaoa = QAOA(n_qubits=4, cost_matrix=cost_matrix, p=1, encoding="ising")

# 4. 运行优化
np.random.seed(42)
params, energy = qaoa.optimize(maxiter=200)

# 5. 查看结果
print(f"最优能量: {energy:.6f}")
print(f"最优比特串（二进制）: {qaoa.optimal_bitstring_str}")
print(f"最优比特串（整数）: {qaoa.optimal_bitstring}")
```

**预期输出**（近似值，因随机初始化略有波动）：
```
最优能量: -3.999...
最优比特串（二进制）: 0101  # 或 1010，表示交替划分
```

> **注意**：MaxCut 在 Ising 编码下，QAOA 最小化的目标是 $\sum_{(i,j)\in E} s_i s_j$，与 MaxCut 目标 $\sum (1-s_is_j)/2$ 等价。最优能量为 $-4$ 对应割值 4。

### 增加层数 p 提升精度

```python
import numpy as np
from qsl.algorithms import QAOA

adj = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
], dtype=float)

# 使用更深的电路 p=3
cost_matrix = QAOA.maxcut_cost_matrix(adj)
qaoa = QAOA(n_qubits=4, cost_matrix=cost_matrix, p=3, encoding="ising")

np.random.seed(123)
params, energy = qaoa.optimize(maxiter=500)

print(f"p=3 最优能量: {energy:.8f}")
print(f"p=3 最优比特串: {qaoa.optimal_bitstring_str}")

# 验证该比特串对应的割值
def compute_cut(bitstring: int, adj: np.ndarray) -> int:
    n = adj.shape[0]
    cut = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                si = (bitstring >> i) & 1
                sj = (bitstring >> j) & 1
                if si != sj:
                    cut += 1
    return cut

best_bit = qaoa.optimal_bitstring
cut_value = compute_cut(best_bit, adj)
print(f"割值: {cut_value} (理论最优: 4)")
```

---

## QAOA 参数详解

### 构造函数

```python
QAOA(
    n_qubits: int,
    cost_matrix: np.ndarray,
    p: int = 1,
    encoding: str = "ising"
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_qubits` | int | 必填 | 量子比特数（变量数） |
| `cost_matrix` | ndarray | 必填 | $n \times n$ 对称矩阵，定义问题代价 |
| `p` | int | `1` | QAOA 层数（深度），越大精度越高但计算越慢 |
| `encoding` | str | `"ising"` | 编码方式：`"ising"`（自旋 $s_i\in\{-1,+1\}$）或 `"qubo"`（二进制 $x_i\in\{0,1\}$） |

### 主要方法

```python
# 运行优化
params, energy = qaoa.optimize(maxiter=200, verbose=False)
#   maxiter: 最大优化迭代次数
#   verbose: 是否打印优化过程
#   返回: (最优参数数组, 最优能量值)

# 获取最优解
bitstring, cost = qaoa.get_optimal_bitstring()
#   返回: (比特串整数, 对应代价值)

# 采样多个候选解
solutions = qaoa.sample_solutions(n_samples=10)
#   返回: [(bitstring, cost), ...] 按代价升序排列
```

### 关键属性

| 属性 | 说明 |
|------|------|
| `qaoa.optimal_energy` | 找到的最优能量值（优化前为 `None`） |
| `qaoa.optimal_bitstring` | 最优解的整数表示（小端序，低位对应 q0） |
| `qaoa.optimal_bitstring_str` | 最优解的二进制字符串（高位在左，如 `"0101"`） |

### 静态方法

```python
cost_matrix = QAOA.maxcut_cost_matrix(adjacency_matrix)
```
将图的邻接矩阵直接转换为 MaxCut 问题的 `cost_matrix`。

---

## Ising 与 QUBO 编码

QAOA 支持两种变量编码方式，内部统一转换为 Ising 形式进行模拟：

### Ising 编码（默认）

- 变量：$s_i \in \{-1, +1\}$（自旋向上/向下）
- 代价函数：$C = \sum_i h_i s_i + \sum_{i<j} J_{ij} s_i s_j$
- `cost_matrix` 的对角线为 $h_i$，非对角线为 $J_{ij}$

### QUBO 编码

- 变量：$x_i \in \{0, 1\}$（二进制）
- 代价函数：$C = \sum_i Q_{ii} x_i + \sum_{i<j} Q_{ij} x_i x_j$
- `cost_matrix` 即为 QUBO 矩阵 $Q$
- 内部自动通过 $x_i = (1 - s_i)/2$ 转换为 Ising 形式

### 编码转换示例

```python
import numpy as np
from qsl.algorithms import QAOA

# 同样的 MaxCut 问题，用 QUBO 编码描述
# QUBO 形式: C = -sum_{(i,j) in E} (x_i + x_j - 2x_i x_j)
n = 4
adj = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
], dtype=float)

# 构造 QUBO 矩阵 Q
Q_qubo = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if adj[i, j] > 0:
            Q_qubo[i, i] += -1
            Q_qubo[j, j] += -1
            Q_qubo[i, j] += 2

# 使用 QUBO 编码创建 QAOA
qaoa_qubo = QAOA(n_qubits=n, cost_matrix=Q_qubo, p=2, encoding="qubo")
np.random.seed(42)
params, energy = qaoa_qubo.optimize(maxiter=300)

print(f"QUBO 编码最优能量: {energy:.6f}")
print(f"QUBO 编码最优比特串: {qaoa_qubo.optimal_bitstring_str}")
# 注意：QUBO 编码的能量尺度/偏移不同，但最优比特串应与 Ising 一致
```

---

## TSP 旅行商问题示例

TSP（Traveling Salesman Problem，旅行商问题）：给定 $n$ 个城市，求经过每个城市恰好一次并回到起点的最短回路。我们以 3 个城市的小规模 TSP 为例演示。

### 编码方案

使用 $n^2$ 个量子比特：$x_{t,i} = 1$ 表示在时间步 $t$ 访问城市 $i$。

```python
import numpy as np
from qsl.algorithms import QAOA

# 3 个城市的距离矩阵
# 城市: 0, 1, 2
dist = np.array([
    [0, 1, 2],  # 0->1: 1, 0->2: 2
    [1, 0, 3],  # 1->0: 1, 1->2: 3
    [2, 3, 0]   # 2->0: 2, 2->1: 3
])
n_cities = 3
n_qubits = n_cities * n_cities  # 9 qubits

# 构造 QUBO 矩阵（简化版，惩罚约束违反）
def build_tsp_qubo(dist, penalty=10.0):
    n = dist.shape[0]
    nq = n * n
    Q = np.zeros((nq, nq))
    
    def idx(t, i):
        return t * n + i
    
    # 1. 目标函数: 最小化路径长度
    for t in range(n):
        t_next = (t + 1) % n
        for i in range(n):
            for j in range(n):
                if i != j:
                    w = dist[i, j]
                    Q[idx(t, i), idx(t_next, j)] += w
    
    # 2. 惩罚项: 每个时间步恰好访问一个城市
    for t in range(n):
        for i in range(n):
            Q[idx(t, i), idx(t, i)] -= penalty
            for j in range(i + 1, n):
                Q[idx(t, i), idx(t, j)] += 2 * penalty
    
    # 3. 惩罚项: 每个城市恰好访问一次
    for i in range(n):
        for t1 in range(n):
            Q[idx(t1, i), idx(t1, i)] -= penalty
            for t2 in range(t1 + 1, n):
                Q[idx(t1, i), idx(t2, i)] += 2 * penalty
    
    return Q

Q_tsp = build_tsp_qubo(dist, penalty=10.0)

# 创建 QAOA（9 量子比特，QUBO 编码，p=1 层以保证速度）
qaoa_tsp = QAOA(n_qubits=n_qubits, cost_matrix=Q_tsp, p=1, encoding="qubo")

np.random.seed(42)
params, energy = qaoa_tsp.optimize(maxiter=200, verbose=True)

print(f"\nTSP 最优能量: {energy:.6f}")
print(f"TSP 最优比特串: {qaoa_tsp.optimal_bitstring_str}")

# 解码结果
bit_str = qaoa_tsp.optimal_bitstring_str
route = []
for t in range(n_cities):
    for i in range(n_cities):
        pos = t * n_cities + i
        # 注意字符串是高位在左
        if bit_str[n_qubits - 1 - pos] == '1':
            route.append(i)
            break

if len(route) == n_cities:
    print(f"路径: {' → '.join(map(str, route))} → {route[0]}")
    total_dist = sum(dist[route[t]][route[(t+1)%n_cities]] for t in range(n_cities))
    print(f"总距离: {total_dist}")
```

---

## 采样与结果分析

QAOA 输出的是量子态概率分布，高概率的比特串都是优质解。通过采样可以获取多个候选方案：

```python
import numpy as np
from qsl.algorithms import QAOA

# 5 节点随机图的 MaxCut
np.random.seed(0)
n = 5
adj = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        if np.random.rand() > 0.4:
            adj[i, j] = adj[j, i] = 1

cost_matrix = QAOA.maxcut_cost_matrix(adj)
qaoa = QAOA(n_qubits=n, cost_matrix=cost_matrix, p=2, encoding="ising")

np.random.seed(42)
params, energy = qaoa.optimize(maxiter=300)

print(f"最优能量: {energy:.6f}")
print(f"最优比特串: {qaoa.optimal_bitstring_str}")

# 采样前 10 个高概率解
print("\n高概率候选解:")
solutions = qaoa.sample_solutions(n_samples=10)
for rank, (bits, cost) in enumerate(solutions, 1):
    bits_str = format(bits, f"0{n}b")
    print(f"  #{rank}: {bits_str}  能量={cost:.4f}")
```

---

## 完整可运行脚本

```python
"""
QAOA 组合优化算法完整示例
包含 MaxCut（4节点环图）和问题求解演示
"""
import numpy as np
from qsl.algorithms import QAOA


def maxcut_demo():
    """4 节点环图 MaxCut 问题演示"""
    print("=" * 60)
    print("示例 1: MaxCut - 4 节点环图")
    print("=" * 60)
    
    adj = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ], dtype=float)
    
    cost_matrix = QAOA.maxcut_cost_matrix(adj)
    
    for p in [1, 2, 3]:
        qaoa = QAOA(n_qubits=4, cost_matrix=cost_matrix, p=p, encoding="ising")
        np.random.seed(42)
        params, energy = qaoa.optimize(maxiter=300)
        bits_str = qaoa.optimal_bitstring_str
        
        def cut_value(bits, adj):
            n = adj.shape[0]
            cut = 0
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j] > 0:
                        si = (bits >> i) & 1
                        sj = (bits >> j) & 1
                        if si != sj:
                            cut += 1
            return cut
        
        cv = cut_value(qaoa.optimal_bitstring, adj)
        print(f"  p={p}: 能量={energy:.6f}, 比特串={bits_str}, 割值={cv}/4")


def qubo_vs_ising_demo():
    """Ising 与 QUBO 编码对比"""
    print("\n" + "=" * 60)
    print("示例 2: Ising vs QUBO 编码对比")
    print("=" * 60)
    
    n = 4
    adj = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ], dtype=float)
    
    # Ising 编码
    cost_ising = QAOA.maxcut_cost_matrix(adj)
    qaoa_ising = QAOA(n_qubits=n, cost_matrix=cost_ising, p=2, encoding="ising")
    np.random.seed(123)
    qaoa_ising.optimize(maxiter=200)
    
    print(f"  Ising 最优比特串: {qaoa_ising.optimal_bitstring_str}")
    print(f"  Ising 最优能量:   {qaoa_ising.optimal_energy:.6f}")
    
    # QUBO 编码（等价问题）
    Q_qubo = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                Q_qubo[i, i] += -1
                Q_qubo[j, j] += -1
                Q_qubo[i, j] += 2
    
    qaoa_qubo = QAOA(n_qubits=n, cost_matrix=Q_qubo, p=2, encoding="qubo")
    np.random.seed(123)
    qaoa_qubo.optimize(maxiter=200)
    
    print(f"  QUBO 最优比特串: {qaoa_qubo.optimal_bitstring_str}")
    print(f"  QUBO 最优能量:   {qaoa_qubo.optimal_energy:.6f}")
    print("  （能量值因编码偏移不同，比特串应一致）")


def sampling_demo():
    """采样多个候选解"""
    print("\n" + "=" * 60)
    print("示例 3: 采样高概率候选解")
    print("=" * 60)
    
    np.random.seed(7)
    n = 5
    adj = np.zeros((n, n))
    edges = [(0,1), (0,3), (1,2), (2,3), (2,4), (3,4)]
    for i, j in edges:
        adj[i, j] = adj[j, i] = 1
    
    cost_matrix = QAOA.maxcut_cost_matrix(adj)
    qaoa = QAOA(n_qubits=n, cost_matrix=cost_matrix, p=2, encoding="ising")
    np.random.seed(42)
    qaoa.optimize(maxiter=400)
    
    print(f"  最优解: {qaoa.optimal_bitstring_str}")
    print(f"  Top-8 候选解:")
    solutions = qaoa.sample_solutions(n_samples=8)
    for rank, (bits, cost) in enumerate(solutions, 1):
        bits_str = format(bits, f"0{n}b")
        print(f"    {rank}. {bits_str}  能量={cost:.4f}")


if __name__ == "__main__":
    maxcut_demo()
    qubo_vs_ising_demo()
    sampling_demo()
```

---

## 🔗 相关阅读

- [04 - Shor 大数分解](04_shor_factorization.md)
- [06 - VQE 量子化学](06_vqe_chemistry.md)
- [算法 API 参考](../api/algorithms.md)
