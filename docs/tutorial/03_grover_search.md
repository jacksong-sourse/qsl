[← 返回文档中心](../index.md)

# QSL 教程 03：Grover 量子搜索算法

Grover 算法是量子计算中最著名的算法之一，它在**无序数据库搜索**问题上实现了相对于经典算法的**平方加速**。本教程将详细讲解 Grover 算法的原理，并通过 QSL 框架演示如何使用它解决实际问题。

---

## 📚 目录

1. [Grover 算法原理简介](#1-grover-算法原理简介)
2. [基础：无序数据库搜索](#2-基础无序数据库搜索)
3. [使用谓词函数搜索](#3-使用谓词函数搜索)
4. [SAT 问题求解](#4-sat-问题求解)
5. [3-SAT 问题示例](#5-3-sat-问题示例)
6. [图着色问题](#6-图着色问题)
7. [BBHT 算法：未知解数的搜索](#7-bbht-算法未知解数的搜索)
8. [结果验证](#8-结果验证)
9. [完整示例汇总](#9-完整示例汇总)

---

## 1. Grover 算法原理简介

### 1.1 问题背景

经典计算机在无序数据库中搜索一个特定项，平均需要检查 **N/2** 个条目，最坏情况下需要检查全部 **N** 个条目，时间复杂度为 **O(N)**。

Grover 算法利用量子力学的**叠加性**和**干涉性**，仅需 **O(√N)** 次 Oracle 查询即可找到目标项，实现了平方加速。对于 N=100 万的数据库，经典算法平均需要 50 万次查询，而 Grover 算法仅需约 1000 次！

### 1.2 核心思想：振幅放大

Grover 算法的核心是**振幅放大（Amplitude Amplification）**技术，通过两个关键操作的迭代来逐步放大目标态的概率幅：

| 操作 | 作用 |
|------|------|
| **Oracle（预言机）** | 对目标解的状态施加 -1 相位翻转，标记出我们要找的项 |
| **Diffusion（扩散算子）** | 关于平均振幅的反射，将标记态的振幅放大 |

### 1.3 算法流程

```
1. 初始化：对所有量子比特作用 Hadamard 门，创建均匀叠加态
   |ψ₀⟩ = H^⊗n |0⟩^⊗n = (1/√N) Σ|x⟩

2. 重复 t_opt 次 Grover 迭代：
   a. Oracle：对解状态施加相位翻转
   b. Diffusion：关于平均振幅的反射

3. 测量：以高概率得到目标解
```

### 1.4 最优迭代次数

设搜索空间大小为 N = 2ⁿ，解的数量为 M：

- sin(θ) = √(M/N)
- 最优迭代次数：t_opt ≈ π/(4θ) - 1/2 ≈ (π/4)√(N/M)
- 理论成功概率：P ≈ sin²((2t_opt+1)θ)，单解时可达 ~96% 以上

---

## 2. 基础：无序数据库搜索

### 2.1 导入模块

首先导入 QSL 的 Grover 搜索模块：

```python
from qsl import GroverSearch, solve_sat
```

### 2.2 搜索单个标记项

最简单的场景：在 N=8（3个量子比特）的搜索空间中查找特定项。

```python
from qsl import GroverSearch

# 创建 3 量子比特的 Grover 搜索器（搜索空间 N = 2^3 = 8）
grover = GroverSearch(n_qubits=3, verbose=True)

# 搜索标记态 |101⟩，即整数 5（二进制 101 = 十进制 5）
marked_states = {5}
result = grover.search_with_oracle_set(marked_states, shots=100)

# 输出结果
print("=" * 50)
print("搜索结果：")
solutions = result.get_solutions()
print(f"找到的解：{solutions}")
print(f"理论成功概率：{result.theory_success_prob:.4%}")
print(f"经验成功率：{result.empirical_success_rate:.4%}")
print(f"Grover 迭代次数：{result.iterations}")
print(f"量子查询次数：{result.quantum_queries}")
```

**代码说明：**
- `n_qubits=3`：使用 3 个量子比特，搜索空间大小为 2³ = 8
- `verbose=True`：输出详细的计算过程
- `search_with_oracle_set()`：直接传入标记态集合，最高效
- `shots=100`：重复测量 100 次以统计成功率
- `result.get_solutions()`：返回去重后的解列表（整数形式）

### 2.3 搜索多个标记项

Grover 算法同样支持搜索多个解：

```python
from qsl import GroverSearch

# 4 量子比特：搜索空间 N = 16
grover = GroverSearch(n_qubits=4, verbose=False)

# 搜索三个标记态：|0011⟩=3, |0111⟩=7, |1011⟩=11
marked_states = {3, 7, 11}
result = grover.search_with_oracle_set(marked_states, shots=100)

solutions = result.get_solutions()
print(f"标记态：{sorted(marked_states)}")
print(f"找到的解：{sorted(solutions)}")
print(f"解的数量：{result.num_solutions}")
print(f"成功率：{result.empirical_success_rate:.2%}")

# 查看测量计数分布
counts = result.get_measurement_counts()
print("\n测量计数分布：")
for state, count in sorted(counts.items()):
    bits = format(state, '04b')
    marker = "✓" if state in marked_states else " "
    print(f"  |{bits}⟩ (int={state:2d}): {count:3d} 次 {marker}")
```

---

## 3. 使用谓词函数搜索

除了直接传入标记集合，你还可以使用 Python 谓词函数定义搜索条件。

### 3.1 基本谓词搜索

```python
from qsl import GroverSearch

# 5 量子比特：搜索空间 N = 32
grover = GroverSearch(n_qubits=5, verbose=False)

# 定义谓词函数：x 是解当且仅当 x 是完全平方数
def is_perfect_square(x):
    root = int(x ** 0.5)
    return root * root == x

# 执行搜索（verbose=True 可查看计算日志）
result = grover.search(is_perfect_square, shots=100)

solutions = result.get_solutions()
print(f"找到的完全平方数：{sorted(solutions)}")
print(f"解的数量：{result.num_solutions}")

# 验证一下
for s in solutions:
    root = int(s ** 0.5)
    print(f"  {s} = {root}²")
```

### 3.2 自定义搜索条件

```python
from qsl import GroverSearch

grover = GroverSearch(n_qubits=6, verbose=False)

# 搜索条件：二进制表示中恰好有 3 个 1 的数
def has_three_ones(x):
    return bin(x).count('1') == 3

result = grover.search(has_three_ones, shots=200)

solutions = sorted(result.get_solutions())
print(f"二进制含 3 个 1 的数（0-63）：")
for s in solutions:
    print(f"  {s:2d} = {format(s, '06b')}")
print(f"共 {len(solutions)} 个解")
```

> **⚠️ 注意**：使用谓词函数 `search()` 时，模拟器会先做一次 O(N) 的经典枚举来统计解数量并构建 Oracle，这是经典模拟的固有开销。真实量子计算机上 Grover 迭代本身仍是 O(√N) 次查询。如果条件可表达为布尔表达式，使用 `solve_sat()` 效率更高（直接编译量子电路，无经典枚举）。

---

## 4. SAT 问题求解

SAT（布尔可满足性问题）是计算机科学中最重要的 NP 完全问题之一。QSL 提供了 `solve_sat()` 函数，使用 Grover 算法直接求解 CNF 形式的 SAT 问题。

### 4.1 CNF 格式说明

CNF（合取范式）格式规则：
- 每个变量用正整数表示：1, 2, 3, ... 对应 x₁, x₂, x₃, ...
- 变量取反用负数表示：-1 表示 ¬x₁（NOT x₁）
- 每个子句是文字的**逻辑或（OR）**，用列表表示
- 整个公式是子句的**逻辑与（AND）**，用列表的列表表示

**示例**：
```python
cnf_clauses = [[1, -2, 3], [-1, 2], [-3]]
```
表示：
```
(x₁ ∨ ¬x₂ ∨ x₃) ∧ (¬x₁ ∨ x₂) ∧ (¬x₃)
```

### 4.2 简单 SAT 示例

```python
from qsl import solve_sat

# CNF 公式: (x₁ ∨ ¬x₂) ∧ (x₂ ∨ x₃) ∧ (¬x₁ ∨ ¬x₃)
cnf_clauses = [[1, -2], [2, 3], [-1, -3]]
n_qubits = 3  # 3 个变量

result = solve_sat(cnf_clauses, n_qubits=n_qubits, shots=100, verbose=True)

solutions = result.get_solutions()
print(f"\nSAT 解（整数形式）：{solutions}")

# 将整数解转换为变量赋值
for sol in solutions:
    assignment = {}
    for i in range(n_qubits):
        var_num = i + 1
        bit = (sol >> i) & 1
        assignment[f"x{var_num}"] = bool(bit)
    bits_str = format(sol, f'0{n_qubits}b')
    print(f"  |{bits_str}⟩ (int={sol}): {assignment}")
```

**解的二进制解释**：
- 整数解的第 0 位（LSB）对应 x₁
- 整数解的第 1 位对应 x₂
- 以此类推，第 i 位对应 x_{i+1}
- 例如解 `4` = `100`₂ 表示 x₁=0, x₂=0, x₃=1

---

## 5. 3-SAT 问题示例

3-SAT 是 SAT 问题的特例，每个子句恰好包含 3 个文字，是第一个被证明为 NP 完全的问题。

### 5.1 经典 3-SAT 实例

```python
from qsl import solve_sat

# 一个 3-SAT 实例：
# (x₁∨x₂∨x₃) ∧ (¬x₁∨¬x₂∨x₃) ∧ (x₁∨¬x₂∨¬x₃) ∧ (¬x₁∨x₂∨¬x₃)
cnf_clauses = [
    [1, 2, 3],
    [-1, -2, 3],
    [1, -2, -3],
    [-1, 2, -3],
]
n_qubits = 3

result = solve_sat(cnf_clauses, n_qubits=n_qubits, shots=100)
solutions = sorted(result.get_solutions())

print("3-SAT 问题求解：")
print(f"公式：(x₁∨x₂∨x₃) ∧ (¬x₁∨¬x₂∨x₃) ∧ (x₁∨¬x₂∨¬x₃) ∧ (¬x₁∨x₂∨¬x₃)")
print(f"找到 {len(solutions)} 个解：\n")

for sol in solutions:
    bits = format(sol, f'0{n_qubits}b')
    # bits 高位对应 x3，低位对应 x1
    x1 = (sol >> 0) & 1
    x2 = (sol >> 1) & 1
    x3 = (sol >> 2) & 1
    sat = all(
        any(
            (lit > 0 and ((sol >> (abs(lit)-1)) & 1)) or
            (lit < 0 and not ((sol >> (abs(lit)-1)) & 1))
            for lit in clause
        )
        for clause in cnf_clauses
    )
    status = "✓" if sat else "✗"
    print(f"  |{bits}⟩ (int={sol}): x₁={x1}, x₂={x2}, x₃={x3} {status}")
```

### 5.2 多解 3-SAT 问题

下面是一个 4 变量 3-SAT 问题，它有多个解：

```python
from qsl import solve_sat
from qsl.ai.verifier import verify_sat

# 4 变量 3-SAT
# (x₁∨x₂∨x₃) ∧ (¬x₁∨¬x₂∨x₄) ∧ (x₁∨¬x₃∨¬x₄) ∧ (¬x₂∨x₃∨x₄)
cnf_clauses = [
    [1, 2, 3],
    [-1, -2, 4],
    [1, -3, -4],
    [-2, 3, 4],
]
n_qubits = 4

print("4 变量 3-SAT 问题：")
for i, c in enumerate(cnf_clauses):
    lits = []
    for lit in c:
        if lit > 0:
            lits.append(f"x{lit}")
        else:
            lits.append(f"¬x{-lit}")
    print(f"  子句 {i+1}: {' ∨ '.join(lits)}")

result = solve_sat(cnf_clauses, n_qubits=n_qubits, shots=200)
solutions = sorted(result.get_solutions())

print(f"\n找到 {len(solutions)} 个解：")
for sol in solutions:
    bits = format(sol, f'0{n_qubits}b')
    x = [(sol >> i) & 1 for i in range(n_qubits)]
    v = verify_sat(cnf_clauses, sol)
    status = "✓" if v.passed else "✗"
    print(f"  |{bits}⟩ (int={sol}): x₁={x[0]}, x₂={x[1]}, x₃={x[2]}, x₄={x[3]} {status}")
```

> **💡 提示**：受限于经典模拟器的量子比特上限（26 比特含 ancilla），本教程使用较小规模的 SAT 问题演示。在真实量子硬件上，Grover 算法可处理更大规模的搜索问题，且量子查询复杂度保持 O(√N) 的平方加速。

---

## 6. 图着色问题

图着色问题是经典的组合优化问题：给定一个图，用 k 种颜色给顶点着色，使得相邻顶点颜色不同。我们可以将其编码为 SAT 问题求解。

### 6.1 问题编码：2 着色（二分图判定）

最简单的图着色是 **2 着色问题**（判定图是否为二分图）：
- 每个顶点用 1 个量子比特表示颜色（0 或 1）
- 对每条边 (u, v)，添加约束：u 和 v 颜色不同
- "颜色不同" 在 CNF 中表示为：(x_u ∨ x_v) ∧ (¬x_u ∨ ¬x_v)

### 6.2 三角形 2 着色（不可满足）

三角形图（3 个顶点两两相连）不是二分图，因此无法 2 着色：

```python
from qsl import solve_sat
from qsl.utils.exceptions import NoSolutionError

print("=== 三角形 2 着色（不可满足）===")
# 顶点: 0, 1, 2（两两相连）
# 边: (0,1), (1,2), (0,2)
edges_triangle = [(0, 1), (1, 2), (0, 2)]

# 构造 2 着色 SAT
# x_{v+1} 表示顶点 v 的颜色（1-based 变量编号）
cnf_2color_tri = []
for u, v in edges_triangle:
    # u != v: (x_u OR x_v) AND (NOT x_u OR NOT x_v)
    cnf_2color_tri.append([u + 1, v + 1])
    cnf_2color_tri.append([-(u + 1), -(v + 1)])

try:
    result = solve_sat(cnf_2color_tri, n_qubits=3, shots=50)
    print(f"找到解：{result.get_solutions()}")
except NoSolutionError:
    print("预期结果：NoSolutionError - 三角形无法 2 着色（不是二分图）")
```

### 6.3 路径图 2 着色（可满足）

3 顶点路径图（0-1-2，顶点1连接0和2）是二分图，可以 2 着色：

```python
from qsl import solve_sat
from qsl.ai.verifier import verify_sat

print("\n=== 3顶点路径图 2 着色（可满足）===")
# 顶点: 0-1-2（路径，非环）
# 边: (0,1), (1,2)
edges_path = [(0, 1), (1, 2)]

cnf_path = []
for u, v in edges_path:
    cnf_path.append([u + 1, v + 1])
    cnf_path.append([-(u + 1), -(v + 1)])

result = solve_sat(cnf_path, n_qubits=3, shots=100)
solutions = sorted(result.get_solutions())

print(f"图：0-1-2 路径图")
print(f"公式：相邻顶点颜色不同")
print(f"找到 {len(solutions)} 个解：\n")

for sol in solutions:
    bits = format(sol, '03b')
    colors = [(sol >> i) & 1 for i in range(3)]
    v = verify_sat(cnf_path, sol)
    status = "✓" if v.passed else "✗"
    print(f"  |{bits}⟩ (int={sol}): 顶点颜色 = {colors} {status}")

print("\n（合法着色：交替着色模式，如 010 或 101）")
```

### 6.4 3 着色问题编码思路

对于 3 着色问题（用 3 种颜色给图着色，相邻顶点颜色不同），每个顶点需要 **2 个量子比特** 来编码 3 种颜色：

**编码方案**：
- 每个顶点用 2 bit 表示颜色：`01`=颜色0, `10`=颜色1, `11`=颜色2
- 需要排除无效编码 `00`：添加子句 `(low_bit ∨ high_bit)` 确保至少一位为 1
- 对每条边 (u, v)，添加约束确保 u 和 v 的两位颜色编码不完全相同

**约束类型**：
1. **有效性约束**：排除 `00` 无效颜色 → 1 个子句/顶点
2. **邻接约束**：排除 (01,01), (10,10), (11,11) 三种同色情况 → 3 个子句/边

> **⚠️ 模拟限制**：多比特比较的 CNF 编码会产生 4 文字子句，编译为量子电路时需要大量 ancilla 比特。当前模拟器上限为 26 量子比特，三角形 3 着色（6 主比特 + 65 ancilla = 71）超出模拟范围。但核心编码思路已通过 2 着色示例充分展示，3 着色在真机或更大规模模拟器上遵循相同原理。

**2 着色与 3 着色的编码对比**：

| 问题 | 比特/顶点 | 子句/边 | 三角形(3顶点)总量子比特 |
|------|----------|---------|------------------------|
| 2 着色 | 1 | 2 | 3 (无 ancilla) |
| 3 着色 | 2 | 3+1(有效性) | 6 + ancilla |
| k 着色 | ⌈log₂k⌉ | k 个同色排除子句 | 更多 |

---

## 7. BBHT 算法：未知解数的搜索

当解的数量 M 未知时，标准 Grover 算法无法计算最优迭代次数。QSL 的 `solve_sat()` 内部使用 **BBHT（Boyer-Brassard-Høyer-Tapp）算法**进行指数搜索，无需预先知道解的数量。

### 7.1 BBHT 算法原理

BBHT 算法采用**指数搜索**策略：
1. 初始化迭代次数上限 m = 1
2. 从 [0, m) 随机选择迭代次数 t
3. 执行 t 次 Grover 迭代后测量
4. 如果找到解则停止；否则 m = λ × m（λ ≈ 1.34）
5. 重复直到 m 超过 √N
6. 若单轮失败则重启（最多 8 轮，失败概率指数衰减）

BBHT 的期望 Oracle 查询复杂度仍为 **O(√(N/M))**，与已知 M 时同阶。

### 7.2 BBHT 自动搜索示例

```python
from qsl import GroverSearch

# 当使用 search_expressions() 且 num_solutions=None 时，自动启用 BBHT
# solve_sat() 默认使用 BBHT（因为 SAT 解数通常未知）

# 让我们用一个解数未知的例子来演示
grover = GroverSearch(n_qubits=5, verbose=True)

# 条件：素数（我们不告诉算法有多少个素数）
def is_prime(x):
    if x < 2:
        return False
    for i in range(2, int(x**0.5) + 1):
        if x % i == 0:
            return False
    return True

# 注意：search() 方法默认会先 O(N) 统计解数（为了计算最优迭代）
# 如果要使用 BBHT 无枚举路径，需要通过 solve_sat 或 search_expressions
# 这里我们演示 search_with_oracle_set 配合 BBHT 行为：

# 直接使用 search 并指定 num_solutions=None 会触发自动统计
result = grover.search(is_prime, num_solutions=None, shots=50)

solutions = sorted(result.get_solutions())
print(f"\n0-31 中的素数（BBHT 找到）：{solutions}")
print(f"解数量 M = {result.num_solutions}")
print(f"Grover 迭代次数 = {result.iterations}")
```

> **💡 技术细节**：`solve_sat()` 函数直接将 CNF 编译为量子 Oracle 电路（X/CNOT/Toffoli/Z 门），**不做任何 2ⁿ 级别的经典枚举**，完全通过 BBHT 量子搜索找到解，这是真正的量子加速路径。

### 7.3 BBHT 在 SAT 中的应用

`solve_sat()` 总是使用 BBHT 算法，因为 SAT 问题的解数通常事先未知：

```python
from qsl import solve_sat

# 构造一个解数未知的 SAT 问题
cnf = [
    [1, 2],
    [-1, 3],
    [2, -3],
    [-2, 4],
    [3, -4],
]
n_qubits = 4

print("解数未知的 SAT 问题（BBHT 自动搜索）：")
print("公式：(x₁∨x₂) ∧ (¬x₁∨x₃) ∧ (x₂∨¬x₃) ∧ (¬x₂∨x₄) ∧ (x₃∨¬x₄)")

result = solve_sat(cnf, n_qubits=n_qubits, shots=100, verbose=True)

solutions = sorted(result.get_solutions())
print(f"\nBBHT 搜索结果：")
print(f"量子 Oracle 查询次数：{result.quantum_queries}")
print(f"找到的解：{solutions}")
for sol in solutions:
    bits = format(sol, f'0{n_qubits}b')
    x = [(sol >> i) & 1 for i in range(n_qubits)]
    print(f"  |{bits}⟩: x₁={x[0]}, x₂={x[1]}, x₃={x[2]}, x₄={x[3]}")
```

---

## 8. 结果验证

QSL 提供了 `verify_grover` 和 `verify_sat` 函数对搜索结果进行独立的经典验证，确保量子计算结果正确。

### 8.1 验证 Grover 搜索结果

```python
from qsl import GroverSearch
from qsl.ai.verifier import verify_grover

# 执行 Grover 搜索
n_qubits = 5
marked = {7, 13, 25}
grover = GroverSearch(n_qubits, verbose=False)
result = grover.search_with_oracle_set(marked, shots=200)

# 获取测量计数
counts = result.get_measurement_counts()
print("测量计数：", counts)

# 经典验证
verification = verify_grover(
    marked_states=list(marked),
    measured=counts,
    n_qubits=n_qubits
)

print("\n验证结果：")
print(f"  通过: {verification.passed}")
print(f"  消息: {verification.message}")
print(f"  详情:")
for k, v in verification.details.items():
    print(f"    {k}: {v}")
```

### 8.2 验证 SAT 结果

```python
from qsl import solve_sat
from qsl.ai.verifier import verify_sat

# 求解 SAT
cnf_clauses = [[1, -2], [2, 3], [-1, -3], [1, 3]]
n_qubits = 3

result = solve_sat(cnf_clauses, n_qubits=n_qubits, shots=50)
solutions = result.get_solutions()

print(f"SAT 解：{solutions}")

# 对每个找到的解进行经典验证
for sol in solutions:
    verification = verify_sat(cnf_clauses, sol)
    bits = format(sol, f'0{n_qubits}b')
    status = "✓ 通过" if verification.passed else "✗ 失败"
    print(f"  解 |{bits}⟩ (int={sol}): {status}")
    print(f"    {verification.message}")
```

### 8.3 验证函数 API

| 函数 | 参数 | 说明 |
|------|------|------|
| `verify_grover(marked_states, measured, n_qubits)` | 标记态列表、测量计数字典{int:count}、比特数 | 验证 top-k 命中标记集且成功率超过经典随机 |
| `verify_sat(clauses, assignment)` | CNF 子句列表、赋值（int/str/dict） | 将赋值代回子句逐一验证是否满足 |

两个函数都返回 `VerificationResult(passed: bool, message: str, details: dict)` 对象。

---

## 9. 完整示例汇总

下面是一个整合了所有功能的完整可运行示例：

```python
"""
QSL Grover 搜索算法完整示例
涵盖：数据库搜索、谓词搜索、SAT求解、3-SAT、图着色、BBHT、结果验证
"""

from qsl import GroverSearch, solve_sat
from qsl.ai.verifier import verify_grover, verify_sat
from qsl.utils.exceptions import NoSolutionError


def example_1_database_search():
    """示例1：基础数据库搜索（单标记项）"""
    print("\n" + "="*60)
    print("示例1：无序数据库搜索 - 查找单个标记项")
    print("="*60)
    
    grover = GroverSearch(n_qubits=3, verbose=False)
    result = grover.search_with_oracle_set({5}, shots=100)
    
    print(f"搜索空间：N = 2^{grover.n} = {grover.N}")
    print(f"标记态：|101⟩ = int(5)")
    print(f"找到的解：{result.get_solutions()}")
    print(f"迭代次数：{result.iterations}")
    print(f"成功率：{result.empirical_success_rate:.2%}")
    
    counts = result.get_measurement_counts()
    v = verify_grover([5], counts, n_qubits=3)
    print(f"经典验证：{'通过' if v.passed else '失败'} - {v.message}")


def example_2_predicate_search():
    """示例2：谓词函数搜索"""
    print("\n" + "="*60)
    print("示例2：谓词函数搜索 - 查找完全平方数")
    print("="*60)
    
    def is_perfect_square(x):
        root = int(x ** 0.5)
        return root * root == x
    
    grover = GroverSearch(n_qubits=5, verbose=False)
    result = grover.search(is_perfect_square, shots=100)
    
    solutions = sorted(result.get_solutions())
    expected = [x for x in range(32) if is_perfect_square(x)]
    
    print(f"搜索空间：0-31")
    print(f"完全平方数：{expected}")
    print(f"Grover 找到：{solutions}")
    print(f"解数量：{result.num_solutions}")
    print(f"成功率：{result.empirical_success_rate:.2%}")


def example_3_sat_basic():
    """示例3：基础 SAT 求解"""
    print("\n" + "="*60)
    print("示例3：SAT 问题求解")
    print("="*60)
    
    cnf = [[1, -2], [2, 3], [-1, -3]]
    n = 3
    
    result = solve_sat(cnf, n_qubits=n, shots=100)
    solutions = sorted(result.get_solutions())
    
    print(f"CNF 公式：(x₁∨¬x₂) ∧ (x₂∨x₃) ∧ (¬x₁∨¬x₃)")
    print(f"找到 {len(solutions)} 个解：")
    for sol in solutions:
        bits = format(sol, f'0{n}b')
        x1 = (sol >> 0) & 1
        x2 = (sol >> 1) & 1
        x3 = (sol >> 2) & 1
        v = verify_sat(cnf, sol)
        status = "✓" if v.passed else "✗"
        print(f"  |{bits}⟩ (int={sol}): x1={x1}, x2={x2}, x3={x3} {status}")


def example_4_3sat():
    """示例4：3-SAT 问题"""
    print("\n" + "="*60)
    print("示例4：3-SAT 问题（4变量）")
    print("="*60)
    
    cnf = [
        [1, 2, 3],
        [-1, -2, 4],
        [1, -3, -4],
        [-2, 3, 4],
    ]
    n = 4
    
    result = solve_sat(cnf, n_qubits=n, shots=200)
    solutions = sorted(result.get_solutions())
    
    print(f"4 变量 3-SAT（4个子句）")
    print(f"找到 {len(solutions)} 个解：")
    for sol in solutions[:6]:
        bits = format(sol, f'0{n}b')
        v = verify_sat(cnf, sol)
        status = "✓" if v.passed else "✗"
        x = [(sol >> i) & 1 for i in range(n)]
        print(f"  |{bits}⟩ (int={sol}): {x} {status}")
    if len(solutions) > 6:
        print(f"  ... 共 {len(solutions)} 个解")


def example_5_graph_coloring():
    """示例5：图着色问题"""
    print("\n" + "="*60)
    print("示例5：图着色问题（2着色）")
    print("="*60)
    
    print("三角形图 2 着色（不可满足）：")
    edges_tri = [(0, 1), (1, 2), (0, 2)]
    cnf_tri = []
    for u, v in edges_tri:
        cnf_tri.append([u + 1, v + 1])
        cnf_tri.append([-(u + 1), -(v + 1)])
    try:
        solve_sat(cnf_tri, n_qubits=3, shots=50)
        print("  意外找到解！")
    except NoSolutionError:
        print("  ✓ 正确返回无解（三角形不是二分图，无法2着色）")
    
    print("\n路径图 2 着色（可满足）：")
    edges_path = [(0, 1), (1, 2)]
    cnf_path = []
    for u, v in edges_path:
        cnf_path.append([u + 1, v + 1])
        cnf_path.append([-(u + 1), -(v + 1)])
    result = solve_sat(cnf_path, n_qubits=3, shots=100)
    solutions = sorted(result.get_solutions())
    print(f"  图：0-1-2 路径图")
    print(f"  找到 {len(solutions)} 个合法着色：")
    for sol in solutions:
        bits = format(sol, '03b')
        colors = [(sol >> i) & 1 for i in range(3)]
        v = verify_sat(cnf_path, sol)
        status = "✓" if v.passed else "✗"
        print(f"    |{bits}⟩: {colors} {status}")


def example_6_bbht():
    """示例6：BBHT 未知解数搜索"""
    print("\n" + "="*60)
    print("示例6：BBHT 算法（未知解数）")
    print("="*60)
    
    cnf = [[1, 2], [-1, 3], [2, -3], [-2, 4], [3, -4]]
    n = 4
    
    result = solve_sat(cnf, n_qubits=n, shots=100, verbose=False)
    
    print(f"CNF：5个子句，解数未知，BBHT自动搜索")
    print(f"量子 Oracle 查询次数：{result.quantum_queries}")
    solutions = sorted(result.get_solutions())
    print(f"找到的解：{solutions}")
    for sol in solutions:
        v = verify_sat(cnf, sol)
        print(f"  int={sol}: {v.message}")
    print("（BBHT 无需预先知道解数，自动调整迭代次数）")


if __name__ == "__main__":
    print("QSL v0.6.3 - Grover 搜索算法完整教程示例")
    print("=" * 60)
    
    example_1_database_search()
    example_2_predicate_search()
    example_3_sat_basic()
    example_4_3sat()
    example_5_graph_coloring()
    example_6_bbht()
    
    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)
```

---

## 📝 本章小结

| 功能 | API | 说明 |
|------|-----|------|
| 创建搜索器 | `GroverSearch(n_qubits)` | 初始化 n 量子比特 Grover 搜索 |
| 标记集搜索 | `.search_with_oracle_set(marked, shots)` | 已知标记态时最高效 |
| 谓词搜索 | `.search(predicate, shots)` | 用 Python 函数定义搜索条件 |
| SAT 求解 | `solve_sat(cnf, n_qubits, shots)` | 求解 CNF 格式 SAT 问题（BBHT） |
| 获取解 | `result.get_solutions()` | 返回去重后的解整数列表 |
| 测量计数 | `result.get_measurement_counts()` | 返回{state: count}字典 |
| 成功率 | `result.empirical_success_rate` | 经验成功概率 |
| 验证 Grover | `verify_grover(marked, counts, n)` | 经典验证搜索结果 |
| 验证 SAT | `verify_sat(clauses, assignment)` | 代回子句验证赋值 |

### 关键要点

1. **平方加速**：Grover 在无序搜索上提供 O(√N) vs O(N) 的平方加速
2. **振幅放大**：通过 Oracle + Diffusion 迭代放大目标态概率幅
3. **BBHT 算法**：解数未知时通过指数搜索自动调整迭代次数
4. **SAT 求解**：`solve_sat()` 直接编译量子电路，无经典枚举开销
5. **结果验证**：始终使用 `verify_*` 函数对量子结果进行经典校验

### 下一步

- 学习 [04 - Shor 大数分解](04_shor_factorization.md)：了解量子算法如何破解 RSA 密码
- 探索 [05 - QAOA 组合优化](05_qaoa_optimization.md)：用量子近似优化求解最大割等问题
- 查看 [08 - 可视化指南](08_visualization.md)：绘制量子电路和测量直方图
