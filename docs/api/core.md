[← 返回文档中心](../index.md)

# 核心 API 参考

核心模块提供量子态表示、噪声模型、Grover 搜索算法以及布尔表达式解析等基础功能。

---

## QuantumState

**量子态向量类**，用于表示 n 量子比特的纯态。内部存储 2^n 个复振幅，初始态为 |0...0⟩。

### 构造函数

```python
QuantumState(n_qubits: int)
```

**参数：**
- `n_qubits` (int)：量子比特数（1 ≤ n_qubits ≤ 26）

**属性：**
- `n_qubits` (int)：量子比特数
- `size` (int)：希尔伯特空间维度（= 2^n）
- `amplitudes` (numpy.ndarray)：复振幅数组，`amplitudes[i]` 对应基态 |i⟩ 的振幅

### 方法

#### pretty_print()

美观打印 Dirac 记号量子态到标准输出。

```python
pretty_print(decimals: int = 4, threshold: float = 1e-6)
```

**参数：**
- `decimals` (int)：振幅保留小数位，默认 4
- `threshold` (float)：振幅模长小于该阈值的项被省略，默认 1e-6

**示例：**
```python
from qsl import QuantumState
state = QuantumState(2)
state.h(0)
state.cnot(0, 1)
state.pretty_print()
# 输出: (0.7071+0j)|00⟩ + (0.7071+0j)|11⟩
```

---

#### measure()

按玻恩定则随机采样一个基态。

```python
measure(collapse: bool = False) -> Tuple[int, float]
```

**参数：**
- `collapse` (bool)：如果为 True，测量后态坍缩到测量结果，默认 False

**返回值：**
- `Tuple[int, float]`：(测量结果整数, 对应概率)

**示例：**
```python
state = QuantumState(3)
for q in range(3):
    state.h(q)
result, prob = state.measure()
print(f"测量结果: |{format(result, '03b')}⟩, 概率: {prob:.4f}")
```

---

#### expectation()

免采样直接计算解析期望值 ⟨ψ|O|ψ⟩。

```python
expectation(observable) -> float
```

**参数：**
- `observable`：可观测算符，支持三种格式：
  - Pauli 字符串，如 `"XXZ"`（qubit 0 在最左，I 表示恒等）
  - `[(coeff, pauli_str), ...]` 列表，如 `[(0.5, "ZZ"), (1.2, "XX")]`
  - (2^n, 2^n) 稠密 numpy 矩阵

**返回值：**
- `float`：期望值（自动取实部）

**示例：**
```python
state = QuantumState(2)
state.h(0)
state.cnot(0, 1)
# 计算 Z⊗Z 期望值
exp_zz = state.expectation("ZZ")
print(f"<ZZ> = {exp_zz:.4f}")  # Bell 态应为 1.0
```

---

## DensityMatrix

**密度矩阵类**，用于表示混合量子态，支持退相干信道和含噪声操作。

```python
rho = Σ_i p_i |ψ_i⟩⟨ψ_i|
```

### 构造函数

```python
DensityMatrix(n_qubits: int)
```

**参数：**
- `n_qubits` (int)：量子比特数（1 ≤ n_qubits ≤ 16）

### 静态方法

#### from_pure()

从纯态创建密度矩阵：ρ = |ψ⟩⟨ψ|。

```python
@staticmethod
from_pure(state: QuantumState) -> DensityMatrix
```

#### from_probabilities()

从概率混合创建混合态：ρ = Σ p_i |ψ_i⟩⟨ψ_i|。

```python
@staticmethod
from_probabilities(states: List[Tuple[float, QuantumState]]) -> DensityMatrix
```

### 方法

#### apply_depolarizing()

应用退极化信道：ρ → (1-p)ρ + p·I/N。

```python
apply_depolarizing(p: float)
```

**参数：**
- `p` (float)：退极化概率 [0, 1]

#### apply_amplitude_damping()

应用振幅阻尼（T1 衰减）信道到所有量子比特。

```python
apply_amplitude_damping(gamma: float)
```

**参数：**
- `gamma` (float)：阻尼概率 [0, 1]

#### apply_phase_damping()

应用相位阻尼（T2 退相干）信道。

```python
apply_phase_damping(gamma: float)
```

**参数：**
- `gamma` (float)：退相位概率 [0, 1]

#### measure()

在计算基下测量系统。

```python
measure(collapse: bool = False) -> Tuple[int, float]
```

**参数：**
- `collapse` (bool)：测量后是否坍缩，默认 False

**返回值：**
- `Tuple[int, float]`：(结果索引, 概率)

**属性：**
- `n_qubits` (int)：量子比特数
- `dim` (int)：希尔伯特空间维度
- `purity()`：计算纯度 Tr(ρ²)
- `von_neumann_entropy()`：计算冯·诺依曼熵 S = -Tr(ρ log₂ ρ)

---

## NoiseModel

**噪声模型**，密度矩阵模拟路径的信道参数集合。

### 构造参数

```python
NoiseModel(
    depolarizing: float = 0.0,
    amplitude_damping: float = 0.0,
    phase_damping: float = 0.0,
    readout_error: float = 0.0
)
```

**参数：**
- `depolarizing` (float)：每个门作用后应用的退极化概率 [0, 1]，默认 0.0
- `amplitude_damping` (float)：每个门作用后应用的振幅阻尼（T1）概率 [0, 1]，默认 0.0
- `phase_damping` (float)：每个门作用后应用的相位阻尼（T2）概率 [0, 1]，默认 0.0
- `readout_error` (float)：读出错误概率，每个比特测量时翻转的概率 [0, 1]，默认 0.0

---

## GroverSearch

**Grover 量子搜索算法**类，封装完整的 Grover 迭代流程，包括自动计算最优迭代次数、自动构建 Oracle。

### 构造函数

```python
GroverSearch(n_qubits: int, verbose: bool = False)
```

**参数：**
- `n_qubits` (int)：量子比特数（搜索空间 N = 2^n）
- `verbose` (bool)：是否输出详细计算日志，默认 False

### 方法

#### search()

执行 Grover 搜索（黑盒 Oracle 路径）。

```python
search(
    condition: Callable[[int], bool],
    num_solutions: Optional[int] = None,
    shots: int = 1
) -> GroverResult
```

**参数：**
- `condition` (Callable[[int], bool])：Oracle 函数 f(x) -> bool，True 表示 x 是解
- `num_solutions` (Optional[int])：已知解的数量（None 则自动统计，O(N) 模拟开销），默认 None
- `shots` (int)：测量次数，默认 1

**返回值：**
- `GroverResult`：包含搜索的全部结果和元数据

**异常：**
- `NoSolutionError`：搜索空间无解时抛出

**示例：**
```python
from qsl import GroverSearch

# 搜索小于 16 的偶数
searcher = GroverSearch(4, verbose=True)
result = searcher.search(lambda x: x % 2 == 0, shots=100)
print(f"找到的解: {result.get_solutions()}")
print(f"经验成功率: {result.empirical_success_rate:.2%}")
```

---

#### search_with_oracle_set()

使用预标记的状态集合执行 Grover 搜索（跳过条件函数枚举步骤，更高效）。

```python
search_with_oracle_set(
    marked_states: Set[int],
    shots: int = 1
) -> GroverResult
```

**参数：**
- `marked_states` (Set[int])：解的基态索引集合
- `shots` (int)：测量次数，默认 1

**返回值：**
- `GroverResult`

**示例：**
```python
searcher = GroverSearch(4)
result = searcher.search_with_oracle_set({3, 5, 9}, shots=100)
print(f"标记态: |0011⟩, |0101⟩, |1001⟩")
print(f"计数: {result.get_measurement_counts()}")
```

---

#### get_solutions()

（GroverResult 方法）返回所有测量到的解。

```python
# 通过 GroverResult 实例调用
result.get_solutions() -> List[int]
```

**返回值：**
- `List[int]`：去重后的解列表

---

## GroverResult

Grover 搜索的完整结果数据类。

### 属性

- `n_qubits` (int)：量子比特数
- `num_solutions` (Optional[int])：解的数量 M（BBHT 路径下为 None）
- `iterations` (Optional[int])：实际执行的 Grover 迭代次数
- `theta` (Optional[float])：理论角度（弧度）
- `theory_success_prob` (Optional[float])：理论成功概率
- `measurements` (List[Tuple[int, float, bool]])：测量结果列表 [(结果整数, 概率, 是否为解), ...]
- `quantum_queries` (Optional[int])：实际 Oracle 查询次数
- `empirical_success_rate` (float)：经验成功概率（= 成功次数 / 总测量次数）
- `shots` (int)：总测量次数

### 方法

#### get_solutions()

返回所有测量到的解（去重后）。

```python
get_solutions() -> List[int]
```

#### get_measurement_counts()

返回测量结果的计数分布。

```python
get_measurement_counts() -> Dict[int, int]
```

**返回值：**
- `Dict[int, int]`：{基态整数: 出现次数}

**示例：**
```python
counts = result.get_measurement_counts()
for state, count in sorted(counts.items()):
    print(f"|{format(state, '04b')}⟩: {count}")
```

#### best_measurement()

返回概率最大的测量结果。

```python
best_measurement() -> Tuple[int, float]
```

**返回值：**
- `Tuple[int, float]`：(基态整数, 概率)

---

## solve_sat()

使用 Grover 搜索求解 SAT 问题（CNF 形式）。CNF 公式直接编译为量子相位 Oracle 电路（X/CNOT/Toffoli/Z 门），不做经典枚举。解数量 M 未知时使用 BBHT 指数搜索，期望 O(√(N/M)) 次 Oracle 查询。

```python
def solve_sat(
    cnf_clauses: list[list[int]],
    n_qubits: int,
    shots: int = 100,
    verbose: bool = False
) -> GroverResult
```

**参数：**
- `cnf_clauses` (list[list[int]])：CNF 子句列表。每个子句是文字列表，正整数表示变量，负整数表示取反变量。
  例如 `[[1, -2, 3], [-1, 2], [-3]]` 表示 `(x1 ∨ ¬x2 ∨ x3) ∧ (¬x1 ∨ x2) ∧ (¬x3)`
- `n_qubits` (int)：布尔变量数量
- `shots` (int)：测量次数，默认 100
- `verbose` (bool)：是否打印详细进度，默认 False

**返回值：**
- `GroverResult`：包含搜索结果

**示例：**
```python
from qsl.algorithms import solve_sat

# (x0 ∨ x1) ∧ (¬x0 ∨ x1) ∧ (x0 ∨ ¬x2)
cnf = [[1, 2], [-1, 2], [1, -3]]
result = solve_sat(cnf, n_qubits=3, shots=50, verbose=True)
print(f"解: {[format(s, '03b') for s in result.get_solutions()]}")
```

---

## parse_bool()

解析布尔表达式字符串为抽象语法树（AST）。

```python
def parse_bool(expression: str) -> BooleanExpr
```

**参数：**
- `expression` (str)：布尔表达式字符串，如 `"x0 & x1 | ~x2"`

**语法（按优先级递增）：**
- `|`：OR（最低优先级）
- `^`：XOR
- `&`：AND
- `~`：NOT（右结合）
- 变量格式：`x0`, `x1`, `x2`, ...（x 后跟数字）
- 括号 `()` 可改变优先级

**返回值：**
- `BooleanExpr`：AST 根节点，支持 `evaluate(assignment: int) -> bool` 方法

**异常：**
- `BooleanParseError`：语法错误时抛出（带源码位置）

**示例：**
```python
from qsl import parse_bool

expr = parse_bool("x0 & (x1 | ~x2)")
# 评估赋值 x0=1, x1=0, x2=0 (二进制 001 = 整数 1)
print(expr.evaluate(0b001))  # True
print(expr.evaluate(0b100))  # False (x0=0)
```
