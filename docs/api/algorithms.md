[← 返回文档中心](../index.md)

# 算法 API 参考

qsl.algorithms 模块提供量子算法实现，包括量子傅里叶变换、Shor 整数分解、QAOA 组合优化和 VQE 变分量子特征求解器。

---

## QuantumFourierTransform

量子傅里叶变换（QFT）类。

QFT|j⟩ = 1/√N · Σₖ₌₀^{N-1} exp(2πi·j·k/N) |k⟩，其中 N = 2^n_qubits。

### 构造函数

```python
QuantumFourierTransform(n_qubits: int)
```

**参数：**
- `n_qubits` (int): 量子比特数

**属性：**
- `n_qubits` (int): 量子比特数
- `N` (int): 希尔伯特空间维度 = 2^n_qubits

### 方法

#### build_circuit()

构建 QFT 电路，返回门操作列表。

```python
def build_circuit(self) -> list
```

**返回：**
- `list`: 门操作列表，每个元素为字典，包含 'gate'、'targets'、'params'、'control' 等键

**示例：**
```python
from qsl.algorithms.qft import QuantumFourierTransform

qft = QuantumFourierTransform(3)
circuit = qft.build_circuit()
print(f"QFT 电路包含 {len(circuit)} 个门操作")
```

#### get_matrix()

计算完整的 QFT 酉矩阵。

```python
def get_matrix(self) -> np.ndarray
```

**返回：**
- `np.ndarray`: N×N 的复数酉矩阵，F_{jk} = 1/√N · exp(2πi·j·k/N)

**示例：**
```python
import numpy as np
from qsl.algorithms.qft import QuantumFourierTransform

qft = QuantumFourierTransform(2)
F = qft.get_matrix()
# 验证酉性: F†F = I
print(f"酉性验证: {np.allclose(F.conj().T @ F, np.eye(4))}")
```

#### inverse()

返回逆 QFT（IQFT）电路。

```python
def inverse(self) -> list
```

**返回：**
- `list`: IQFT 门操作列表

#### apply(state_vector)

将 QFT 应用于态向量（基于电路的模拟）。

```python
def apply(self, state_vector: np.ndarray) -> np.ndarray
```

**参数：**
- `state_vector` (np.ndarray): 形状为 (N,) 的复数态向量

**返回：**
- `np.ndarray`: 变换后的态向量

**示例：**
```python
import numpy as np
from qsl.algorithms.qft import QuantumFourierTransform

qft = QuantumFourierTransform(2)
state = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
transformed = qft.apply(state)
print(f"QFT(|00⟩) = {transformed}")
```

---

## ShorSolver

Shor 整数因子分解算法。使用量子相位估计（受控模幂 + 逆 QFT）查找模幂的周期，然后通过经典后处理提取因子。

### 构造函数

```python
ShorSolver(N: int, max_control_qubits: int = 18, allow_classical_fallback: bool = False)
```

**参数：**
- `N` (int): 要分解的复合整数（必须 ≥ 2）
- `max_control_qubits` (int): 相位估计控制寄存器的最大量子比特数，默认 18
- `allow_classical_fallback` (bool): 超出模拟能力时是否回退到经典周期查找，默认 False

**属性：**
- `N` (int): 待分解的整数
- `factors` (list[int] | None): 计算出的因子（未计算时为 None）

### 方法

#### factor()

使用 Shor 算法分解 N。

```python
def factor(self, max_attempts: int = 10, _max_depth: int = 100, _depth: int = 0) -> list[int]
```

**参数：**
- `max_attempts` (int): 尝试的最大随机 a 值数量，默认 10

**返回：**
- `list[int]`: 质因子列表（确保每个因子都是质数）

**异常：**
- `RuntimeError`: 所需控制寄存器超过 max_control_qubits 且未允许经典回退
- `RecursionError`: 递归深度超过 _max_depth

**示例：**
```python
from qsl.algorithms.shor import ShorSolver

# 分解 15 = 3 × 5
solver = ShorSolver(15)
factors = solver.factor()
print(f"15 的素因子: {factors}")

# 分解 21 = 3 × 7
solver21 = ShorSolver(21)
print(f"21 = {' × '.join(map(str, solver21.factor()))}")
```

---

## QAOA

量子近似优化算法（Quantum Approximate Optimization Algorithm），用于求解组合优化问题。

### 构造函数

```python
QAOA(n_qubits: int, cost_matrix: np.ndarray, p: int = 1, encoding: str = "ising")
```

**参数：**
- `n_qubits` (int): 量子比特数（变量数）
- `cost_matrix` (np.ndarray): 对称 n×n 矩阵，定义问题代价
  - `encoding="ising"`: Cost = Σᵢ Q[i][i]·sᵢ + Σᵢ<ⱼ Q[i][j]·sᵢ·sⱼ，sᵢ ∈ {-1, +1}
  - `encoding="qubo"`: Cost = Σᵢ Q[i][i]·xᵢ + Σᵢ<ⱼ Q[i][j]·xᵢ·xⱼ，xᵢ ∈ {0, 1}
- `p` (int): QAOA 层数（深度），默认 1，必须 ≥ 1
- `encoding` (str): "ising"（默认）或 "qubo"

**属性：**
- `n_qubits` (int): 量子比特数
- `optimal_energy` (float | None): 找到的最优能量（优化前为 None）
- `optimal_bitstring` (int | None): 最优解的基态索引（整数形式，小端序编码）
- `optimal_bitstring_str` (str | None): 最优解的比特串（高位在左，如 "1010"）

### 方法

#### optimize()

运行 QAOA 优化以找到最优参数。

```python
def optimize(self, maxiter: int = 200, verbose: bool = False) -> tuple[np.ndarray, float]
```

**参数：**
- `maxiter` (int): 最大优化迭代