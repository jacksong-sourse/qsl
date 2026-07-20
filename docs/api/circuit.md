[← 返回文档中心](../index.md)

# 电路 API 参考

电路模块提供对标 Qiskit 的量子电路对象模型，支持门操作、参数化、电路变换、执行模拟以及与其他框架的互转。

---

## QuantumCircuit

**量子电路类**，是构建量子程序的核心对象。支持逐门追加、电路拼接、逆置、分解、参数化绑定、执行模拟等功能。

### 构造函数

```python
QuantumCircuit(
    num_qubits: int,
    name: str = "",
    global_phase: float = 0.0
)
```

**参数：**
- `num_qubits` (int)：量子比特数（必须 ≥ 1）
- `name` (str)：电路名称，默认空字符串
- `global_phase` (float)：全局相位（弧度），默认 0.0

**属性：**
- `num_qubits` (int)：量子比特数
- `n_qubits` (int)：`num_qubits` 的别名（接口统一）
- `data` (List[Instruction])：指令列表
- `name` (str)：电路名
- `global_phase` (float)：全局相位
- `parameters` (set)：电路中所有符号参数的集合

**示例：**
```python
from qsl.circuit import QuantumCircuit

qc = QuantumCircuit(2, name="Bell")
qc.h(0)
qc.cx(0, 1)
print(qc.draw())
```

---

### 单量子比特门

所有单比特门方法均返回 `self`，支持链式调用。

| 方法 | 说明 | 签名 |
|------|------|------|
| `h(q)` | Hadamard 门 | `h(q: int)` |
| `x(q)` | Pauli-X 门（量子 NOT） | `x(q: int)` |
| `y(q)` | Pauli-Y 门 | `y(q: int)` |
| `z(q)` | Pauli-Z 门（相位翻转） | `z(q: int)` |
| `s(q)` | S 门（√Z，相位门） | `s(q: int)` |
| `t(q)` | T 门（π/8 门，√S） | `t(q: int)` |
| `sdg(q)` | S† 门（S 的逆） | `sdg(q: int)` |
| `tdg(q)` | T† 门（T 的逆） | `tdg(q: int)` |
| `sx(q)` | √X 门 | `sx(q: int)` |
| `sxdg(q)` | √X† 门 | `sxdg(q: int)` |
| `id(q)` | 恒等门 | `id(q: int)` |

### 旋转门

| 方法 | 说明 | 签名 |
|------|------|------|
| `rx(q, theta)` | 绕 X 轴旋转 θ | `rx(q: int, theta: float)` |
| `ry(q, theta)` | 绕 Y 轴旋转 θ | `ry(q: int, theta: float)` |
| `rz(q, phi)` | 绕 Z 轴旋转 φ | `rz(q: int, phi: float)` |
| `p(q, lam)` | 相位门 P(λ) | `p(q: int, lam: float)` |
| `u(q, theta, phi, lam)` | 通用单比特门 U(θ,φ,λ) | `u(q: int, theta: float, phi: float, lam: float)` |

**参数说明：**
- `q` (int)：目标量子比特索引
- `theta`, `phi`, `lam` (float)：旋转角度（弧度）

---

### 两量子比特门

| 方法 | 说明 | 签名 |
|------|------|------|
| `cx(c, t)` | 受控 NOT（CNOT） | `cx(control: int, target: int)` |
| `cy(c, t)` | 受控 Y | `cy(control: int, target: int)` |
| `cz(c, t)` | 受控 Z | `cz(control: int, target: int)` |
| `ch(c, t)` | 受控 H | `ch(control: int, target: int)` |
| `cs(c, t)` | 受控 S | `cs(control: int, target: int)` |
| `ct(c, t)` | 受控 T | `ct(control: int, target: int)` |
| `swap(a, b)` | SWAP 门 | `swap(q0: int, q1: int)` |
| `iswap(a, b)` | iSWAP 门 | `iswap(q0: int, q1: int)` |
| `dcx(a, b)` | DCX 门（双 CNOT） | `dcx(q0: int, q1: int)` |
| `ecr(a, b)` | ECR 门（回声交叉共振） | `ecr(q0: int, q1: int)` |

### 受控旋转门

| 方法 | 说明 | 签名 |
|------|------|------|
| `crx(theta, c, t)` | 受控 RX | `crx(theta: float, control: int, target: int)` |
| `cry(theta, c, t)` | 受控 RY | `cry(theta: float, control: int, target: int)` |
| `crz(theta, c, t)` | 受控 RZ | `crz(theta: float, control: int, target: int)` |
| `cp(lam, c, t)` | 受控相位 | `cp(lam: float, control: int, target: int)` |
| `cu(theta, phi, lam, c, t, gamma=0.0)` | 受控 U | `cu(theta, phi, lam, control, target, gamma=0.0)` |

### 两比特纠缠门

| 方法 | 说明 | 签名 |
|------|------|------|
| `rxx(theta, a, b)` | XX 旋转（Ising XX） | `rxx(theta: float, q0: int, q1: int)` |
| `ryy(theta, a, b)` | YY 旋转 | `ryy(theta: float, q0: int, q1: int)` |
| `rzz(theta, a, b)` | ZZ 旋转（Ising ZZ） | `rzz(theta: float, q0: int, q1: int)` |

---

### 三量子比特及多量子比特门

| 方法 | 说明 | 签名 |
|------|------|------|
| `ccx(a, b, t)` | Toffoli 门（CCNOT，双控制非） | `ccx(c1: int, c2: int, target: int)` |
| `cswap(c, a, b)` | Fredkin 门（受控 SWAP） | `cswap(control: int, q0: int, q1: int)` |
| `mcx(controls, target)` | 多控制 X 门 | `mcx(controls: Sequence[int], target: int)` |
| `mcz(controls)` | 多控制 Z 门 | `mcz(qubits: Sequence[int])` |
| `barrier(qubits=None)` | 屏障（可视化/编译提示） | `barrier(qubits: Optional[Sequence[int]] = None)` |

---

### 电路操作方法

#### inverse()

返回逆电路：逆序排列指令 + 每个门取逆。

```python
inverse(inplace: bool = False) -> QuantumCircuit
```

**参数：**
- `inplace` (bool)：是否就地修改，默认 False（返回新电路）

---

#### compose()

拼接另一个电路。

```python
compose(
    other: QuantumCircuit,
    qubits: Optional[Sequence[int]] = None,
    inplace: bool = False,
    front: bool = False
) -> QuantumCircuit
```

**参数：**
- `other` (QuantumCircuit)：要拼接的电路
- `qubits` (Optional[Sequence[int]])：other 的比特映射到本电路的比特（默认恒等映射）
- `inplace` (bool)：是否就地修改，默认 False
- `front` (bool)：True 时把 other 插到本电路之前，默认 False（追加到末尾）

---

#### decompose()

把多比特门拆解到基础门集。

```python
decompose(
    basis: Optional[set] = None,
    reps: int = 10
) -> QuantumCircuit
```

**参数：**
- `basis` (Optional[set])：目标门名集合，默认 `{cx, id, rz, ry, sx, x, z, h, s, sdg, t, tdg, p, swap}`
- `reps` (int)：递归分解最大轮数，默认 10

---

#### transpile()

转译入口：分解到基础门集 + 可选布局/SWAP 插入 + 门消减优化。

```python
transpile(
    basis_gates: Optional[set] = None,
    coupling_map: Optional[List[Tuple[int, int]]] = None,
    optimization_level: int = 1,
    seed: Optional[int] = None
) -> QuantumCircuit
```

**参数：**
- `basis_gates` (Optional[set])：目标门集
- `coupling_map` (Optional[List[Tuple[int, int]]])：设备耦合边列表
- `optimization_level` (int)：优化级别（0=不优化, 1=消恒等/合并旋转, 2=再加对消），默认 1
- `seed` (Optional[int])：随机种子

---

#### reverse_bits()

返回一个新电路，所有比特索引按 `[n-1, ..., 0]` 反转。

```python
reverse_bits() -> QuantumCircuit
```

---

#### copy()

返回电路的深拷贝。

```python
copy() -> QuantumCircuit
```

---

### 参数化电路

#### Parameter

符号参数类，用于构建变分电路。

```python
Parameter(name: str)
```

**参数：**
- `name` (str)：参数名称

---

#### bind_parameters()

将符号参数绑定为数值。

```python
bind_parameters(
    mapping: Dict[Parameter, float],
    inplace: bool = False
) -> QuantumCircuit
```

**参数：**
- `mapping` (Dict[Parameter, float] 或 Dict[str, float])：{Parameter: value} 或 {参数名: value}
- `inplace` (bool)：是否就地修改，默认 False

---

#### assign_parameters()

`bind_parameters` 的别名（Qiskit 兼容）。

```python
assign_parameters(mapping, inplace: bool = False) -> QuantumCircuit
```

---

#### assign()

`assign_parameters` 的简写。

```python
assign(mapping, inplace: bool = False) -> QuantumCircuit
```

**参数化电路示例：**
```python
from qsl.circuit import QuantumCircuit, Parameter

theta = Parameter("θ")
qc = QuantumCircuit(1)
qc.rx(0, theta)
qc.ry(0, theta * 2)

bound = qc.bind_parameters({theta: 0.5})
```

---

### 执行方法

#### execute()

运行电路并采样，返回 ExecutionResult。

```python
execute(
    shots: int = 1024,
    seed: Optional[int] = None,
    initial_state: Optional[np.ndarray] = None
) -> ExecutionResult
```

**参数：**
- `shots` (int)：采样次数，默认 1024
- `seed` (Optional[int])：随机种子（结果可复现）
- `initial_state` (Optional[np.ndarray])：可选初始态向量（默认 |0...0⟩）

**返回值：**
- `ExecutionResult`：包含 counts、statevector 等

---

#### measure_all()

便捷方法：直接执行电路并返回测量计数字典。

```python
measure_all(
    shots: int = 1024,
    seed: Optional[int] = None
) -> Dict[int, int]
```

**参数：**
- `shots` (int)：采样次数，默认 1024
- `seed` (Optional[int])：随机种子

**返回值：**
- `Dict[int, int]`：{基态整数: 出现次数}

---

#### execute_density()

密度矩阵模拟路径，支持噪声信道。

```python
execute_density(
    shots: int = 1024,
    seed: Optional[int] = None,
    noise: Optional[NoiseModel] = None,
    initial_state: Optional[np.ndarray] = None
) -> ExecutionResult
```

**参数：**
- `shots` (int)：采样次数，默认 1024
- `seed` (Optional[int])：随机种子
- `noise` (Optional[NoiseModel])：噪声模型实例，None 为理想演化
- `initial_state` (Optional[np.ndarray])：可选初始态向量

**注意：** 密度矩阵维度为 4^n，建议 n ≤ 12。

---

#### expectation()

免采样直接计算解析期望值 ⟨ψ|O|ψ⟩。

```python
expectation(
    observable,
    initial_state: Optional[np.ndarray] = None
) -> float
```

**参数：**
- `observable`：可观测算符（Pauli 字符串/Pauli 串列表/矩阵）
- `initial_state` (Optional[np.ndarray])：可选初始态

---

#### statevector()

返回电路作用后的态向量。

```python
statevector(initial_state: Optional[np.ndarray] = None) -> np.ndarray
```

---

#### draw()

绘制电路。

```python
draw(output: str = "text", **kwargs)
```

**参数：**
- `output` (str)：输出格式，`"text"` 返回 ASCII 字符串；`"mpl"`/`"matplotlib"` 调用 matplotlib 绘制，返回 `(fig, ax)`
- `**kwargs`：传递给底层绘制函数（如 `ax`, `style`, `fold` 等）

**示例：**
```python
qc = QuantumCircuit(2)
qc.h(0); qc.cx(0, 1)
print(qc.draw())  # 文本绘制
fig, ax = qc.draw(output="mpl", style="iqp")  # matplotlib 绘制
```

---

### 统计属性与方法

| 方法/属性 | 说明 |
|-----------|------|
| `size()` | 返回指令总数 |
| `depth()` | 返回电路深度（贪心层调度） |
| `count_ops()` | 返回按门名统计的数量字典 `Dict[str, int]` |
| `width()` | 返回电路宽度（量子比特数） |
| `num_nonlocal_gates()` | 返回非局部门数（≥2 比特的门，barrier 不计） |

---

## Gate

**通用量子门类**，支持控制位添加、矩阵幂、逆门等代数操作。

### 构造函数

```python
Gate(
    name: str,
    matrix: Optional[np.ndarray] = None,
    num_qubits: int = 1,
    params: Optional[Sequence] = None,
    matrix_fn=None,
    label: Optional[str] = None,
    definition: Optional[list] = None
)
```

**参数：**
- `name` (str)：门名（如 "h", "cx", "rz"）
- `matrix` (Optional[np.ndarray])：(2^k, 2^k) 复数酉矩阵；参数化门可为 None
- `num_qubits` (int)：作用比特数 k
- `params` (Optional[Sequence])：门的参数列表（数值或 ParameterExpression）
- `matrix_fn`：可选，params -> ndarray 的函数，用于参数化门
- `label` (Optional[str])：绘图标签
- `definition` (Optional[list])：可选的门级展开定义

### 属性

- `name` (str)：门名
- `num_qubits` (int)：作用比特数
- `params` (list)：参数列表
- `label` (str)：绘图标签
- `is_parameterized` (bool)：是否仍含未绑定的符号参数

### 方法

#### control()

生成 n 控制位版本的门。

```python
control(n: int = 1) -> Gate
```

**参数：**
- `n` (int)：控制位数，默认 1

**返回值：**
- `Gate`：受控版本的新门

---

#### power()

门的 k 次幂 U^k（通过对角化计算，支持分数/负数幂）。

```python
power(k: float) -> Gate
```

**参数：**
- `k` (float)：幂次

---

#### inverse()

返回逆门（共轭转置）。

```python
inverse(name_suffix: str = "†") -> Gate
```

---

#### to_matrix()

返回数值酉矩阵（numpy ndarray）。含未绑定参数时抛出 ValueError。

```python
to_matrix() -> np.ndarray
```

---

#### copy()

返回门的深拷贝。

```python
copy() -> Gate
```

---

## Instruction

**电路指令类**，封装一个门及其作用的量子比特。

```python
Instruction(gate: Gate, qubits: Sequence[int])
```

**属性：**
- `gate` (Gate)：量子门对象
- `qubits` (Tuple[int, ...])：作用的量子比特索引元组

---

## ExecutionResult

电路执行结果类，包含测量计数、最终态向量等信息。

### 属性

- `counts` (Dict[int, int])：测量计数 {基态整数: 次数}
- `state`：最终量子态（QuantumState 或 DensityMatrix）
- `circuit` (QuantumCircuit)：执行的电路
- `shots` (int)：总测量次数

### 方法

#### get_counts()

返回测量计数，可选二进制字符串键。

```python
get_counts(binary: bool = False) -> Dict
```

**参数：**
- `binary` (bool)：True 时键为二进制字符串（如 "011"），默认 False 为整数

---

#### probabilities()

返回测量概率分布列表（按基态索引顺序）。

```python
probabilities() -> List[float]
```

---

#### probabilities_dict()

返回测量概率分布字典。

```python
probabilities_dict() -> Dict[int, float]
```

---

#### most_frequent()

返回出现频率最高的 n 个测量结果。

```python
most_frequent(n: int = 1) -> List[Tuple[int, int]]
```

**参数：**
- `n` (int)：返回前 n 个，默认 1

**返回值：**
- `List[Tuple[int, int]]`：[(基态, 次数), ...]，按次数降序排列

---

#### statevector()

返回最终态向量的 numpy 数组。

```python
statevector() -> np.ndarray
```

**示例：**
```python
qc = QuantumCircuit(2)
qc.h(0); qc.cx(0, 1)
result = qc.execute(shots=1000)
print("计数:", result.counts)
print("概率:", result.probabilities_dict())
print("最频繁:", result.most_frequent(1))
```

---

## library（标准电路库）

常用量子电路的工厂函数模块，通过 `qsl.circuit.library` 访问。

### bell_state()

Bell 态制备电路（2 比特）。

```python
bell_state(which: str = "phi+") -> QuantumCircuit
```

**参数：**
- `which` (str)：Bell 态类型，可选 `"phi+"`, `"phi-"`, `"psi+"`, `"psi-"`，默认 `"phi+"`

---

### ghz_state()

GHZ 纠缠态制备电路。

```python
ghz_state(n: int) -> QuantumCircuit
```

**参数：**
- `n` (int)：量子比特数

---

### w_state()

W 态制备电路：(|10..0⟩ + |01..0⟩ + ... + |0..01⟩)/√n。

```python
w_state(n: int) -> QuantumCircuit
```

**参数：**
- `n` (int)：量子比特数

---

### qft()

量子傅里叶变换电路。

```python
qft(
    n: int,
    inverse: bool = False,
    do_swaps: bool = True,
    approximation_degree: int = 0
) -> QuantumCircuit
```

**参数：**
- `n` (int)：比特数
- `inverse` (bool)：是否逆 QFT（IQFT），默认 False
- `do_swaps` (bool)：末尾是否带 SWAP 反转比特序，默认 True
- `approximation_degree` (int)：忽略小角度受控相位的程度，0 为精确

---

### grover_diffusion()

Grover 扩散算子电路：D = H^⊗n X^⊗n MCZ X^⊗n H^⊗n。

```python
grover_diffusion(n: int) -> QuantumCircuit
```

**参数：**
- `n` (int)：量子比特数

---

### teleportation()

量子隐形传态电路（3 比特）。

```python
teleportation() -> QuantumCircuit
```

---

### random_circuit()

随机电路生成器（用于基准测试/量子优越性演示）。

```python
random_circuit(
    n: int,
    depth: int,
    seed: Optional[int] = None,
    gate_set: Sequence[str] = ("h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "cx", "cz")
) -> QuantumCircuit
```

**参数：**
- `n` (int)：比特数
- `depth` (int)：层数
- `seed` (Optional[int])：随机种子
- `gate_set` (Sequence[str])：候选门集合

---

### quantum_walk_cycle()

环上离散时间量子行走的一步。

```python
quantum_walk_cycle(n_positions: int) -> QuantumCircuit
```

**参数：**
- `n_positions` (int)：位置数，用 ⌈log₂(n_positions)⌉ 比特编码 + 1 个硬币比特

**示例：**
```python
from qsl.circuit import library

bell = library.bell_state("phi+")
ghz = library.ghz_state(3)
qft_circ = library.qft(4, inverse=False)
rand_circ = library.random_circuit(5, depth=10, seed=42)
```

---

## QASM 序列化

### dumps_qasm2()

将电路序列化为 OpenQASM 2.0 字符串。

```python
dumps_qasm2(circuit: QuantumCircuit) -> str
```

---

### loads_qasm2()

从 OpenQASM 2.0 字符串解析电路。

```python
loads_qasm2(qasm_str: str) -> QuantumCircuit
```

---

### dumps_qasm3()

将电路序列化为 OpenQASM 3.0 字符串。

```python
dumps_qasm3(circuit: QuantumCircuit) -> str
```

**示例：**
```python
from qsl.circuit import QuantumCircuit, dumps_qasm2, loads_qasm2

qc = QuantumCircuit(2)
qc.h(0); qc.cx(0, 1)
qasm_str = dumps_qasm2(qc)
qc2 = loads_qasm2(qasm_str)
```

---

## 框架转换器

### to_qiskit()

将 QSL 电路转换为 Qiskit QuantumCircuit。

```python
to_qiskit(circuit: QuantumCircuit):
```

**返回值：**
- `qiskit.QuantumCircuit`（需安装 qiskit）

---

### from_qiskit()

从 Qiskit QuantumCircuit 转换为 QSL 电路。

```python
from_qiskit(qiskit_circuit) -> QuantumCircuit
```

---

### to_cirq()

将 QSL 电路转换为 Cirq 电路。

```python
to_cirq(circuit: QuantumCircuit):
```

**返回值：**
- `cirq.Circuit`（需安装 cirq）
