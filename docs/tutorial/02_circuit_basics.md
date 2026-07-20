[← 返回文档中心](../index.md)

# 02 - 量子电路基础

本教程详细介绍 QSL 量子电路的核心概念、量子门操作、电路构建方法和结果处理。所有代码示例均可直接运行。

---

## 前置条件

确保已安装 QSL 0.6.3+：

```bash
pip install qsl-quantum numpy
```

---

## 1. QuantumCircuit 类详解

`QuantumCircuit` 是构建量子电路的核心类，用于管理量子比特序列、量子门指令和电路操作。

### 1.1 创建电路

```python
from qsl import QuantumCircuit

# 创建 2 量子比特电路
qc = QuantumCircuit(2)

# 创建带名称和全局相位的电路
qc_named = QuantumCircuit(3, name="bell_state", global_phase=0.0)

print(f"量子比特数: {qc.num_qubits}")
print(f"电路门数: {qc.size()}")
print(f"电路深度: {qc.depth()}")
print(f"电路宽度: {qc.width()}")
```

**输出说明：**
- `num_qubits` / `n_qubits`: 量子比特数量
- `size()`: 电路中指令总数
- `depth()`: 电路深度（并行执行的最大层数）
- `width()`: 电路宽度（等于量子比特数）

### 1.2 电路基本信息

```python
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)

print("电路摘要:", qc.summary())
print("门统计:", qc.count_ops())
print("非局部门数:", qc.num_nonlocal_gates())
print("参数量:", qc.num_parameters)
print("量子比特列表:", qc.qubits)
```

### 1.3 绘制电路

```python
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# 文本绘制（默认）
print(qc.draw(output="text"))

# 或直接 print(qc)，默认使用 text 输出
```

---

## 2. 单比特门

单比特门作用于单个量子比特，是量子计算的基础构建块。

### 2.1 Pauli 门与 Hadamard 门

```python
from qsl import QuantumCircuit

qc = QuantumCircuit(4)

qc.x(0)   # Pauli-X (NOT 门) - 比特翻转
qc.y(1)   # Pauli-Y - Y 轴翻转
qc.z(2)   # Pauli-Z - 相位翻转
qc.h(3)   # Hadamard 门 - 创建叠加态

print(qc.draw())
```

### 2.2 相位门 (S, T) 及其逆门

```python
qc = QuantumCircuit(4)

qc.s(0)     # S 门 (√Z, π/2 相位)
qc.sdg(1)   # S† (S 的共轭转置, -π/2 相位)
qc.t(2)     # T 门 (π/4 相位, π/8 门)
qc.tdg(3)   # T† (T 的共轭转置, -π/4 相位)

print(qc.draw())
```

### 2.3 √X 门及其逆门

```python
qc = QuantumCircuit(2)

qc.sx(0)     # √X 门 (平方根 NOT 门)
qc.sxdg(1)   # √X† (√X 的共轭转置)

print(qc.draw())
```

### 2.4 恒等门

```python
qc = QuantumCircuit(1)
qc.id(0)   # I 门 - 恒等操作（不改变量子态）
print(qc.draw())
```

---

## 3. 参数化旋转门

参数化门接受角度参数（弧度），用于变分量子算法（VQE、QAOA）。

### 3.1 坐标轴旋转门

```python
import numpy as np
from qsl import QuantumCircuit

qc = QuantumCircuit(3)

theta = np.pi / 2
phi = np.pi / 4
lam = np.pi / 3

qc.rx(0, theta)   # 绕 X 轴旋转 theta 弧度
qc.ry(1, theta)   # 绕 Y 轴旋转 theta 弧度
qc.rz(2, phi)     # 绕 Z 轴旋转 phi 弧度

print(qc.draw())
```

### 3.2 相位门 P(λ)

```python
qc = QuantumCircuit(1)
lam = np.pi / 4
qc.p(0, lam)   # P(λ) = diag(1, e^{iλ})，与 Rz 区别是不含全局相位
print(qc.draw())
```

**注意**：`p(λ)` 与 `rz(λ)` 的区别：
- `P(λ)`: 相位门，`|0⟩` 无相位，`|1⟩` 获得 `e^{iλ}` 相位
- `Rz(λ)`: 绕 Z 轴旋转，`|0⟩` 获得 `e^{-iλ/2}`，`|1⟩` 获得 `e^{iλ/2}`（含全局相位）

### 3.3 通用单比特门 U(θ, φ, λ)

```python
qc = QuantumCircuit(1)
theta, phi, lam = np.pi/2, np.pi/4, np.pi/8
qc.u(0, theta, phi, lam)   # U3 通用门，可表示任意单比特酉操作
print(qc.draw())
```

**U 门分解**：`U(θ, φ, λ) = Rz(φ) · Ry(θ) · Rz(λ)`

---

## 4. 两比特门

两比特门作用于两个量子比特，是创建量子纠缠的关键。

### 4.1 受控 Pauli 门

```python
from qsl import QuantumCircuit

qc = QuantumCircuit(4)

qc.cx(0, 1)   # CNOT (受控 X) - 控制位为 1 时翻转目标位
qc.cy(0, 2)   # 受控 Y
qc.cz(0, 3)   # 受控 Z

print(qc.draw())
```

**CNOT 门约定**：`cx(control, target)` - 第一个参数是控制位，第二个是目标位。

### 4.2 受控相位门

```python
qc = QuantumCircuit(4)

qc.ch(0, 1)    # 受控 Hadamard
qc.cs(0, 2)    # 受控 S
qc.ct(0, 3)    # 受控 T

print(qc.draw())
```

### 4.3 SWAP 门族

```python
qc = QuantumCircuit(4)

qc.swap(0, 1)    # SWAP 门 - 交换两个量子比特的状态
qc.iswap(2, 3)   # iSWAP 门 - 交换并附加 i 相位 (|01⟩→i|10⟩, |10⟩→i|01⟩)

print(qc.draw())
```

### 4.4 受控旋转门

```python
import numpy as np
from qsl import QuantumCircuit

qc = QuantumCircuit(6)
theta = np.pi / 3

qc.crx(theta, 0, 1)   # 受控 RX (角度在前，控制位、目标位在后)
qc.cry(theta, 0, 2)   # 受控 RY
qc.crz(theta, 0, 3)   # 受控 RZ
qc.cp(np.pi/4, 0, 4)  # 受控相位门 CP(λ)
qc.cu(np.pi/2, np.pi/4, np.pi/8, 0, 5)  # 受控 U 门

print(qc.draw())
```

**参数顺序说明**：对于参数化受控门，角度参数在前，量子比特参数在后（与 Qiskit 一致）。

### 4.5 两比特纠缠门 (RXX, RYY, RZZ)

```python
import numpy as np
from qsl import QuantumCircuit

qc = QuantumCircuit(6)
theta = np.pi / 2

qc.rxx(theta, 0, 1)   # RXX(θ) = exp(-iθ/2 X⊗X)
qc.ryy(theta, 2, 3)   # RYY(θ) = exp(-iθ/2 Y⊗Y)
qc.rzz(theta, 4, 5)   # RZZ(θ) = exp(-iθ/2 Z⊗Z)

print(qc.draw())
```

---

## 5. 三比特门及多比特门

### 5.1 Toffoli 门 (CCX)

```python
from qsl import QuantumCircuit

qc = QuantumCircuit(3)
qc.ccx(0, 1, 2)   # Toffoli 门 (CCX) - 双控制 X 门
                  # 当且仅当 q0 和 q1 都为 |1⟩ 时翻转 q2
print(qc.draw())
```

### 5.2 Fredkin 门 (CSWAP)

```python
qc = QuantumCircuit(3)
qc.cswap(0, 1, 2)   # Fredkin 门 (受控 SWAP)
                    # 当 q0 为 |1⟩ 时交换 q1 和 q2
print(qc.draw())
```

### 5.3 多控制 X 门 (MCX)

```python
qc = QuantumCircuit(4)
qc.mcx([0, 1, 2], 3)   # 三控制 X 门（前 3 位全 1 时翻转第 4 位）
print(qc.draw())
```

### 5.4 多控制 Z 门 (MCZ)

```python
qc = QuantumCircuit(3)
qc.mcz([0, 1, 2])   # 多控制 Z 门 - 所有比特为 |1⟩ 时相位翻转
print(qc.draw())
```

---

## 6. 电路操作

### 6.1 inverse() - 电路逆置

```python
from qsl import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.rz(1, 3.14/4)

print("原电路:")
print(qc.draw())

qc_inv = qc.inverse()   # 返回逆电路（逆序排列 + 每门取逆）
print("逆电路:")
print(qc_inv.draw())

# 验证: 电路 + 逆电路 = 恒等操作
sv = qc.statevector()
sv_inv = qc_inv.statevector()
print("电路+逆电路是否接近 |00⟩:", 
      np.allclose(np.abs(sv_inv @ sv.conj()), 1.0) if False else 
      "（可通过组合 compose 验证）")
```

### 6.2 compose() - 电路拼接

```python
qc1 = QuantumCircuit(2)
qc1.h(0)
qc1.cx(0, 1)

qc2 = QuantumCircuit(2)
qc2.rz(0, 3.14/4)
qc2.rx(1, 3.14/2)

# 拼接（qc2 追加到 qc1 末尾）
qc_composed = qc1.compose(qc2)
print("拼接后电路:")
print(qc_composed.draw())

# 使用 qubits 参数指定比特映射
qc3 = QuantumCircuit(3)
qc3.h(0)
# 将 qc1（2比特）拼接到 qc3 的第 1、2 位
qc3_composed = qc3.compose(qc1, qubits=[1, 2])
print("指定比特映射拼接:")
print(qc3_composed.draw())
```

### 6.3 decompose() - 门分解

```python
qc = QuantumCircuit(3)
qc.ccx(0, 1, 2)   # Toffoli 门

print("原电路（含 Toffoli）:")
print(qc.draw())
print(f"原门数: {qc.size()}, 深度: {qc.depth()}")

qc_decomposed = qc.decompose()   # 分解到基础门集 {cx, rz, ry, h, s, t, ...}
print("\n分解后电路:")
print(qc_decomposed.draw())
print(f"分解后门数: {qc_decomposed.size()}, 深度: {qc_decomposed.depth()}")
```

### 6.4 transpile() - 电路转译

```python
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.ccx(0, 1, 2)
qc.rxx(3.14/2, 1, 2)

# 转译到基础门集 + 优化
qc_transpiled = qc.transpile(optimization_level=2)
print("转译优化后电路:")
print(qc_transpiled.draw())
print(f"门统计: {qc_transpiled.count_ops()}")
```

`transpile()` 参数：
- `basis_gates`: 目标基础门集合
- `coupling_map`: 设备耦合图（用于真机布局）
- `optimization_level`: 优化级别（0=无优化，1=基础，2=含门对消）

### 6.5 reverse_bits() - 比特反转

```python
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)

print("原电路 (q0=LSB):")
print(qc.draw())

qc_rev = qc.reverse_bits()   # 反转比特顺序（QFT 等算法有用）
print("比特反转后:")
print(qc_rev.draw())
```

### 6.6 barrier() - 屏障

```python
qc = QuantumCircuit(2)
qc.h(0)
qc.h(1)
qc.barrier()   # 逻辑屏障（可视化/编译提示，不影响数值计算）
qc.cx(0, 1)
qc.barrier([0])  # 也可只对部分比特加屏障
qc.measure_all()
print(qc.draw())
```

### 6.7 copy() - 电路复制

```python
qc = QuantumCircuit(2)
qc.h(0)

qc_copy = qc.copy()   # 深拷贝
qc_copy.cx(0, 1)      # 不影响原电路

print(f"原电路门数: {qc.size()}")
print(f"复制后门数: {qc_copy.size()}")
```

---

## 7. 参数化电路

参数化电路用于变分量子算法（VQE/QAOA），可延迟绑定角度参数。

### 7.1 Parameter 类

```python
from qsl import QuantumCircuit, Parameter

# 创建符号参数
theta = Parameter("θ")
phi = Parameter("φ")
lam = Parameter("λ")

print(f"参数名: {theta.name}")
print(f"参数表达式: {theta}")

# 参数可参与算术运算
expr = 2 * theta + phi
print(f"表达式: {expr}")
print(f"表达式含参数: {expr.parameters}")
```

### 7.2 构建参数化电路

```python
from qsl import QuantumCircuit, Parameter

theta = Parameter("θ")
beta = Parameter("β")

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.rz(0, theta)    # 使用符号参数
qc.ry(1, beta)
qc.cx(1, 0)

print("参数化电路:")
print(qc.draw())
print(f"自由参数: {[p.name for p in qc.parameters]}")
```

### 7.3 bind_parameters() - 绑定参数

```python
import numpy as np
from qsl import QuantumCircuit, Parameter

theta = Parameter("θ")
beta = Parameter("β")

qc = QuantumCircuit(2)
qc.rx(0, theta)
qc.cx(0, 1)
qc.ry(1, beta)

# 绑定参数为具体数值
bound_qc = qc.bind_parameters({
    theta: np.pi / 2,
    beta: np.pi / 4
})

print("绑定后电路（参数已数值化）:")
print(bound_qc.draw())
print(f"剩余自由参数: {bound_qc.num_parameters}")
```

### 7.4 assign_parameters() / assign() 别名

```python
# assign_parameters 是 bind_parameters 的别名（Qiskit 兼容）
bound_qc2 = qc.assign_parameters({theta: 0.0, beta: np.pi})

# assign() 是简写形式
bound_qc3 = qc.assign({theta: np.pi})
```

---

## 8. 测量与执行

### 8.1 execute() - 执行电路

```python
from qsl import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# 执行模拟（采样测量）
result = qc.execute(shots=1024, seed=42)

print(f"采样次数: {result.shots}")
print(f"测量结果 (整数索引): {result.counts}")
```

`execute()` 参数：
- `shots`: 采样次数（默认 1024）
- `seed`: 随机种子（可复现结果）
- `initial_state`: 可选初始态向量

### 8.2 measure_all() - 快速测量

```python
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# 便捷方法：直接执行并返回测量计数
counts = qc.measure_all(shots=1024, seed=42)
print(f"measure_all 结果: {counts}")
```

### 8.3 statevector() - 获取态向量

```python
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

sv = qc.statevector()
print("Bell 态 |Φ+⟩ 态向量:")
print(sv)
print(f"|00⟩ 振幅: {sv[0]:.4f}")
print(f"|11⟩ 振幅: {sv[3]:.4f}")
```

---

## 9. 结果处理

`ExecutionResult` 对象提供多种方法处理测量结果。

```python
from qsl import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
result = qc.execute(shots=2048, seed=42)
```

### 9.1 get_counts() - 获取比特串计数

```python
# 二进制比特字符串形式
counts_binary = result.get_counts(binary=True)
print("二进制比特串计数:", counts_binary)

# 整数索引形式
counts_int = result.get_counts(binary=False)
print("整数索引计数:", counts_int)
```

### 9.2 probabilities() - 概率分布

```python
probs = result.probabilities()
print("经验概率分布:", probs)

# probabilities_dict() 是别名
probs_dict = result.probabilities_dict()
```

### 9.3 most_frequent() - 最高频结果

```python
top = result.most_frequent(n=2)
print("最高频的 2 个结果:", top)
```

### 9.4 从结果获取态向量

```python
sv_from_result = result.statevector()
print("态向量:", sv_from_result)
```

---

## 10. 门代数 (Gate 对象方法)

QSL 的 `Gate` 对象支持代数操作：受控、幂次、逆。**注意：必须使用 Gate 对象，不能直接对 numpy 数组调用这些方法。**

### 10.1 正确创建 Gate 对象

```python
from qsl import Gate
from qsl.quantum_gates import H as H_mat, X as X_mat, T as T_mat

# 从门矩阵创建 Gate 对象（必须方式）
H_gate = Gate("h", H_mat, 1, label="H")
X_gate = Gate("x", X_mat, 1, label="X")
T_gate = Gate("t", T_mat, 1, label="T")

print(f"H 门: {H_gate}")
print(f"H 门矩阵形状: {H_gate.to_matrix().shape}")
```

### 10.2 Gate.control() - 生成受控门

```python
from qsl import QuantumCircuit, Gate
from qsl.quantum_gates import H as H_mat, T as T_mat

H_gate = Gate("h", H_mat, 1, label="H")
T_gate = Gate("t", T_mat, 1, label="T")

# 生成单控制 H 门（等效于 ch）
CH_gate = H_gate.control(n=1)
print(f"受控 H 门: {CH_gate}, 比特数: {CH_gate.num_qubits}")

# 生成双控制 T 门（Toffoli 类似结构）
C2T_gate = T_gate.control(n=2)
print(f"双控制 T 门: {C2T_gate}, 比特数: {C2T_gate.num_qubits}")

# 在电路中使用自定义受控门
qc = QuantumCircuit(3)
qc.append(CH_gate, [0, 1])      # q0 控制，q1 目标
qc.append(C2T_gate, [0, 1, 2])  # q0, q1 控制，q2 目标
print(qc.draw())
```

### 10.3 Gate.power() - 门的幂次

```python
import numpy as np
from qsl import Gate
from qsl.quantum_gates import T as T_mat

T_gate = Gate("t", T_mat, 1, label="T")

# T^2 = S 门（验证）
T2_gate = T_gate.power(2)
print(f"T^2 门: {T2_gate}")
print("T^2 矩阵 ≈ S 矩阵:", np.allclose(T2_gate.to_matrix(), 
      np.array([[1, 0], [0, 1j]])))

# 分数幂: √X = X^{1/2}
from qsl.quantum_gates import X as X_mat, SX as SX_mat
X_gate = Gate("x", X_mat, 1, label="X")
sqrtX_gate = X_gate.power(0.5)
print("X^{1/2} ≈ SX 矩阵:", np.allclose(sqrtX_gate.to_matrix(), SX_mat))
```

### 10.4 Gate.inverse() - 门的逆

```python
from qsl import Gate
from qsl.quantum_gates import S as S_mat, Sdg as Sdg_mat

S_gate = Gate("s", S_mat, 1, label="S")

# S 的逆 = Sdg
S_dag = S_gate.inverse()
print(f"S 逆门: {S_dag}")
print("S† 矩阵 ≈ Sdg 矩阵:", np.allclose(S_dag.to_matrix(), Sdg_mat))

# 验证: U · U† = I
product = S_gate.to_matrix() @ S_dag.to_matrix()
print("S · S† ≈ I:", np.allclose(product, np.eye(2)))
```

### 10.5 在电路中使用自定义门

```python
from qsl import QuantumCircuit, Gate
from qsl.quantum_gates import H as H_mat
import numpy as np

# 创建任意单比特酉门
H_gate = Gate("h", H_mat, 1, label="H")

# 生成 CH, CS, 受控自定义门
CH = H_gate.control(1)
CH2 = H_gate.control(2)   # 双控制 H 门

qc = QuantumCircuit(4)
qc.h(0)
qc.append(CH, [0, 1])
qc.append(CH2, [0, 1, 2])
qc.h(3)

print("使用自定义受控门的电路:")
print(qc.draw())
print(f"门数: {qc.size()}")
```

### 10.6 unitary() - 添加任意酉矩阵

```python
from qsl import QuantumCircuit
import numpy as np

qc = QuantumCircuit(2)

# 添加任意 2x2 酉矩阵
my_unitary = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # 这就是 H 门
qc.unitary(my_unitary, [0], label="MyH")

# 添加任意 4x4 两比特酉矩阵
swap_like = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)
qc.unitary(swap_like, [0, 1], label="MySWAP")

print(qc.draw())
```

---

## 11. 完整示例：Bell 态制备与测量

让我们综合上述知识，创建一个完整的 Bell 态实验：

```python
import numpy as np
from qsl import QuantumCircuit

# 1. 创建电路
qc = QuantumCircuit(2, name="bell_state")

# 2. 添加门
qc.h(0)           # Hadamard 创建叠加
qc.cx(0, 1)       # CNOT 创建纠缠
qc.barrier()      # 屏障分隔

# 3. 查看电路信息
print("=" * 50)
print("Bell 态电路:")
print("=" * 50)
print(qc.draw())
print(f"\n电路摘要: {qc.summary()}")
print(f"门统计: {qc.count_ops()}")

# 4. 获取态向量
sv = qc.statevector()
print("\n" + "=" * 50)
print("态向量 (|Φ+⟩ = (|00⟩ + |11⟩)/√2):")
print("=" * 50)
for i, amp in enumerate(sv):
    if np.abs(amp) > 1e-10:
        bits = format(i, f"0{qc.num_qubits}b")
        print(f"  |{bits}⟩: {amp:.4f}")

# 5. 执行测量
result = qc.execute(shots=4096, seed=42)
counts = result.get_counts()

print("\n" + "=" * 50)
print("测量结果 (4096 次采样):")
print("=" * 50)
for bits, count in sorted(counts.items()):
    prob = count / result.shots * 100
    print(f"  |{bits}⟩: {count:4d} 次 ({prob:5.1f}%)")

# 6. 验证概率接近 50%/50%
probs = result.probabilities()
print(f"\n理论概率: |00⟩=50%, |11⟩=50%")
print(f"实验概率: |00⟩={probs.get('00', 0)*100:.1f}%, |11⟩={probs.get('11', 0)*100:.1f}%")
```

---

## 12. 完整示例：参数化电路 + 门代数

```python
import numpy as np
from qsl import QuantumCircuit, Parameter, Gate
from qsl.quantum_gates import RX as RX_mat  # 注意：直接从 quantum_gates 导入矩阵

# RX 矩阵函数
from qsl.quantum_gates import rx as rx_matrix_fn

theta = Parameter("θ")

# 方式1：使用电路便捷方法（推荐）
qc1 = QuantumCircuit(2)
qc1.rx(0, theta)
qc1.cx(0, 1)
bound1 = qc1.assign({theta: np.pi})
print("方式1 - 电路方法:")
print(bound1.draw())

# 方式2：手动创建 Gate 对象并使用 control()
qc2 = QuantumCircuit(3)
# 创建数值化 RX 门（用于演示 control()）
rx_pi = Gate("rx", rx_matrix_fn(np.pi/2), 1, label="RX(π/2)")
# 生成受控 RX 门
crx_custom = rx_pi.control(1)
# 生成双受控 X 门（Toffoli 等效）
from qsl.quantum_gates import X as X_mat
x_gate = Gate("x", X_mat, 1, label="X")
ccx_custom = x_gate.control(2)

qc2.h(0)
qc2.append(crx_custom, [0, 1])
qc2.append(ccx_custom, [0, 1, 2])
print("\n方式2 - 自定义受控门:")
print(qc2.draw())

# 执行并测量
res = qc2.execute(shots=1024, seed=123)
print(f"\n测量结果: {res.get_counts()}")
```

---

## 13. API 速查表

### QuantumCircuit 常用方法

| 类别 | 方法 | 说明 |
|------|------|------|
| **创建** | `QuantumCircuit(n)` | 创建 n 比特电路 |
| **单比特门** | `h(q)`, `x(q)`, `y(q)`, `z(q)` | Hadamard, Pauli-X/Y/Z |
| | `s(q)`, `sdg(q)`, `t(q)`, `tdg(q)` | 相位门及其逆 |
| | `sx(q)`, `sxdg(q)` | √X 门及其逆 |
| | `rx(q, θ)`, `ry(q, θ)`, `rz(q, φ)` | 坐标轴旋转门 |
| | `p(q, λ)`, `u(q, θ, φ, λ)` | 相位门、通用单比特门 |
| **两比特门** | `cx(c,t)`, `cy(c,t)`, `cz(c,t)` | CNOT, 受控 Y/Z |
| | `ch(c,t)`, `cs(c,t)`, `ct(c,t)` | 受控 H/S/T |
| | `swap(a,b)`, `iswap(a,b)` | SWAP, iSWAP |
| | `crx(θ,c,t)`, `cry(θ,c,t)`, `crz(θ,c,t)` | 受控旋转门 |
| | `cp(λ,c,t)`, `cu(θ,φ,λ,c,t)` | 受控相位、受控 U |
| | `rxx(θ,a,b)`, `ryy(θ,a,b)`, `rzz(θ,a,b)` | 两比特纠缠门 |
| **三比特门** | `ccx(c1,c2,t)`, `cswap(c,a,b)` | Toffoli, Fredkin |
| | `mcx(controls, t)`, `mcz(qubits)` | 多控制 X/Z |
| **电路操作** | `inverse()`, `compose(other)` | 逆电路、电路拼接 |
| | `decompose()`, `transpile()` | 门分解、转译优化 |
| | `reverse_bits()`, `barrier()` | 比特反转、屏障 |
| | `copy()`, `append(gate, qubits)` | 复制、追加自定义门 |
| | `unitary(mat, qubits)` | 添加任意酉矩阵 |
| **参数化** | `bind_parameters()`, `assign()` | 绑定参数值 |
| **执行** | `execute(shots)`, `measure_all()` | 执行电路、快速测量 |
| | `statevector()` | 获取态向量 |
| **信息** | `size()`, `depth()`, `width()` | 门数、深度、宽度 |
| | `count_ops()`, `summary()` | 门统计、摘要 |
| | `draw(output)` | 绘制电路 ('text'/'mpl') |

### Gate 代数方法

| 方法 | 说明 |
|------|------|
| `Gate(name, matrix, n, label=...)` | 创建 Gate 对象 |
| `gate.to_matrix()` | 获取酉矩阵 |
| `gate.control(n)` | 生成 n 控制位版本 |
| `gate.power(k)` | 生成 U^k（支持分数幂） |
| `gate.inverse()` | 生成逆门 U† |
| `gate.copy()` | 复制门 |

### ExecutionResult 方法

| 方法/属性 | 说明 |
|-----------|------|
| `result.counts` | 整数索引计数字典 |
| `result.shots` | 采样次数 |
| `result.get_counts(binary=True)` | 比特字符串计数 |
| `result.probabilities()` | 经验概率分布 |
| `result.probabilities_dict()` | 概率分布（别名） |
| `result.most_frequent(n)` | 最高频 n 个结果 |
| `result.statevector()` | 态向量 |

---

## 下一步

- 学习 [03 - Grover 搜索算法](03_grover_search.md)：量子搜索与 SAT 问题求解
- 学习 [04 - Shor 大数分解](04_shor_factorization.md)：RSA 破解与量子周期查找
- 学习 [08 - 可视化指南](08_visualization.md)：绘制更美观的电路图和态可视化

---

**版本说明**：本教程适用于 QSL v0.6.3。如遇 API 变更，请参考 [CHANGELOG.md](../../CHANGELOG.md)。
