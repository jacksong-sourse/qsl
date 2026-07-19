"""
标准量子门矩阵模块。

所有函数返回 numpy ndarray，形状为 (2^n, 2^n)，其中 n 是门作用的量子比特数。
本模块提供构建量子电路所需的全部基础门矩阵、旋转门和受控门。

数学约定:
    - 单量子比特门作用于最低位量子比特
    - 多量子比特门的矩阵索引使用标准基态排序:
      |q0 q1 ... qn> 对应十进制索引 sum(q_i * 2^(n-1-i))
    - 相位约定: Rz(φ) = exp(-i*φ/2) |0><0| + exp(i*φ/2) |1><1|

失败模式分析:
    1. 参数类型错误: numpy 矩阵运算会自然抛出异常
    2. n_controls 过大: 指数级内存增长，2^(n_controls+1) 可能超出可用内存
    3. 角度参数为 NaN 或 Inf: 矩阵元素变为 NaN/Inf
    4. 浮点精度: 三角函数使用 numpy 内置，精度与平台相关
"""

import numpy as np
from functools import reduce

# ====================================================================
# 基础矩阵常量
# ====================================================================

I = np.eye(2)                          # 恒等矩阵
X = np.array([[0, 1], [1, 0]])         # Pauli-X (NOT 门)
Y = np.array([[0, -1j], [1j, 0]])      # Pauli-Y
Z = np.array([[1, 0], [0, -1]])        # Pauli-Z (相位翻转)
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)   # Hadamard 门
S = np.array([[1, 0], [0, 1j]])         # 相位门 (sqrt(Z))
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])  # T 门 (pi/8 门)


# ====================================================================
# 旋转门
# ====================================================================

def rx(theta):
    """
    X 轴旋转门。

    矩阵:
        [[ cos(θ/2),  -i*sin(θ/2)],
         [-i*sin(θ/2),   cos(θ/2) ]]

    物理意义: 绕 Bloch 球的 X 轴旋转角度 theta。

    参数:
        theta: 旋转角度 (弧度)

    返回:
        numpy ndarray, shape (2, 2)
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta):
    """
    Y 轴旋转门。

    矩阵:
        [[cos(θ/2), -sin(θ/2)],
         [sin(θ/2),  cos(θ/2)]]

    物理意义: 绕 Bloch 球的 Y 轴旋转角度 theta。

    参数:
        theta: 旋转角度 (弧度)

    返回:
        numpy ndarray, shape (2, 2)
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(phi):
    """
    Z 轴旋转门。

    矩阵:
        [[exp(-i*φ/2), 0          ],
         [0,           exp(i*φ/2) ]]

    物理意义: 绕 Bloch 球的 Z 轴旋转角度 phi。

    参数:
        phi: 旋转角度 (弧度)

    返回:
        numpy ndarray, shape (2, 2)
    """
    return np.array(
        [[np.exp(-1j * phi / 2), 0],
         [0, np.exp(1j * phi / 2)]],
        dtype=complex,
    )


# ====================================================================
# 通用 U3 门
# ====================================================================

def u3(theta, phi, lam):
    """
    通用 U3 门 (任意单量子比特酉变换)。

    U3(θ, φ, λ) = Rz(φ) * Ry(θ) * Rz(λ)

    矩阵:
        [[      cos(θ/2),          -exp(iλ)*sin(θ/2)],
         [exp(iφ)*sin(θ/2),  exp(i(φ+λ))*cos(θ/2)   ]]

    这是 IBM Qiskit 中 U3 门的标准定义，可以表示任意单量子比特酉操作。

    参数:
        theta: 极角 (弧度)
        phi:   方位角 (弧度)
        lam:   Z 旋转角 (弧度)

    返回:
        numpy ndarray, shape (2, 2)
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[c, -np.exp(1j * lam) * s],
         [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]],
        dtype=complex,
    )


# ====================================================================
# 多量子比特门
# ====================================================================

def swap():
    """
    SWAP 门 (2 量子比特)。

    矩阵:
        [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]]

    作用: 交换两个量子比特的状态。

    返回:
        numpy ndarray, shape (4, 4)
    """
    return np.array(
        [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]],
        dtype=complex,
    )


def cswap():
    """
    Fredkin 门 / 受控 SWAP 门 (CSWAP, 3 量子比特)。

    矩阵: 8x8 单位矩阵，其中子块
        |110> <-> |101>
        |101> <-> |110>
    被交换 (即第一个量子比特为 |1> 时交换后两个量子比特)。

    矩阵索引按 |q0 q1 q2> 排序，q0 为控制位。

    返回:
        numpy ndarray, shape (8, 8)
    """
    gate = np.eye(8, dtype=complex)
    # |101> <-> |110>: 索引 5 (101) 和 6 (110)
    gate[5, 5] = 0
    gate[5, 6] = 1
    gate[6, 5] = 1
    gate[6, 6] = 0
    return gate


def mcx(n_controls):
    """
    多控制 X 门 (MCX) / 广义 Toffoli 门。

    作用: 当所有控制量子比特为 |1> 时，翻转目标 (最后一个) 量子比特。
    矩阵大小为 2^(n_controls+1) x 2^(n_controls+1)。

    构造方法:
        单位矩阵，将 |11...10> 和 |11...11> 两个对角元素互换。

    参数:
        n_controls: 控制量子比特的数量

    返回:
        numpy ndarray, shape (2^(n_controls+1), 2^(n_controls+1))

    失败模式:
        - n_controls < 1: 返回 Pauli-X 门 (无控制位，等效于 X)
        - n_controls 过大: 内存不足异常

    示例:
        mcx(1) -> CNOT:  [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
        mcx(2) -> Toffoli: 将 |110> 和 |111> 互换
    """
    n_qubits = n_controls + 1
    dim = 1 << n_qubits
    gate = np.eye(dim, dtype=complex)

    # 所有控制位为 |1> 且目标位为 |0> 的基态索引:
    #   控制位全 1 对应的掩码 = (1 << n_controls) - 1，放在高位
    #   目标位在最低位 (第 0 位)
    #   所以控制位全 1 的索引 = (控制位掩码 << 1)
    control_mask = ((1 << n_controls) - 1) << 1
    # |c...c, 0>: 控制位全 1，目标位 0
    idx_0 = control_mask  # target 位为 0
    # |c...c, 1>: 控制位全 1，目标位 1
    idx_1 = control_mask | 1  # target 位为 1

    gate[idx_0, idx_0] = 0
    gate[idx_0, idx_1] = 1
    gate[idx_1, idx_0] = 1
    gate[idx_1, idx_1] = 0
    return gate


# ====================================================================
# Kronecker 积与受控门
# ====================================================================

def kronecker_prod(*matrices):
    """
    计算多个矩阵的 Kronecker 积。

    kronecker_prod(A, B, C) = A ⊗ B ⊗ C

    使用 functools.reduce 和 np.kron 从左到右累积计算。
    Kronecker 积用于组合作用于不同量子比特的门，
    生成完整希尔伯特空间上的矩阵。

    参数:
        *matrices: 可变数量的 numpy ndarray

    返回:
        numpy ndarray，形状为各矩阵维度的乘积

    示例:
        kronecker_prod(H, I) -> H ⊗ I  (4x4 矩阵，H 作用于第 0 量子比特)
    """
    if not matrices:
        return np.array([[1]], dtype=complex)
    return reduce(np.kron, matrices)


def kron(*matrices):
    """
    kronecker_prod 的向后兼容别名 (已弃用)。

    该名称遮蔽了 numpy.kron, 新代码请使用 kronecker_prod
    或直接调用 np.kron。
    """
    import warnings
    warnings.warn(
        "qsl.quantum_gates.kron 已弃用 (与 numpy.kron 命名冲突), "
        "请改用 kronecker_prod 或 np.kron。",
        DeprecationWarning, stacklevel=2,
    )
    return kronecker_prod(*matrices)


def controlled_gate(gate, n_controls):
    """
    将单量子比特门嵌入为受控版本。

    在 n_controls 个控制量子比特全部为 |1> 时，
    将 gate 作用于目标 (最后一个) 量子比特。

    数学定义:
        C^n(G) = I^(2^n - 2) ⊕ G

    即矩阵为分块对角的，除了右下角 2x2 子块为 gate 本身。

    参数:
        gate:       单量子比特门矩阵 (2x2 numpy ndarray)
        n_controls: 控制量子比特数

    返回:
        完整 n_qubits = n_controls + 1 的酉矩阵，
        shape (2^(n_controls+1), 2^(n_controls+1))

    失败模式:
        - gate 形状不是 (2, 2): 引发异常
        - n_controls < 0: 索引错误
        - n_controls == 0: 直接返回 gate 的副本
    """
    gate = np.asarray(gate, dtype=complex)
    if gate.shape != (2, 2):
        raise ValueError(f"gate 必须是 2x2 矩阵，实际形状为 {gate.shape}")

    if n_controls == 0:
        return gate.copy()

    n_qubits = n_controls + 1
    dim = 1 << n_qubits
    result = np.eye(dim, dtype=complex)

    # 控制位全 1 + 目标位任意 => 子矩阵索引:
    #   高位 n_controls 位全 1，最低位为目标位
    control_mask = ((1 << n_controls) - 1) << 1
    idx_0 = control_mask          # |11...10>
    idx_1 = control_mask | 1       # |11...11>

    result[idx_0, idx_0] = gate[0, 0]
    result[idx_0, idx_1] = gate[0, 1]
    result[idx_1, idx_0] = gate[1, 0]
    result[idx_1, idx_1] = gate[1, 1]
    return result


# ====================================================================
# 衍生单比特门 (sqrt 系列 / 共轭转置 / 相位门)
# ====================================================================

Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)          # S†
Tdg = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)  # T†
SX = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex) / 2  # sqrt(X)
SXdg = SX.conj().T                                          # sqrt(X)†


def p(lam):
    """
    相位门 P(λ) = diag(1, e^{iλ})。

    与 Rz 的区别: P(λ) 不含全局相位, T = P(π/4), S = P(π/2), Z = P(π)。

    参数:
        lam: 相位角 (弧度)
    返回:
        numpy ndarray, shape (2, 2)
    """
    return np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=complex)


def u(theta, phi, lam):
    """U 门 (u3 别名, 与 Qiskit UGate 一致)。"""
    return u3(theta, phi, lam)


# ====================================================================
# 受控旋转门 / 受控通用门
# ====================================================================

def crx(theta):
    """受控 RX 门 (控制位在高位)。"""
    return controlled_gate(rx(theta), 1)


def cry(theta):
    """受控 RY 门 (控制位在高位)。"""
    return controlled_gate(ry(theta), 1)


def crz(theta):
    """受控 RZ 门 (控制位在高位)。"""
    return controlled_gate(rz(theta), 1)


def cp(lam):
    """受控相位门 CP(λ): 仅 |11> 获得相位 e^{iλ}。"""
    gate = np.eye(4, dtype=complex)
    gate[3, 3] = np.exp(1j * lam)
    return gate


def cu(theta, phi, lam, gamma=0.0):
    """
    受控 U 门 (控制位高位, 目标位低位), 带全局相位 gamma:
        CU = |0><0|⊗I + |1><1|⊗e^{iγ}U(θ,φ,λ)
    """
    gate = np.eye(4, dtype=complex)
    gate[2:, 2:] = np.exp(1j * gamma) * u3(theta, phi, lam)
    return gate


def ch():
    """受控 Hadamard 门 (控制位高位)。"""
    return controlled_gate(H, 1)


def cs():
    """受控 S 门 (控制位高位)。"""
    return controlled_gate(S, 1)


def csdg():
    """受控 S† 门 (控制位高位)。"""
    return controlled_gate(Sdg, 1)


def ct():
    """受控 T 门 (控制位高位)。"""
    return controlled_gate(T, 1)


def ctdg():
    """受控 T† 门 (控制位高位)。"""
    return controlled_gate(Tdg, 1)


def dcx():
    """双 CNOT 门 DCX = CNOT(q0,q1)·CNOT(q1,q0)。"""
    gate = np.zeros((4, 4), dtype=complex)
    gate[0, 0] = 1
    gate[1, 3] = 1
    gate[2, 1] = 1
    gate[3, 2] = 1
    return gate


# ====================================================================
# 两比特纠缠门
# ====================================================================

def iswap():
    """
    iSWAP 门: 交换两比特并附加相位 i。
        |01> -> i|10>, |10> -> i|01>
    """
    gate = np.eye(4, dtype=complex)
    gate[1, 1] = 0
    gate[2, 2] = 0
    gate[1, 2] = 1j
    gate[2, 1] = 1j
    return gate


def rxx(theta):
    """
    RXX(θ) = exp(-i θ/2 X⊗X)。
    矩阵基序 |q1 q0> (q1 高位)。
    """
    c = np.cos(theta / 2)
    s = -1j * np.sin(theta / 2)
    xx = np.kron(X, X)
    return c * np.eye(4, dtype=complex) + s * xx


def ryy(theta):
    """RYY(θ) = exp(-i θ/2 Y⊗Y)。"""
    c = np.cos(theta / 2)
    s = -1j * np.sin(theta / 2)
    yy = np.kron(Y, Y)
    return c * np.eye(4, dtype=complex) + s * yy


def rzz(theta):
    """RZZ(θ) = exp(-i θ/2 Z⊗Z)。"""
    return np.diag(np.exp(-1j * theta / 2 * np.array([1, -1, -1, 1])))


def ecr():
    """
    ECR 门 (echoed cross-resonance), 与 Qiskit ECRGate 一致:
        ECR = 1/√2 [[0,1,0,i],[1,0,-i,0],[0,i,0,1],[-i,0,1,0]]
    """
    inv = 1.0 / np.sqrt(2)
    return inv * np.array(
        [[0, 1, 0, 1j],
         [1, 0, -1j, 0],
         [0, 1j, 0, 1],
         [-1j, 0, 1, 0]],
        dtype=complex,
    )


def xx_plus_yy(theta, beta=0.0):
    """XX+YY 门 (Qiskit XXPlusYYGate)。"""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    gate = np.eye(4, dtype=complex)
    gate[1, 1] = c
    gate[2, 2] = c
    gate[1, 2] = -1j * s * np.exp(-1j * beta)
    gate[2, 1] = -1j * s * np.exp(1j * beta)
    return gate


# ====================================================================
# 导出列表
# ====================================================================

__all__ = [
    # 基础矩阵
    "I", "X", "Y", "Z", "H", "S", "T",
    # 衍生单比特门
    "Sdg", "Tdg", "SX", "SXdg", "p",
    # 旋转门
    "rx", "ry", "rz",
    # 通用门
    "u3", "u",
    # 多量子比特门
    "swap", "cswap", "mcx",
    # 受控门
    "crx", "cry", "crz", "cp", "cu", "ch",
    "cs", "csdg", "ct", "ctdg", "dcx",
    # 两比特纠缠门
    "iswap", "rxx", "ryy", "rzz", "ecr", "xx_plus_yy",
    # 工具函数
    "kronecker_prod", "kron", "controlled_gate",
]
