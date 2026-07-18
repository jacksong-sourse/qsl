"""
量子态向量 (Quantum State Vector) 模块。

数学基础:
    |psi> = sum_{i=0}^{2^n-1} alpha_i |i>
    其中 alpha_i 为复振幅，sum |alpha_i|^2 = 1

模拟器通过经典数组存储 2^n 个复振幅来表示 n 量子比特的纯态。
这是所有计算的基础数据结构。

失败模式分析:
    1. 量子比特数 < 1: 无物理意义
    2. 量子比特数 > 20: 2^20 = 1,048,576 个复数 ≈ 16MB，超过则内存爆炸
    3. 系统量子比特索引越界: 门操作目标不存在
    4. 浮点精度累积: 多次门操作后归一化可能漂移
    5. 概率计算下溢: |alpha|^2 可能小于浮点最小正数
"""

import math
import cmath
import random
import numpy as np
from typing import List, Tuple, Set, Optional

from ..utils.validation import (
    validate_n_qubits,
    validate_qubit_index,
    validate_qubit_indices,
)
from ..utils.exceptions import (
    StateNormalizationError,
    DuplicateQubitError,
)


# 全局最大量子比特数限制
MAX_QUBITS = 20


class QuantumState:
    """
    n 量子比特纯态的向量表示。

    内部使用 Python list 存储 2^n 个复振幅。
    初始态为 |0...0>，即 amplitudes[0] = 1+0j，其余为 0。

    属性:
        n_qubits: 量子比特数
        size: 希尔伯特空间维度 (= 2^n)
        amplitudes: 复振幅列表，amplitudes[i] 对应基态 |i> 的振幅
    """

    def __init__(self, n_qubits: int):
        """
        初始化 n 量子比特的 |0...0> 态。

        参数:
            n_qubits: 量子比特数 (1 <= n_qubits <= 20)

        失败模式:
            - n_qubits < 1: 抛出 InvalidQubitCountError
            - n_qubits > 20: 抛出 InvalidQubitCountError
        """
        validate_n_qubits(n_qubits, MAX_QUBITS)
        self._n = n_qubits
        self._N = 1 << n_qubits
        self.amplitudes = np.zeros(self._N, dtype=complex)
        self.amplitudes[0] = 1.0 + 0j
        self._gate_history: list = []
        self._readout_error: float = 0.0

    # ----------------------------------------------------------------
    # 属性访问
    # ----------------------------------------------------------------

    @property
    def n_qubits(self) -> int:
        return self._n

    @property
    def size(self) -> int:
        return self._N

    # ----------------------------------------------------------------
    # 内部工具方法
    # ----------------------------------------------------------------

    @staticmethod
    def _bit(value: int, position: int) -> int:
        """提取 value 的第 position 位 (0-indexed, LSB 为第 0 位)。"""
        return (value >> position) & 1

    # ----------------------------------------------------------------
    # 单量子比特门
    # ----------------------------------------------------------------

    def x(self, target: int):
        """
        Pauli-X 门 (量子 NOT 门)。

        X|0> = |1>, X|1> = |0>
        矩阵: [[0, 1], [1, 0]]

        实现: 交换 target 位为 0 和 target 位为 1 的振幅对。
        时间复杂度: O(N)

        失败模式:
            - target 越界: 抛出 QubitIndexError
        """
        validate_qubit_index(target, self._n)
        mask = 1 << target
        indices = np.arange(self._N)
        target_bit_zero = (indices & mask) == 0
        target_bit_one = ~target_bit_zero
        
        a_zero = self.amplitudes[target_bit_zero].copy()
        a_one = self.amplitudes[target_bit_one].copy()
        
        self.amplitudes[target_bit_zero] = a_one
        self.amplitudes[target_bit_one] = a_zero

    def y(self, target: int):
        """
        Pauli-Y 门。

        Y|0> = i|1>, Y|1> = -i|0>
        矩阵: [[0, -i], [i, 0]]

        实现:
            alpha'_0 = -i * alpha_1
            alpha'_1 =  i * alpha_0

        失败模式:
            - target 越界: 抛出 QubitIndexError
        """
        validate_qubit_index(target, self._n)
        mask = 1 << target
        indices = np.arange(self._N)
        target_bit_zero = (indices & mask) == 0
        target_bit_one = ~target_bit_zero
        
        a_zero = self.amplitudes[target_bit_zero].copy()
        a_one = self.amplitudes[target_bit_one].copy()
        
        self.amplitudes[target_bit_zero] = -1j * a_one
        self.amplitudes[target_bit_one] = 1j * a_zero

    def z(self, target: int):
        """
        Pauli-Z 门 (相位翻转门)。

        Z|0> = |0>, Z|1> = -|1>
        矩阵: [[1, 0], [0, -1]]

        实现: target 位为 1 的振幅乘以 -1。
        时间复杂度: O(N)

        失败模式:
            - target 越界: 抛出 QubitIndexError
        """
        validate_qubit_index(target, self._n)
        mask = 1 << target
        indices = np.arange(self._N)
        target_bit_one = (indices & mask) != 0
        self.amplitudes[target_bit_one] *= -1

    def h(self, target: int):
        """
        Hadamard 门。

        H|0> = (|0> + |1>) / sqrt(2)
        H|1> = (|0> - |1>) / sqrt(2)
        矩阵: [[1, 1], [1, -1]] / sqrt(2)

        实现:
            alpha'_i0 = (alpha_i0 + alpha_i1) / sqrt(2)
            alpha'_i1 = (alpha_i0 - alpha_i1) / sqrt(2)

        时间复杂度: O(N)

        失败模式:
            - target 越界: 抛出 QubitIndexError
        """
        validate_qubit_index(target, self._n)
        mask = 1 << target
        inv_sqrt2 = 1.0 / math.sqrt(2)
        indices = np.arange(self._N)
        target_bit_zero = (indices & mask) == 0
        target_bit_one = ~target_bit_zero
        
        a_zero = self.amplitudes[target_bit_zero].copy()
        a_one = self.amplitudes[target_bit_one].copy()
        
        self.amplitudes[target_bit_zero] = (a_zero + a_one) * inv_sqrt2
        self.amplitudes[target_bit_one] = (a_zero - a_one) * inv_sqrt2

    def s(self, target: int):
        """
        S 门 (相位门, sqrt(Z))。

        S|0> = |0>, S|1> = i|1>
        矩阵: [[1, 0], [0, i]]

        实现: target 位为 1 的振幅乘以 i。

        失败模式:
            - target 越界: 抛出 QubitIndexError
        """
        validate_qubit_index(target, self._n)
        mask = 1 << target
        indices = np.arange(self._N)
        target_bit_one = (indices & mask) != 0
        self.amplitudes[target_bit_one] *= 1j

    def t(self, target: int):
        """
        T 门 (pi/8 门, sqrt(S))。

        T|0> = |0>, T|1> = exp(i*pi/4) |1>
        矩阵: [[1, 0], [0, exp(i*pi/4)]]

        失败模式:
            - target 越界: 抛出 QubitIndexError
        """
        validate_qubit_index(target, self._n)
        mask = 1 << target
        phase = cmath.exp(1j * math.pi / 4)
        indices = np.arange(self._N)
        target_bit_one = (indices & mask) != 0
        self.amplitudes[target_bit_one] *= phase

    def rx(self, target: int, theta: float):
        """
        Rotation-X gate: exp(-i * theta/2 * X).
        
        RX(theta) = [[cos(theta/2), -i*sin(theta/2)],
                      [-i*sin(theta/2), cos(theta/2)]]
        """
        validate_qubit_index(target, self._n)
        c = math.cos(theta / 2)
        s = -1j * math.sin(theta / 2)
        mask = 1 << target
        indices = np.arange(self._N)
        target_bit_zero = (indices & mask) == 0
        target_bit_one = ~target_bit_zero
        
        a_zero = self.amplitudes[target_bit_zero].copy()
        a_one = self.amplitudes[target_bit_one].copy()
        
        self.amplitudes[target_bit_zero] = c * a_zero + s * a_one
        self.amplitudes[target_bit_one] = s * a_zero + c * a_one

    def ry(self, target: int, theta: float):
        """
        Rotation-Y gate: exp(-i * theta/2 * Y).
        
        RY(theta) = [[cos(theta/2), -sin(theta/2)],
                      [sin(theta/2), cos(theta/2)]]
        """
        validate_qubit_index(target, self._n)
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        mask = 1 << target
        indices = np.arange(self._N)
        target_bit_zero = (indices & mask) == 0
        target_bit_one = ~target_bit_zero
        
        a_zero = self.amplitudes[target_bit_zero].copy()
        a_one = self.amplitudes[target_bit_one].copy()
        
        self.amplitudes[target_bit_zero] = c * a_zero - s * a_one
        self.amplitudes[target_bit_one] = s * a_zero + c * a_one

    def rz(self, target: int, phi: float):
        """
        Rotation-Z gate: exp(-i * phi/2 * Z).
        
        RZ(phi) = [[exp(-i*phi/2), 0],
                    [0, exp(i*phi/2)]]
        """
        validate_qubit_index(target, self._n)
        mask = 1 << target
        indices = np.arange(self._N)
        target_bit_zero = (indices & mask) == 0
        target_bit_one = ~target_bit_zero
        
        self.amplitudes[target_bit_zero] *= cmath.exp(-1j * phi / 2)
        self.amplitudes[target_bit_one] *= cmath.exp(1j * phi / 2)

    # ----------------------------------------------------------------
    # 两量子比特门
    # ----------------------------------------------------------------

    def cnot(self, control: int, target: int):
        """
        受控 NOT 门 (CNOT, CX)。

        CNOT|c,t> = |c, c XOR t>
        即: 当 control = |1> 时翻转 target。

        矩阵:
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0]]

        失败模式:
            - control 或 target 越界: 抛出 QubitIndexError
            - control == target: 物理上无意义，抛出 DuplicateQubitError

        时间复杂度: O(N)
        """
        validate_qubit_index(control, self._n)
        validate_qubit_index(target, self._n)
        if control == target:
            raise DuplicateQubitError([control, target])
        c_mask = 1 << control
        t_mask = 1 << target
        indices = np.arange(self._N)
        c_one_t_zero = ((indices & c_mask) != 0) & ((indices & t_mask) == 0)
        c_one_t_one = ((indices & c_mask) != 0) & ((indices & t_mask) != 0)
        
        a_t0 = self.amplitudes[c_one_t_zero].copy()
        a_t1 = self.amplitudes[c_one_t_one].copy()
        
        self.amplitudes[c_one_t_zero] = a_t1
        self.amplitudes[c_one_t_one] = a_t0

    def cz(self, control: int, target: int):
        """
        受控 Z 门。

        CZ|c,t> = (-1)^{c AND t} |c,t>
        即: 当 control=1 且 target=1 时相位翻转。

        失败模式:
            - control 或 target 越界: 抛出 QubitIndexError
            - control == target: 抛出异常

        时间复杂度: O(N)
        """
        validate_qubit_index(control, self._n)
        validate_qubit_index(target, self._n)
        if control == target:
            raise DuplicateQubitError([control, target])
        c_mask = 1 << control
        t_mask = 1 << target
        both_mask = c_mask | t_mask
        indices = np.arange(self._N)
        both_one = (indices & both_mask) == both_mask
        self.amplitudes[both_one] *= -1

    def swap(self, q0: int, q1: int):
        """
        SWAP 门。

        SWAP|a,b> = |b,a>

        实现: 交换 q0 和 q1 不同时的振幅对。

        失败模式:
            - q0 或 q1 越界: 抛出 QubitIndexError
            - q0 == q1: 无操作 (退化情况，允许但 warn)

        时间复杂度: O(N)
        """
        validate_qubit_index(q0, self._n)
        validate_qubit_index(q1, self._n)
        if q0 == q1:
            return
        mask0 = 1 << q0
        mask1 = 1 << q1
        indices = np.arange(self._N)
        b0 = (indices & mask0) != 0
        b1 = (indices & mask1) != 0
        
        diff = b0 != b1
        a_diff = self.amplitudes[diff].copy()
        
        swapped_indices = indices[diff] ^ mask0 ^ mask1
        self.amplitudes[swapped_indices] = a_diff

    # ----------------------------------------------------------------
    # 三量子比特门
    # ----------------------------------------------------------------

    def toffoli(self, c1: int, c2: int, target: int):
        """
        Toffoli 门 (CCNOT, 双控制非门)。

        Toffoli|a,b,c> = |a, b, c XOR (a AND b)>
        即: 当两个控制位都为 |1> 时翻转目标位。

        Toffoli 门是通用可逆门 —— 任意经典布尔函数都可以
        用 Toffoli 门加辅助比特实现。

        失败模式:
            - 任意索引越界: 抛出 QubitIndexError
            - c1, c2, target 中有重复: 抛出异常
            - c1 == c2 == target 或其他组合重复

        时间复杂度: O(N)
        """
        validate_qubit_index(c1, self._n)
        validate_qubit_index(c2, self._n)
        validate_qubit_index(target, self._n)
        if len({c1, c2, target}) < 3:
            raise DuplicateQubitError([c1, c2, target])
        c1_mask = 1 << c1
        c2_mask = 1 << c2
        t_mask = 1 << target
        both_mask = c1_mask | c2_mask
        indices = np.arange(self._N)
        c_both_one = (indices & both_mask) == both_mask
        t_zero = (indices & t_mask) == 0
        t_one = ~t_zero
        
        swap_mask = c_both_one & t_zero
        swap_indices = indices[swap_mask]
        
        a_t0 = self.amplitudes[swap_indices].copy()
        a_t1 = self.amplitudes[swap_indices ^ t_mask].copy()
        
        self.amplitudes[swap_indices] = a_t1
        self.amplitudes[swap_indices ^ t_mask] = a_t0

    # ----------------------------------------------------------------
    # 多量子比特门
    # ----------------------------------------------------------------

    def mcz(self, qubits: List[int]):
        """
        多控制 Z 门。

        MCZ|q_1...q_k> = (-1)^{q_1 AND ... AND q_k} |q_1...q_k>
        即: 当所有指定的量子比特都为 |1> 时相位翻转。

        这是 Grover 扩散算子的核心组件:
            D = H^[n] X^[n] MCZ X^[n] H^[n]

        失败模式:
            - qubits 中有索引越界: 抛出 QubitIndexError
            - qubits 中有重复索引: 抛出 DuplicateQubitError
            - qubits 为空: 抛出 ValueError

        时间复杂度: O(N)
        """
        validate_qubit_indices(qubits, self._n)
        mask = 0
        for q in qubits:
            mask |= (1 << q)
        indices = np.arange(self._N)
        all_one = (indices & mask) == mask
        self.amplitudes[all_one] *= -1

    # ----------------------------------------------------------------
    # Oracle 相关
    # ----------------------------------------------------------------

    def phase_oracle(self, marked: Set[int]):
        """
        相位 Oracle: 对 marked 中的基态施加 -1 相位。

        Oracle|x> = (-1)^{f(x)} |x>
        其中 f(x) = 1 当 x in marked，否则 0

        失败模式:
            - marked 中的索引越界: 静默跳过越界的索引
            - marked 为空: 无操作 (合法，但无意义)
            - marked 过大: 等于 N 时所有态都翻转，相当于全局相位 -1

        时间复杂度: O(|marked|)
        """
        if not marked:
            return
        for x in marked:
            if 0 <= x < self._N:
                self.amplitudes[x] = -self.amplitudes[x]
            else:
                import warnings
                warnings.warn(
                    f"phase_oracle: marked index {x} is out of bounds "
                    f"[0, {self._N - 1}], skipping.",
                    RuntimeWarning, stacklevel=2
                )

    # ----------------------------------------------------------------
    # 扩散算子 (Grover 专用)
    # ----------------------------------------------------------------

    def diffusion_operator(self):
        """
        Grover 扩散算子。

        D = H^[n] X^[n] MCZ X^[n] H^[n]

        关于均匀叠加态 |psi_0> 的反射:
            D = 2|psi_0><psi_0| - I

        作用: 将振幅关于 |psi_0> 做镜像反射。

        失败模式:
            - n_qubits = 0: 已在初始化时被阻止
        """
        all_qubits = list(range(self._n))
        for q in all_qubits:
            self.h(q)
            self.x(q)
        self.mcz(all_qubits)
        for q in all_qubits:
            self.x(q)
            self.h(q)

    # ----------------------------------------------------------------
    # 测量
    # ----------------------------------------------------------------

    def probability(self, x: int) -> float:
        """
        返回测量得到 |x> 的概率 P(x) = |alpha_x|^2。

        失败模式:
            - x 越界: 返回 0.0 而非报错 (处理边界情况)
            - 振幅 NaN: 概率会是 NaN，但这里不做检查以保持性能
        """
        if x < 0 or x >= self._N:
            return 0.0
        amp = self.amplitudes[x]
        return (amp.real * amp.real + amp.imag * amp.imag)

    def probabilities(self) -> List[float]:
        """返回所有基态的概率分布。"""
        return [self.probability(i) for i in range(self._N)]

    def measure(self, collapse: bool = False) -> Tuple[int, float]:
        """
        按玻恩定则随机采样一个基态。

        数学: P(result = i) = |<i|psi>|^2 = |alpha_i|^2

        参数:
            collapse: 如果为 True，测量后态坍缩到测量结果

        返回: (测量结果整数, 对应概率)

        失败模式:
            - 浮点累积误差: 使用 cumulative 遍历保证正确采样
            - 所有概率为零: 返回最后一个状态 (容错)
        """
        r = random.random()
        cumulative = 0.0
        probs = np.array([self.probability(i) for i in range(self._N)])
        prob_sum = np.sum(probs)
        if prob_sum > 0:
            probs = probs / prob_sum
        for i in range(self._N):
            cumulative += probs[i]
            if r < cumulative:
                original_p = probs[i]
                result = i
                if self._readout_error > 0:
                    import random as _random
                    for bit in range(self._n):
                        if _random.random() < self._readout_error:
                            result ^= (1 << bit)
                if collapse:
                    self._collapse_to(result)
                return result, original_p
        last = self._N - 1
        original_p = self.probability(last)
        if collapse:
            self._collapse_to(last)
        return last, original_p

    def _collapse_to(self, index: int):
        """坍缩量子态到指定的基态 |index>。"""
        for i in range(self._N):
            self.amplitudes[i] = 0j
        self.amplitudes[index] = 1.0 + 0j

    def measure_most_likely(self) -> Tuple[int, float]:
        """
        返回概率最大的测量结果。

        返回: (最可能的基态, 对应概率)

        失败模式:
            - 多个状态概率相同: 返回索引最小的那个
        """
        best_idx = 0
        best_prob = 0.0
        for i in range(self._N):
            p = self.probability(i)
            if p > best_prob:
                best_prob = p
                best_idx = i
        return best_idx, best_prob

    def sample(self, shots: int, collapse: bool = False) -> List[Tuple[int, float]]:
        """
        执行多次测量。

        参数:
            shots: 测量次数
            collapse: 如果为 True，每次测量后态坍缩到测量结果

        返回:
            [(测量结果, 概率), ...]

        失败模式:
            - shots <= 0: 返回空列表
            - 极低概率事件: 可能测不到 (这是物理特性，非 bug)
        """
        if shots <= 0:
            return []
        results = []
        for _ in range(shots):
            result, prob = self.measure(collapse=collapse)
            results.append((result, prob))
        return results

    # ----------------------------------------------------------------
    # 态信息与诊断
    # ----------------------------------------------------------------

    def check_normalization(self, tolerance: float = 1e-10) -> bool:
        """
        验证归一化: sum |alpha_i|^2 ≈ 1。

        失败模式:
            - 多次门操作后浮点误差累积: 返回 False
            - 代码 bug 导致振幅被破坏: 返回 False

        用于调试和测试断言。
        """
        total = sum(self.probability(i) for i in range(self._N))
        return abs(total - 1.0) < tolerance

    def normalize(self):
        """
        强制重新归一化量子态。

        将每个振幅除以 sqrt(sum |alpha_i|^2)。

        失败模式:
            - 所有振幅为零: 无法归一化，抛出 StateNormalizationError
            - 数值下溢: total_prob 可能为 0.0
        """
        total = sum(self.probability(i) for i in range(self._N))
        if total < 1e-30:
            raise StateNormalizationError(total)
        norm = math.sqrt(total)
        for i in range(self._N):
            self.amplitudes[i] /= norm

    def get_state_vector(self) -> List[complex]:
        """返回振幅列表的副本。"""
        return self.amplitudes.copy()

    def clone(self) -> 'QuantumState':
        """深拷贝当前量子态，绕过验证以提高性能。"""
        new_state = QuantumState.__new__(QuantumState)
        new_state._n = self._n
        new_state._N = self._N
        new_state.amplitudes = self.amplitudes.copy()
        new_state._gate_history = self._gate_history.copy()
        new_state._readout_error = self._readout_error
        return new_state

    # ----------------------------------------------------------------
    # 显示
    # ----------------------------------------------------------------

    def summarize(self, top_n: int = 8) -> str:
        """
        生成量子态的文本摘要。

        只显示概率最高的前 top_n 个基态。

        失败模式:
            - top_n <= 0: 返回空摘要
            - 所有振幅为零: 显示 "空态" 消息
        """
        if top_n <= 0:
            return ""
        pairs = [(i, self.amplitudes[i]) for i in range(self._N)]
        nonzero = [(i, a) for i, a in pairs if abs(a) > 1e-12]
        if not nonzero:
            return "  (空态: 所有振幅为零)\n"
        nonzero.sort(key=lambda x: -(x[1].real ** 2 + x[1].imag ** 2))
        lines = [
            f"  {'基态':>12}  {'振幅':>24}  {'概率':>12}",
            f"  {'-'*12}  {'-'*24}  {'-'*12}",
        ]
        for i, a in nonzero[:top_n]:
            bits = format(i, f'0{self._n}b')
            prob = a.real * a.real + a.imag * a.imag
            lines.append(
                f"  |{bits}>     {a.real:+.8f}{a.imag:+.8f}j     {prob:.8f}"
            )
        if len(nonzero) > top_n:
            lines.append(f"  ... (还有 {len(nonzero) - top_n} 个非零振幅)")
        return '\n'.join(lines) + '\n'

    def print_state(self, top_n: int = 8):
        """打印量子态摘要到标准输出。"""
        print(self.summarize(top_n))

    # ----------------------------------------------------------------
    # 魔法方法
    # ----------------------------------------------------------------

    def __repr__(self) -> str:
        return f"QuantumState(n_qubits={self._n}, dim={self._N})"

    def __len__(self) -> int:
        return self._N

    def __getitem__(self, index: int) -> complex:
        return self.amplitudes[index]

    # ----------------------------------------------------------------
    # 通用门应用 (通过 numpy 矩阵)
    # ----------------------------------------------------------------

    def apply_gate(self, matrix, *targets: int):
        """
        Apply an arbitrary gate matrix to the specified qubits.

        Uses the matrix representation to transform the state vector.
        The gate matrix should be 2^k x 2^k for k = len(targets).

        Args:
            matrix: numpy ndarray of shape (2^k, 2^k) representing the gate
            *targets: qubit indices the gate acts on, from LSB to MSB

        Raises:
            QubitIndexError: if any target is out of bounds
            ValueError: if matrix shape doesn't match number of targets
        """
        matrix = np.asarray(matrix, dtype=complex)
        k = len(targets)
        expected_dim = 1 << k

        if matrix.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Gate matrix shape {matrix.shape} does not match "
                f"{k} qubit targets (expected {expected_dim}x{expected_dim})"
            )

        for t in targets:
            validate_qubit_index(t, self._n)

        if k == 0:
            return

        new_amps = np.zeros(self._N, dtype=complex)

        # Create masks and bit ordering
        target_masks = [1 << t for t in targets]

        max_amp = np.max(np.abs(self.amplitudes))
        threshold = max_amp * 1e-15 if max_amp > 0 else 1e-40
        
        for i in range(self._N):
            if abs(self[i]) < threshold:
                continue

            row_idx = 0
            for bit_pos, mask in enumerate(reversed(target_masks)):
                if i & mask:
                    row_idx |= (1 << bit_pos)

            # Apply all columns of the matrix
            for col_idx in range(expected_dim):
                if abs(matrix[row_idx, col_idx]) < 1e-30:
                    continue

                # Build the output basis state index
                j = i
                for bit_pos, mask in enumerate(target_masks):
                    if col_idx & (1 << bit_pos):
                        j |= mask
                    else:
                        j &= ~mask

                new_amps[j] += matrix[row_idx, col_idx] * self[i]

        self.amplitudes = new_amps

    def apply_gate_dict(self, gate: dict):
        """
        Apply a gate from a dictionary representation.

        Supports standard gates and FUSED_U3 gates produced by gate_fusion.

        Args:
            gate: Dictionary with 'gate', 'targets', and optional 'params' and 'control'
        """
        gate_type = gate.get('gate', '')
        targets = gate.get('targets', [])
        params = gate.get('params', {})
        control = gate.get('control')

        if gate_type == 'FUSED_U3':
            matrix = np.array(params.get('matrix'), dtype=complex)
            if targets:
                self.apply_gate(matrix, *targets)
        elif gate_type == 'H':
            for t in targets:
                self.h(t)
        elif gate_type == 'X':
            for t in targets:
                self.x(t)
        elif gate_type == 'Y':
            for t in targets:
                self.y(t)
        elif gate_type == 'Z':
            for t in targets:
                self.z(t)
        elif gate_type == 'S':
            for t in targets:
                self.s(t)
        elif gate_type == 'T':
            for t in targets:
                self.t(t)
        elif gate_type == 'CNOT':
            if control is not None and targets:
                self.cnot(control, targets[0])
        elif gate_type == 'CZ':
            if control is not None and targets:
                self.cz(control, targets[0])
        elif gate_type == 'SWAP':
            if len(targets) >= 2:
                self.swap(targets[0], targets[1])
        elif gate_type == 'TOFFOLI':
            if len(targets) >= 3:
                self.toffoli(targets[0], targets[1], targets[2])
        elif gate_type == 'RX':
            theta = params.get('theta', 0)
            for t in targets:
                self.rx(t, theta)
        elif gate_type == 'RY':
            theta = params.get('theta', 0)
            for t in targets:
                self.ry(t, theta)
        elif gate_type == 'RZ':
            phi = params.get('phi', 0)
            for t in targets:
                self.rz(t, phi)
        elif gate_type == 'ORACLE':
            expression = params.get('expression', '')
            if expression:
                from ..core.parser import parse_bool, build_oracle_function
                expr = parse_bool(expression)
                oracle = build_oracle_function([expr])
                marked = set()
                for i in range(self._N):
                    if oracle(i):
                        marked.add(i)
                self.phase_oracle(marked)

    def draw_circuit(self, ascii: bool = True) -> str:
        """
        Draw an ASCII representation of a circuit applied to this state.

        Renders the gate operations stored in _gate_history.

        Args:
            ascii: if True, use ASCII characters; if False, use Unicode

        Returns:
            String representation of the circuit diagram
        """
        n = self._n

        # Default: show qubit lines if no gate history
        if not self._gate_history:
            if ascii:
                lines = []
                for i in range(n):
                    lines.append(f"q{i}: " + "---" * min(10, max(1, n)))
                return '\n'.join(lines)
            else:
                lines = []
                for i in range(n):
                    lines.append(f"q{i}: " + "\u2500" * min(30, max(3, n * 3)))
                return '\n'.join(lines)

        # Build circuit columns from gate history
        max_cols = len(self._gate_history)
        # Build a grid: rows=qubits, cols=gates
        grid = [["---" for _ in range(max_cols)] for _ in range(n)]

        for col, gate_op in enumerate(self._gate_history):
            gate_type = gate_op.get('gate', '?')
            targets = gate_op.get('targets', [])
            control = gate_op.get('control')
            label = gate_type[:3]  # Short label

            if control is not None:
                # Multi-qubit gate with control
                if 0 <= control < n:
                    grid[control][col] = "\u25CF-"  # control dot
                for t in targets:
                    if 0 <= t < n:
                        grid[t][col] = "[" + label + "]"
                # Draw vertical connection for CNOT-like gates
                all_qubits = sorted([control] + targets)
                for qi in range(all_qubits[0] + 1, all_qubits[-1]):
                    if qi < n and qi not in (control, *targets):
                        grid[qi][col] = " | "
            else:
                for t in targets:
                    if 0 <= t < n:
                        grid[t][col] = "[" + label + "]"

        # Assemble lines
        lines = []
        for qi in range(n):
            line = f"q{qi}: " + "".join(grid[qi])
            lines.append(line)

        return '\n'.join(lines)

    def add_noise(self, depolarizing_p: float = 0.0,
                  readout_error_p: float = 0.0):
        """
        Add depolarizing noise and readout error to the quantum state.

        Depolarizing channel:
            rho -> (1-p) * rho + p * I/2^n
        For a pure state, this mixes in the maximally mixed state.

        Readout error:
            Each qubit has probability p of being flipped during measurement.

        Args:
            depolarizing_p: Depolarizing probability [0, 1]
            readout_error_p: Readout error probability per qubit [0, 1]

        Note:
            This modifies the state in-place. The depolarizing channel
            reduces purity and coherence.
        """
        if depolarizing_p < 0 or depolarizing_p > 1:
            raise ValueError(
                f"depolarizing_p must be in [0, 1], got {depolarizing_p}"
            )
        if readout_error_p < 0 or readout_error_p > 1:
            raise ValueError(
                f"readout_error_p must be in [0, 1], got {readout_error_p}"
            )

        # Apply depolarizing noise via stochastic Pauli errors
        if depolarizing_p > 0:
            import random as _random
            # For each qubit, with probability p, apply X, Y, or Z with equal probability
            for q in range(self._n):
                if _random.random() < depolarizing_p:
                    err = _random.choice([0, 1, 2])
                    if err == 0:
                        self.x(q)
                    elif err == 1:
                        self.y(q)
                    else:
                        self.z(q)

        # Apply readout error: flipping probability for each qubit
        if readout_error_p > 0:
            # Readout error can be modeled by applying X with probability p
            # to each qubit before measurement. We store this metadata.
            # In practice, this modifies measurement probabilities.
            self._readout_error = readout_error_p
        else:
            self._readout_error = 0.0


class DensityMatrix:
    """
    Density matrix representation for mixed quantum states.

    Supports mixed states, decoherence channels, and noisy operations
    that cannot be represented by pure state vectors alone.

    rho = sum_i p_i |psi_i><psi_i|

    Properties:
        dim: Hilbert space dimension (= 2^n)
        n_qubits: Number of qubits
        matrix: Density matrix as list of lists of complex numbers

    Usage:
        >>> dm = DensityMatrix.from_pure(QuantumState(2))
        >>> dm.apply_depolarizing(0.05)
        >>> result, prob = dm.measure()
    """

    def __init__(self, n_qubits: int):
        """
        Initialize a pure |0...0> density matrix.

        Args:
            n_qubits: Number of qubits (1 <= n_qubits <= 16)
        """
        validate_n_qubits(n_qubits, 16)
        self._n = n_qubits
        self._N = 1 << n_qubits
        self._rho = np.zeros((self._N, self._N), dtype=complex)
        self._rho[0, 0] = 1.0 + 0j

    @staticmethod
    def from_pure(state: QuantumState) -> 'DensityMatrix':
        """Create density matrix from pure state: rho = |psi><psi|."""
        dm = DensityMatrix.__new__(DensityMatrix)
        dm._n = state.n_qubits
        dm._N = state.size
        validate_n_qubits(dm._n, 16)
        amps = state.amplitudes
        dm._rho = np.outer(amps, amps.conj())
        return dm

    @staticmethod
    def from_probabilities(states: List[Tuple[float, QuantumState]]) -> 'DensityMatrix':
        """Create mixed state: rho = sum p_i |psi_i><psi_i|."""
        first = states[0][1]
        dm = DensityMatrix.__new__(DensityMatrix)
        dm._n = first.n_qubits
        dm._N = first.size
        validate_n_qubits(dm._n, 16)
        dm._rho = np.zeros((dm._N, dm._N), dtype=complex)
        for prob, state in states:
            amps = state.amplitudes
            dm._rho += prob * np.outer(amps, amps.conj())
        return dm

    @property
    def n_qubits(self) -> int:
        return self._n

    @property
    def dim(self) -> int:
        return self._N

    def get_matrix(self) -> List[List[complex]]:
        """Return a copy of the density matrix."""
        return [row[:] for row in self._rho]

    def purity(self) -> float:
        """Compute purity: Tr(rho^2)."""
        tr_rho2 = 0.0
        for i in range(self._N):
            for k in range(self._N):
                tr_rho2 += (self._rho[i][k] * self._rho[k][i]).real
        return tr_rho2

    def von_neumann_entropy(self) -> float:
        """Compute von Neumann entropy: S = -Tr(rho log2 rho)."""
        import numpy as np
        mat = np.array(self._rho, dtype=complex)
        eigenvalues = np.linalg.eigvalsh(mat)
        entropy = 0.0
        for ev in eigenvalues:
            if ev > 1e-12:
                entropy -= ev * math.log2(ev)
        return max(0.0, entropy)

    def trace(self) -> float:
        """Compute trace: should be 1.0 for valid states."""
        return sum(self._rho[i][i].real for i in range(self._N))

    def probability(self, index: int) -> float:
        """Probability of measuring basis state |index>:  <i|rho|i>."""
        if index < 0 or index >= self._N:
            return 0.0
        return self._rho[index][index].real

    def measure(self, collapse: bool = False) -> Tuple[int, float]:
        """
        Measure the system in the computational basis.

        Args:
            collapse: If True, collapse state to measured basis.

        Returns:
            (result index, probability)
        """
        r = random.random()
        cumulative = 0.0
        for i in range(self._N):
            cumulative += self._rho[i][i].real
            if r < cumulative:
                if collapse:
                    self._collapse_to(i)
                return i, self._rho[i][i].real
        last = self._N - 1
        if collapse:
            self._collapse_to(last)
        return last, self._rho[last][last].real

    def _collapse_to(self, index: int):
        """Collapse density matrix to pure |index> state."""
        for i in range(self._N):
            for j in range(self._N):
                self._rho[i][j] = 0j
        self._rho[index][index] = 1.0 + 0j

    def apply_unitary(self, matrix):
        """
        Apply unitary transformation: rho -> U rho U^dagger.

        Args:
            matrix: Unitary matrix as numpy array (N x N) or list of lists.
        """
        import numpy as np
        U = np.asarray(matrix, dtype=complex)
        rho = np.array(self._rho, dtype=complex)
        rho_new = U @ rho @ U.conj().T
        self._rho = [[rho_new[i, j] for j in range(self._N)] for i in range(self._N)]

    def apply_depolarizing(self, p: float):
        """
        Apply depolarizing channel: rho -> (1 - p)*rho + p*I/N.

        Args:
            p: Depolarizing probability in [0, 1]
        """
        if p < 0 or p > 1:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if p == 0:
            return
        identity_prob = p / self._N
        for i in range(self._N):
            for j in range(self._N):
                self._rho[i][j] *= (1.0 - p)
            self._rho[i][i] += identity_prob

    def apply_amplitude_damping(self, gamma: float):
        """
        Apply amplitude damping (T1 decay) channel.

        Kraus operators (per qubit):
            K0 = |0><0| + sqrt(1-gamma)|1><1|
            K1 = sqrt(gamma)|0><1|

        Args:
            gamma: Damping probability in [0, 1]
        """
        if gamma < 0 or gamma > 1:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if gamma == 0:
            return

        import numpy as np
        sqrt_1mg = math.sqrt(1.0 - gamma)
        sqrt_g = math.sqrt(gamma)

        rho = np.array(self._rho, dtype=complex)

        # Apply per-qubit: for simplicity, apply on qubit 0 only
        # Full multi-qubit version would apply to all qubits
        new_rho = np.zeros((self._N, self._N), dtype=complex)
        mask = 1  # qubit 0

        for i in range(self._N):
            i0 = (i & ~mask)  # i with qubit 0 forced to |0>
            i1 = i0 | mask    # i with qubit 0 forced to |1>

            for j in range(self._N):
                j0 = (j & ~mask)
                j1 = j0 | mask

                val = rho[i, j]

                if (i & mask) == 0 and (j & mask) == 0:
                    # |0><0| terms: K0 has 1 for |0>, plus rho[1,1] term from K1
                    new_rho[i, j] += val  # K0|0>|0>K0^dagger
                    if i1 < self._N and j1 < self._N:
                        new_rho[i, j] += gamma * rho[i1, j1]  # K1|1><1|K1^dagger
                elif (i & mask) == 0 and (j & mask) != 0:
                    # |0><1|: K0|0><1|K0^dagger = sqrt(1-gamma) * rho[i,j]
                    new_rho[i, j] += sqrt_1mg * val
                elif (i & mask) != 0 and (j & mask) == 0:
                    # |1><0|: same
                    new_rho[i, j] += sqrt_1mg * val
                else:
                    # |1><1|: K0|1><1|K0^dagger = (1-gamma) * rho[i,j]
                    new_rho[i, j] += (1.0 - gamma) * val

        self._rho = [[new_rho[i, j] for j in range(self._N)] for i in range(self._N)]

    def apply_phase_damping(self, gamma: float):
        """
        Apply phase damping (T2 dephasing) channel.

        Kraus operators:
            K0 = sqrt(1 - gamma) * I
            K1 = sqrt(gamma) * |0><0|
            K2 = sqrt(gamma) * |1><1| (but with phase -1)

        Args:
            gamma: Dephasing probability in [0, 1]
        """
        if gamma < 0 or gamma > 1:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")

        rho = self._rho
        N = self._N
        scale = math.sqrt(1.0 - gamma)

        new_rho = [[0j] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                # Off-diagonal elements decay, diagonal preserved
                num_diff = bin(i ^ j).count('1')
                new_rho[i][j] = scale ** num_diff * rho[i][j]

        self._rho = new_rho

    def partial_trace(self, qubit: int) -> 'DensityMatrix':
        """
        Trace out a single qubit.

        Args:
            qubit: Index of qubit to trace out

        Returns:
            DensityMatrix with one fewer qubit
        """
        mask = 1 << qubit
        new_n = self._n - 1
        new_N = 1 << new_n

        dm = DensityMatrix.__new__(DensityMatrix)
        dm._n = new_n
        dm._N = new_N
        dm._rho = [[0j] * new_N for _ in range(new_N)]

        for i in range(self._N):
            for j in range(self._N):
                if (i & mask) == (j & mask):
                    i_new = (i & (mask - 1)) | ((i >> (qubit + 1)) << qubit)
                    j_new = (j & (mask - 1)) | ((j >> (qubit + 1)) << qubit)
                    dm._rho[i_new][j_new] += self._rho[i][j]

        return dm

    def fidelity(self, other: 'DensityMatrix') -> float:
        """Compute fidelity F = Tr(sqrt(sqrt(rho1) * rho2 * sqrt(rho1)))^2."""
        import numpy as np
        rho1 = np.array(self._rho, dtype=complex)
        rho2 = np.array(other._rho, dtype=complex)
        try:
            eigvals, eigvecs = np.linalg.eigh(rho1)
            sqrt_rho1 = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0))) @ eigvecs.conj().T
            inner = sqrt_rho1 @ rho2 @ sqrt_rho1
            eigvals_inner = np.linalg.eigvalsh(inner)
            fid = max(0.0, sum(max(0, math.sqrt(ev.real)) for ev in eigvals_inner)) ** 2
        except np.linalg.LinAlgError:
            fid = abs(np.trace(rho1 @ rho2))
        return min(1.0, fid)

    def __repr__(self) -> str:
        purity_val = self.purity()
        return (f"DensityMatrix(n_qubits={self._n}, dim={self._N}, "
                f"purity={purity_val:.4f})")
