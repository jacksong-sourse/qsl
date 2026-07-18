"""
QSL - Quantum Search Language (Legacy Monolithic Version)
==========================================================
基于"前提-工具-问题-主函数"框架的量子搜索领域专用语言。

**DEPRECATED**: This standalone qsl.py is the legacy monolithic version.
Please use the `qsl` package instead:
    pip install qsl-quantum
    from qsl import QSLProgram, QSLCompiler

数学基础：
  H = (C^2)^{⊗n} ≅ C^{2^n}  (n量子比特希尔伯特空间)
  Grover 搜索：O(√N) vs 经典 O(N)
  布尔表达式 → 量子Oracle 的综合
"""

import math
import cmath
import random
import re
from typing import List, Tuple, Callable, Dict, Optional, Set
from dataclasses import dataclass, field

# ╔══════════════════════════════════════════════════════════════╗
# ║  PART 1: 量子态向量 (Quantum State Vector)                   ║
# ║  |ψ⟩ = Σ_{i=0}^{2^n-1} α_i |i⟩                             ║
# ╚══════════════════════════════════════════════════════════════╝

class QuantumState:
    """
    n量子比特的纯态 |ψ⟩，用 2^n 个复振幅表示。
    
    数学表示：
        |ψ⟩ = Σ_{i=0}^{N-1} α_i |i⟩
        其中 N = 2^n, Σ|α_i|² = 1
        |i⟩ 是计算基态 |b_{n-1}...b_0⟩
    """

    def __init__(self, n_qubits: int):
        if n_qubits < 1:
            raise ValueError("量子比特数必须 >= 1")
        if n_qubits > 20:
            raise ValueError("模拟器限制：量子比特数 <= 20（内存限制）")

        self.n = n_qubits
        self.N = 1 << n_qubits  # 2^n
        # 初始态 |0...0⟩
        self.amps = [0j] * self.N
        self.amps[0] = 1.0 + 0j

    # ----------------------------------------------------------
    # 基本门操作（所有门都是就地修改状态向量）
    # ----------------------------------------------------------

    def _bit(self, x: int, k: int) -> int:
        """提取整数 x 的第 k 位（0-indexed，从LSB开始）"""
        return (x >> k) & 1

    # --- Pauli-X 门 (量子非门) ---
    # X|0⟩ = |1⟩, X|1⟩ = |0⟩
    # 矩阵：[[0, 1], [1, 0]]
    def x(self, target: int):
        """
        对第 target 个量子比特施加 Pauli-X（NOT）门。
        
        作用：交换 target=0 和 target=1 态的振幅。
        
        算法：遍历所有 target 位为 0 的状态 i，与 target 位为 1 的状态 j=i⊕(1<<target) 交换振幅。
        时间复杂度：O(N)
        """
        mask = 1 << target
        for i in range(self.N):
            if (i & mask) == 0:  # target 位为 0
                j = i | mask      # target 位为 1
                self.amps[i], self.amps[j] = self.amps[j], self.amps[i]

    # --- Hadamard 门 ---
    # H|0⟩ = (|0⟩+|1⟩)/√2, H|1⟩ = (|0⟩-|1⟩)/√2
    # 矩阵：[[1, 1], [1, -1]] / √2
    def h(self, target: int):
        """
        对第 target 个量子比特施加 Hadamard 门。
        
        数学推导：
            设 i_0 为 target=0 的态，i_1 为 target=1 的态。
            变换后：
                α'_{i_0} = (α_{i_0} + α_{i_1}) / √2
                α'_{i_1} = (α_{i_0} - α_{i_1}) / √2
        
        时间复杂度：O(N)
        """
        mask = 1 << target
        inv_sqrt2 = 1.0 / math.sqrt(2)
        for i in range(self.N):
            if (i & mask) == 0:
                j = i | mask
                a_i = self.amps[i]
                a_j = self.amps[j]
                self.amps[i] = (a_i + a_j) * inv_sqrt2
                self.amps[j] = (a_i - a_j) * inv_sqrt2

    # --- Pauli-Z 门 ---
    # Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
    # 矩阵：[[1, 0], [0, -1]]
    def z(self, target: int):
        """
        对第 target 个量子比特施加 Pauli-Z 门。
        
        数学推导：
            Z|b⟩ = (-1)^b |b⟩
            即 target=1 的态振幅乘以 -1。
        
        时间复杂度：O(N)
        """
        mask = 1 << target
        for i in range(self.N):
            if i & mask:  # target 位为 1
                self.amps[i] = -self.amps[i]

    # --- CNOT 门 (受控非门) ---
    # CNOT|a,b⟩ = |a, a⊕b⟩
    def cnot(self, control: int, target: int):
        """
        受控非门：若 control 为 |1⟩，则翻转 target。
        
        数学推导：
            CNOT|a⟩|b⟩ = |a⟩|a⊕b⟩
            
        实现：遍历 control=1 且 target=0 的状态，
              与 control=1 且 target=1 的状态交换振幅。
        
        时间复杂度：O(N)
        """
        c_mask = 1 << control
        t_mask = 1 << target
        for i in range(self.N):
            if (i & c_mask) and (i & t_mask) == 0:  # control=1, target=0
                j = i ^ t_mask  # control=1, target=1
                self.amps[i], self.amps[j] = self.amps[j], self.amps[i]

    # --- Toffoli 门 (CCNOT, 受控受控非门) ---
    # Toffoli|a,b,c⟩ = |a,b, c⊕(a∧b)⟩
    def toffoli(self, c1: int, c2: int, target: int):
        """
        Toffoli 门（双控制非门）：若 c1 和 c2 都为 |1⟩，则翻转 target。
        
        数学推导：
            Toffoli|a⟩|b⟩|c⟩ = |a⟩|b⟩|c⊕(a∧b)⟩
            
        Toffoli 门是通用可逆门——任意布尔函数都可以用 Toffoli 门
        加上辅助比特来实现。这是"前提→二进制→量子电路"编译的基础。
        
        时间复杂度：O(N)
        """
        c1_mask = 1 << c1
        c2_mask = 1 << c2
        t_mask = 1 << target
        both_mask = c1_mask | c2_mask
        for i in range(self.N):
            if (i & both_mask) == both_mask and (i & t_mask) == 0:
                j = i ^ t_mask
                self.amps[i], self.amps[j] = self.amps[j], self.amps[i]

    # --- 多控制 Z 门 ---
    # MCZ|q₁...qₙ⟩ = (-1)^{q₁∧...∧qₙ} |q₁...qₙ⟩
    def mcz(self, qubits: List[int]):
        """
        多控制 Z 门：当所有指定量子比特都为 |1⟩ 时，相位翻转。
        
        数学推导：
            MCZ|x₁...xₖ⟩ = 
              |x₁...xₖ⟩    若任意 xⱼ = 0
              -|11...1⟩    若所有 xⱼ = 1
            
        这是 Grover 扩散算子的核心组件。
        扩散算子 D = H^{⊗n} · X^{⊗n} · MCZ · X^{⊗n} · H^{⊗n}
        
        时间复杂度：O(N)
        """
        mask = 0
        for q in qubits:
            mask |= (1 << q)
        for i in range(self.N):
            if (i & mask) == mask:  # 所有控制位都是 1
                self.amps[i] = -self.amps[i]

    # --- 相位 Oracle ---
    def phase_oracle(self, marked: Set[int]):
        """
        相位 Oracle：对 marked 中的状态施加相位翻转 (-1)。
        
        数学推导：
            Oracle|x⟩ = (-1)^{f(x)}|x⟩
            其中 f(x) = 1 当 x ∈ marked，否则 0
        
        等效于：Oracle = I - 2 Σ_{x∈marked} |x⟩⟨x|
        
        时间复杂度：O(|marked|) —— 只处理被标记的状态
        """
        for x in marked:
            if 0 <= x < self.N:
                self.amps[x] = -self.amps[x]

    # ----------------------------------------------------------
    # 量子态信息
    # ----------------------------------------------------------

    def probability(self, x: int) -> float:
        """返回测量得到 |x⟩ 的概率 P(x) = |α_x|²"""
        return abs(self.amps[x]) ** 2

    def probabilities(self) -> List[float]:
        """返回所有基态的概率分布"""
        return [abs(a) ** 2 for a in self.amps]

    def measure(self) -> Tuple[int, float]:
        """
        量子测量：按概率分布 |α_i|² 随机坍缩到一个基态。
        
        数学推导（玻恩定则）：
            P(得到结果 i) = |⟨i|ψ⟩|² = |α_i|²
            测量后态坍缩为：|ψ⟩ → |i⟩
        
        返回：(测量结果, 对应概率)
        """
        r = random.random()
        cumulative = 0.0
        for i in range(self.N):
            cumulative += abs(self.amps[i]) ** 2
            if r < cumulative:
                return i, abs(self.amps[i]) ** 2
        # 处理浮点精度问题
        return self.N - 1, abs(self.amps[-1]) ** 2

    def measure_most_likely(self) -> Tuple[int, float]:
        """返回概率最大的测量结果"""
        probs = self.probabilities()
        max_i = max(range(self.N), key=lambda i: probs[i])
        return max_i, probs[max_i]

    def print_state(self, top_n: int = 8):
        """打印当前量子态（显示前 top_n 个非零振幅）"""
        print(f"\n  {'基态 |i⟩':>12}  {'振幅 α_i':>20}  {'概率 |α_i|²':>14}")
        print(f"  {'─'*12}  {'─'*20}  {'─'*14}")
        nonzero = [(i, a) for i, a in enumerate(self.amps) if abs(a) > 1e-10]
        nonzero.sort(key=lambda x: -abs(x[1]) ** 2)
        for i, a in nonzero[:top_n]:
            bin_str = format(i, f'0{self.n}b')
            prob = abs(a) ** 2
            print(f"  |{bin_str}⟩     {a.real:+.6f}{a.imag:+.6f}j     {prob:.6f}")
        if len(nonzero) > top_n:
            print(f"  ... (共 {len(nonzero)} 个非零振幅)")

    def clone(self) -> 'QuantumState':
        """深拷贝量子态"""
        new_state = QuantumState(self.n)
        new_state.amps = self.amps.copy()
        return new_state


# ╔══════════════════════════════════════════════════════════════╗
# ║  PART 2: 布尔表达式解析器 (Boolean Expression Parser)         ║
# ║  将 "x0 & x1 | ~x2" 编译为可评估的布尔函数                     ║
# ╚══════════════════════════════════════════════════════════════╝

class BooleanExpr:
    """布尔表达式的抽象语法树节点"""

    def evaluate(self, assignment: int) -> bool:
        """在给定的 n 位赋值下评估表达式的值"""
        raise NotImplementedError

    def to_qasm_style(self) -> str:
        """转为可读的字符串表示"""
        raise NotImplementedError


class VarExpr(BooleanExpr):
    """变量节点：x[k] 表示第 k 个量子比特"""
    def __init__(self, index: int):
        self.index = index

    def evaluate(self, assignment: int) -> bool:
        return bool((assignment >> self.index) & 1)

    def to_qasm_style(self) -> str:
        return f"x{self.index}"


class NotExpr(BooleanExpr):
    """逻辑非：~expr"""
    def __init__(self, expr: BooleanExpr):
        self.expr = expr

    def evaluate(self, assignment: int) -> bool:
        return not self.expr.evaluate(assignment)

    def to_qasm_style(self) -> str:
        return f"~({self.expr.to_qasm_style()})"


class AndExpr(BooleanExpr):
    """逻辑与：left & right"""
    def __init__(self, left: BooleanExpr, right: BooleanExpr):
        self.left = left
        self.right = right

    def evaluate(self, assignment: int) -> bool:
        return self.left.evaluate(assignment) and self.right.evaluate(assignment)

    def to_qasm_style(self) -> str:
        return f"({self.left.to_qasm_style()} & {self.right.to_qasm_style()})"


class OrExpr(BooleanExpr):
    """逻辑或：left | right"""
    def __init__(self, left: BooleanExpr, right: BooleanExpr):
        self.left = left
        self.right = right

    def evaluate(self, assignment: int) -> bool:
        return self.left.evaluate(assignment) or self.right.evaluate(assignment)

    def to_qasm_style(self) -> str:
        return f"({self.left.to_qasm_style()} | {self.right.to_qasm_style()})"


class XorExpr(BooleanExpr):
    """逻辑异或：left ^ right"""
    def __init__(self, left: BooleanExpr, right: BooleanExpr):
        self.left = left
        self.right = right

    def evaluate(self, assignment: int) -> bool:
        return self.left.evaluate(assignment) != self.right.evaluate(assignment)

    def to_qasm_style(self) -> str:
        return f"({self.left.to_qasm_style()} ^ {self.right.to_qasm_style()})"


class BooleanParser:
    """
    布尔表达式递归下降解析器。
    
    语法规则 (按优先级递增)：
        expr     := xor_expr ('|' xor_expr)*            # OR，最低优先级
        xor_expr := and_expr ('^' and_expr)*            # XOR
        and_expr := not_expr ('&' not_expr)*            # AND
        not_expr := '~' not_expr | atom                 # NOT
        atom     := IDENTIFIER | '(' expr ')'
        IDENTIFIER := 'x' DIGIT+ | [a-zA-Z_][a-zA-Z0-9_]*
    
    这个语法支持任意嵌套的布尔表达式，例如：
        (x0 & x1) | (~x2 ^ x3)
    """

    def __init__(self, source: str):
        self.source = source
        self.pos = 0

    def parse(self) -> BooleanExpr:
        """解析完整表达式，入口点"""
        result = self._parse_expr()
        self._skip_whitespace()
        if self.pos < len(self.source):
            raise ValueError(f"表达式解析错误：'{self.source}' 中位置 {self.pos} 处有未预期的字符 '{self.source[self.pos]}'")
        return result

    def _skip_whitespace(self):
        while self.pos < len(self.source) and self.source[self.pos] in ' \t\n\r':
            self.pos += 1

    def _peek(self) -> str:
        self._skip_whitespace()
        if self.pos < len(self.source):
            return self.source[self.pos]
        return ''

    def _consume(self) -> str:
        self._skip_whitespace()
        if self.pos < len(self.source):
            ch = self.source[self.pos]
            self.pos += 1
            return ch
        return ''

    def _parse_expr(self) -> BooleanExpr:
        """expr := xor_expr ('|' xor_expr)*"""
        left = self._parse_xor_expr()
        while self._peek() == '|':
            self._consume()  # 吃掉 '|'
            right = self._parse_xor_expr()
            left = OrExpr(left, right)
        return left

    def _parse_xor_expr(self) -> BooleanExpr:
        """xor_expr := and_expr ('^' and_expr)*"""
        left = self._parse_and_expr()
        while self._peek() == '^':
            self._consume()
            right = self._parse_and_expr()
            left = XorExpr(left, right)
        return left

    def _parse_and_expr(self) -> BooleanExpr:
        """and_expr := not_expr ('&' not_expr)*"""
        left = self._parse_not_expr()
        while self._peek() == '&':
            self._consume()
            right = self._parse_not_expr()
            left = AndExpr(left, right)
        return left

    def _parse_not_expr(self) -> BooleanExpr:
        """not_expr := '~' not_expr | atom"""
        if self._peek() == '~':
            self._consume()
            expr = self._parse_not_expr()
            return NotExpr(expr)
        return self._parse_atom()

    def _parse_atom(self) -> BooleanExpr:
        """atom := IDENTIFIER | '(' expr ')'"""
        ch = self._peek()
        if ch == '(':
            self._consume()  # 吃掉 '('
            expr = self._parse_expr()
            if self._peek() != ')':
                raise ValueError(f"缺少右括号: '{self.source}'")
            self._consume()  # 吃掉 ')'
            return expr
        else:
            # 解析标识符
            ident = self._parse_identifier()
            # 特殊处理：x后跟数字 -> 变量
            if ident.startswith('x') and ident[1:].isdigit():
                idx = int(ident[1:])
                return VarExpr(idx)
            # 通用变量名（用于约束命名等）
            return VarExpr(ident)

    def _parse_identifier(self) -> str:
        """解析标识符：[a-zA-Z_][a-zA-Z0-9_]* 或 x后跟数字"""
        self._skip_whitespace()
        start = self.pos
        if self.pos < len(self.source) and (self.source[self.pos].isalpha() or self.source[self.pos] == '_'):
            self.pos += 1
            while self.pos < len(self.source) and (self.source[self.pos].isalnum() or self.source[self.pos] == '_'):
                self.pos += 1
        if start == self.pos:
            raise ValueError(f"期望标识符，在位置 {self.pos}: '{self.source}'")
        return self.source[start:self.pos]


def parse_bool(expression: str) -> BooleanExpr:
    """便捷函数：解析布尔表达式字符串"""
    return BooleanParser(expression).parse()


# ╔══════════════════════════════════════════════════════════════╗
# ║  PART 3: Grover 量子搜索算法                                  ║
# ║  G = D · Oracle, G^t|ψ₀⟩ → |sol⟩                            ║
# ╚══════════════════════════════════════════════════════════════╝

class GroverSearch:
    """
    Grover 量子搜索算法的完整实现。
    
    算法流程：
        1. 初始化：|ψ₀⟩ = H^{⊗n} |0⟩^{⊗n} = 1/√N Σ|x⟩
        2. 对 t = 1, 2, ..., t_opt:
           a. 施加 Oracle：标记目标态（相位翻转）
           b. 施加扩散算子 D = H^{⊗n}·X^{⊗n}·MCZ·X^{⊗n}·H^{⊗n}
        3. 测量
    
    数学推导（详见类文档）
    """

    def __init__(self, n_qubits: int, verbose: bool = True):
        """
        参数：
            n_qubits: 量子比特数（搜索空间大小 N = 2^n）
            verbose: 是否打印详细计算过程
        """
        self.n = n_qubits
        self.N = 1 << n_qubits
        self.verbose = verbose
        self._log_lines: List[str] = []

    def _log(self, msg: str):
        if self.verbose:
            print(msg)
        self._log_lines.append(msg)

    def get_log(self) -> str:
        return '\n'.join(self._log_lines)

    def _build_phase_oracle_mask(self, condition: Callable[[int], bool]) -> Set[int]:
        """
        从条件函数构建相位 Oracle 的标记集。
        
        对于每个状态 x ∈ {0,1}^n，若 condition(x)=True 则标记。
        """
        marked = set()
        for x in range(self.N):
            if condition(x):
                marked.add(x)
        return marked

    def search(self,
               condition: Callable[[int], bool],
               num_solutions: Optional[int] = None,
               shots: int = 1) -> List[Tuple[int, float]]:
        """
        执行 Grover 搜索。
        
        参数：
            condition: Oracle 函数 f(x)，返回 True 表示 x 是解
            num_solutions: 已知的解的数量（None 则自动检测）
            shots: 测量次数
        
        返回：
            [(测量结果, 概率), ...]
        
        ==== 完整数学推导 ====
        
        设 M = 解的数量，N = 2^n = 搜索空间大小。
        
        定义两个正交归一态：
            |sol⟩  = 1/√M  Σ_{x 是解}     |x⟩  (解空间的均匀叠加)
            |bad⟩  = 1/√(N-M) Σ_{x 不是解} |x⟩  (非解空间的均匀叠加)
        
        初始均匀叠加态可表示为：
            |ψ₀⟩ = 1/√N Σ_{x} |x⟩
                 = √(N-M)/N |bad⟩ + √(M/N) |sol⟩
                 = cos(θ) |bad⟩ + sin(θ) |sol⟩
        
        其中 sin(θ) = √(M/N)，θ = arcsin(√(M/N))。
        当 M ≪ N 时，θ ≈ √(M/N) ≈ 0，|ψ₀⟩ 几乎全是 |bad⟩。
        
        Oracle 算子：O = I - 2|sol⟩⟨sol|
        (仅在解空间上相位翻转)
        
        扩散算子：D = 2|ψ₀⟩⟨ψ₀| - I
        (关于 |ψ₀⟩ 的反射)
        
        Grover 算子：G = D · O
        
        G 在 {|sol⟩, |bad⟩} 二维子空间中的矩阵表示：
            G|sol⟩ = cos(2θ) |sol⟩ + sin(2θ) |bad⟩
            G|bad⟩ = -sin(2θ) |sol⟩ + cos(2θ) |bad⟩
        
        即 G 是角度 2θ 的旋转。因此：
            G^t |ψ₀⟩ = sin((2t+1)θ) |sol⟩ + cos((2t+1)θ) |bad⟩
        
        期望 sin((2t+1)θ) ≈ 1，即 (2t+1)θ ≈ π/2，
        因此最优迭代次数：
            t_opt = round((π/2 - θ) / (2θ))
                  ≈ round(π/4 · √(N/M))   (当 M ≪ N 时)
        
        测量成功概率：
            P_success = sin²((2·t_opt + 1)θ) ≈ 1 - O(M/N)
        """
        # 步骤 1：构建 Oracle 标记集
        marked = self._build_phase_oracle_mask(condition)
        M = len(marked)

        if M == 0:
            self._log("  ⚠ 搜索空间中没有解！")
            return []

        if num_solutions is not None:
            M = num_solutions

        # 步骤 2：计算最优迭代次数
        theta = math.asin(math.sqrt(M / self.N))
        t_opt = max(1, round((math.pi / 2 - theta) / (2 * theta)))
        success_prob = math.sin((2 * t_opt + 1) * theta) ** 2

        self._log(f"\n{'='*60}")
        self._log(f"  Grover 量子搜索算法")
        self._log(f"{'='*60}")
        self._log(f"  量子比特数 n   = {self.n}")
        self._log(f"  搜索空间 N     = 2^{self.n} = {self.N}")
        self._log(f"  解的数量 M     = {M}")
        self._log(f"  sin²(θ)       = M/N = {M}/{self.N} = {M/self.N:.6f}")
        self._log(f"  θ             = arcsin(√({M/self.N:.6f})) = {theta:.6f} rad")
        self._log(f"  最优迭代 t_opt = round(π/(4θ)) = {t_opt}")
        self._log(f"  理论成功概率   = sin²((2·{t_opt}+1)·{theta:.6f}) = {success_prob:.4%}")
        self._log(f"  经典搜索需要   = O(N) = {self.N} 次查询")
        self._log(f"  量子搜索需要   = O(√N) ≈ {int(math.sqrt(self.N))} 次查询")
        self._log(f"  量子加速倍数   ≈ √N = {math.sqrt(self.N):.1f}x")

        # 步骤 3：初始化量子态
        state = QuantumState(self.n)

        # 步骤 4：创建均匀叠加态
        # H^{⊗n} |0⟩^{⊗n} = 1/√N Σ|x⟩
        for q in range(self.n):
            state.h(q)

        self._log(f"\n  [初始态] 均匀叠加：|ψ₀⟩ = 1/√{self.N} Σ|x⟩")

        # 步骤 5：Grover 迭代
        for t in range(1, t_opt + 1):
            # 5a：Oracle —— 标记解状态
            state.phase_oracle(marked)

            # 5b：扩散算子 D = H^{⊗n} · X^{⊗n} · MCZ · X^{⊗n} · H^{⊗n}
            for q in range(self.n):
                state.h(q)
                state.x(q)
            # MCZ：当所有 n 个量子比特都是 |1⟩ 时相位翻转
            state.mcz(list(range(self.n)))
            for q in range(self.n):
                state.x(q)
                state.h(q)

            # 当前成功振幅
            current_amp = math.sin((2 * t + 1) * theta) ** 2
            self._log(f"  [迭代 {t}/{t_opt}] 当前成功概率 = {current_amp:.4%}")

        # 步骤 6：测量
        results = []
        for shot in range(shots):
            result, prob = state.measure()
            results.append((result, prob))

        # 输出结果
        self._log(f"\n  {'─'*50}")
        self._log(f"  测量结果：")
        for i, (result, prob) in enumerate(results):
            bits = format(result, f'0{self.n}b')
            is_solution = condition(result)
            self._log(f"    Shot {i+1}: |{bits}⟩ (int={result}), "
                      f"概率={prob:.4%}, {'✓ 是解' if is_solution else '✗ 不是解'}")

        # 找到的概率最高的状态
        best, best_prob = state.measure_most_likely()
        best_bits = format(best, f'0{self.n}b')
        self._log(f"\n  最优结果：|{best_bits}⟩ (int={best}), 概率={best_prob:.4%}")
        self._log(f"  {'='*60}\n")

        return results


# ╔══════════════════════════════════════════════════════════════╗
# ║  PART 4: QSL 编译器 — 前提·工具·问题·主函数                      ║
# ╚══════════════════════════════════════════════════════════════╝

@dataclass
class QSLProgram:
    """
    QSL 程序的数据结构。
    
    结构对应四项：
        - premise  (前提): 定义搜索空间约束的布尔表达式列表
        - tools    (工具):  辅助函数/变换
        - question (问题):  搜索目标描述
        - main      (主函数): Grover 搜索配置
    """
    name: str
    n_qubits: int
    premises: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    question: str = ""
    main_algorithm: str = "grover"
    shots: int = 1


class QSLCompiler:
    """
    QSL 编译器：将"前提-工具-问题-主函数"结构编译为量子电路并执行。
    
    编译流程：
        QSL源码 → 布尔表达式AST → 组合Oracle → Grover电路 → 执行 → 结果
        
    这是整个 QSL 语言的核心——将声明式的搜索问题描述
    编译为可执行的 Grover 量子搜索。
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _build_combined_condition(self,
                                   program: QSLProgram) -> Callable[[int], bool]:
        """
        将所有前提条件组合为一个复合 Oracle 函数。
        
        每个前提是一个布尔表达式，所有前提必须同时满足（AND 关系）。
        这定义了搜索子空间 S_P = {x : 所有前提都为真}。
        """
        parsed_premises = [parse_bool(p) for p in program.premises]

        def combined(x: int) -> bool:
            return all(p.evaluate(x) for p in parsed_premises)

        return combined

    def compile_and_run(self, program: QSLProgram) -> List[Tuple[int, float]]:
        """
        编译并执行 QSL 程序。
        
        返回：[(测量结果, 概率), ...]
        """
        if self.verbose:
            self._print_header(program)

        # 步骤 1：解析前提 → 组合 Oracle
        parsed = []
        for p in program.premises:
            expr = parse_bool(p)
            parsed.append(expr)
            if self.verbose:
                print(f"    前提: {p}  →  {expr.to_qasm_style()}")

        # 步骤 2：构建组合条件函数 f(x) = ∧_i premise_i(x)
        def oracle(x: int) -> bool:
            return all(p.evaluate(x) for p in parsed)

        # 步骤 3：验证至少有一个解（经典检查）
        M = sum(1 for x in range(1 << program.n_qubits) if oracle(x))
        if M == 0:
            print("\n  ⚠ 错误：没有状态满足所有前提条件！")
            print("    请检查你的前提表达式是否有矛盾。")
            return []

        if self.verbose:
            self._print_premise_analysis(program.n_qubits, M, parsed)

        # 步骤 4：执行 Grover 搜索
        grover = GroverSearch(program.n_qubits, verbose=self.verbose)
        results = grover.search(
            condition=oracle,
            num_solutions=M,
            shots=program.shots
        )

        return results

    def _print_header(self, program: QSLProgram):
        print(f"\n{'█'*60}")
        print(f"█  QSL 量子搜索程序: {program.name}")
        print(f"█  量子比特数: {program.n_qubits} (搜索空间: 2^{program.n_qubits} = {1<<program.n_qubits})")
        print(f"{'█'*60}")
        print(f"\n  ┌─ 前提 (Premise) ─ 定义搜索约束")
        print(f"  │")

    def _print_premise_analysis(self, n: int, M: int, parsed: list):
        N = 1 << n
        print(f"  │")
        print(f"  ├─ 工具 (Tools) ─ 自动生成组合 Oracle")
        print(f"  │   Oracle(x) = {' ∧ '.join(p.to_qasm_style() for p in parsed)}")
        print(f"  │")
        print(f"  ├─ 问题 (Question) ─ 搜索满足所有前提的赋值")
        print(f"  │   搜索空间: {N} 个状态")
        print(f"  │   满足前提: {M} 个状态 (占 {M/N*100:.2f}%)")
        print(f"  │")
        print(f"  └─ 主函数 (Main) ─ Grover 量子搜索")


# ╔══════════════════════════════════════════════════════════════╗
# ║  PART 5: QSL DSL 解析器 (文本语法 → QSLProgram)               ║
# ╚══════════════════════════════════════════════════════════════╝

def parse_qsl(source: str) -> QSLProgram:
    """
    解析 QSL 领域特定语言源码。
    
    QSL 语法：
    
        program "名称" {
            qubits: N
            
            premise {
                <布尔表达式>
                <布尔表达式>
                ...
            }
            
            tools {
                oracle: combine_all
            }
            
            question {
                find: assignment
                of: [x0, x1, ...]
                where: all premises satisfied
            }
            
            main {
                algorithm: grover
                shots: N
            }
        }
    
    每个 premise 是一个布尔表达式，支持：
        - 变量：x0, x1, x2, ... (索引从 0 开始)
        - 运算符：& (AND), | (OR), ^ (XOR), ~ (NOT)
        - 括号：() 用于分组
    """
    program = None
    current_section = None

    lines = source.strip().split('\n')

    for line in lines:
        # 去除注释
        if '#' in line:
            line = line[:line.index('#')]
        line = line.strip()
        if not line:
            continue

        # 程序声明
        if line.startswith('program '):
            # program "名称" {
            match = re.match(r'program\s+"([^"]*)"\s*\{?', line)
            if match:
                program = QSLProgram(name=match.group(1), n_qubits=0)
                current_section = None
            continue

        # qubits 声明
        if 'qubits:' in line:
            match = re.search(r'qubits\s*:\s*(\d+)', line)
            if match and program:
                program.n_qubits = int(match.group(1))
            continue

        # 段声明
        if 'premise' in line and '{' in line:
            current_section = 'premise'
            continue
        if 'tools' in line and '{' in line:
            current_section = 'tools'
            continue
        if 'question' in line and '{' in line:
            current_section = 'question'
            continue
        if 'main' in line and '{' in line:
            current_section = 'main'
            continue

        # 段结束
        if line == '}':
            current_section = None
            continue

        # 内容
        if program and current_section == 'premise':
            # 跳过以 constraint: 开头的关键字
            content = line
            if content.startswith('constraint:'):
                content = content[len('constraint:'):].strip()
            if content and not content.startswith('//'):
                program.premises.append(content)

        elif program and current_section == 'tools':
            if 'oracle:' in line:
                program.tools.append(line.strip())

        elif program and current_section == 'main':
            if 'algorithm:' in line:
                alg_match = re.search(r'algorithm\s*:\s*(\w+)', line)
                if alg_match:
                    program.main_algorithm = alg_match.group(1)
            if 'shots:' in line:
                shots_match = re.search(r'shots\s*:\s*(\d+)', line)
                if shots_match:
                    program.shots = int(shots_match.group(1))

    if program is None:
        raise ValueError("解析错误：未找到 program 声明")

    return program


# ╔══════════════════════════════════════════════════════════════╗
# ║  PART 6: 示例程序                                            ║
# ╚══════════════════════════════════════════════════════════════╝

def demo_1_sat_solver():
    """
    示例 1：3-SAT 问题求解
    
    问题：找到满足以下子句的布尔赋值 (x0, x1, x2)：
        (x0 ∨ ¬x1) ∧ (x1 ∨ x2) ∧ (¬x0 ∨ ¬x2)
    
    这是一个经典的 3-SAT 实例，有多个解。
    
    数学分析：
        n = 3, N = 2³ = 8
        满足所有子句的赋值数 M = ?
        经典枚举需要检查 8 个状态
        量子搜索约需 √8 ≈ 2.8 次查询
    """
    # 方法 1：使用 Python API
    program = QSLProgram(
        name="3-SAT 求解器",
        n_qubits=3,
        premises=[
            "x0 | ~x1",     # 子句 1：x0 ∨ ¬x1
            "x1 | x2",      # 子句 2：x1 ∨ x2
            "~x0 | ~x2",    # 子句 3：¬x0 ∨ ¬x2
        ],
        question="找到满足所有子句的赋值",
        main_algorithm="grover",
        shots=3
    )

    compiler = QSLCompiler(verbose=True)
    results = compiler.compile_and_run(program)
    return results


def demo_2_sat_with_dsl():
    """
    示例 2：使用 QSL DSL 语法解 3-SAT
    
    展示 QSL 文本语法的使用方式。
    """
    qsl_source = '''
    program "3-SAT 求解器 (DSL)" {
        qubits: 3
        
        premise {
            x0 | ~x1
            x1 | x2
            ~x0 | ~x2
        }
        
        tools {
            oracle: combine_all
        }
        
        question {
            find: assignment
            where: all premises satisfied
        }
        
        main {
            algorithm: grover
            shots: 3
        }
    }
    '''

    program = parse_qsl(qsl_source)
    compiler = QSLCompiler(verbose=True)
    results = compiler.compile_and_run(program)
    return results


def demo_3_exact_one():
    """
    示例 3：恰好一个量子比特为 1
    
    前提：在 3 个量子比特中，恰好有一个为 |1⟩。
    即：(x0 ∧ ¬x1 ∧ ¬x2) ∨ (¬x0 ∧ x1 ∧ ¬x2) ∨ (¬x0 ∧ ¬x1 ∧ x2)
    
    数学分析：
        n = 3, N = 8
        3 个状态中有恰好一个 1：|001⟩, |010⟩, |100⟩
        M = 3, θ = arcsin(√(3/8)) ≈ 0.659 rad
        t_opt ≈ round(π/(4×0.659)) ≈ 1
    """
    # 编码"恰好一个 1"：两两之间至少有一个为 0
    program = QSLProgram(
        name="恰好一个量子比特为 1",
        n_qubits=3,
        premises=[
            "x0 | x1 | x2",       # 至少一个为 1
            "~x0 | ~x1",          # x0 和 x1 不能同时为 1
            "~x0 | ~x2",          # x0 和 x2 不能同时为 1
            "~x1 | ~x2",          # x1 和 x2 不能同时为 1
        ],
        question="找到恰好一个量子比特为 |1⟩ 的状态",
        shots=3
    )

    compiler = QSLCompiler(verbose=True)
    results = compiler.compile_and_run(program)
    return results


def demo_4_graph_coloring():
    """
    示例 4：图着色问题（简化为 3 顶点 2 色）
    
    3 个顶点 A, B, C，边 (A,B) 和 (B,C)，用 2 种颜色。
    每个顶点用 1 个量子比特表示（0=红色, 1=蓝色）。
    约束：相邻顶点必须不同色。
        A ≠ B 即 x0 ≠ x1  →  x0 ^ x1 = 1
        B ≠ C 即 x1 ≠ x2  →  x1 ^ x2 = 1
    
    前提：(x0 ^ x1) & (x1 ^ x2)
    
    数学分析：
        n = 3, N = 8
        解：010 (A=红,B=蓝,C=红) 和 101 (A=蓝,B=红,C=蓝)
        M = 2
    """
    program = QSLProgram(
        name="图着色 (3顶点2色)",
        n_qubits=3,
        premises=[
            "x0 ^ x1",   # A ≠ B
            "x1 ^ x2",   # B ≠ C
        ],
        question="找到所有合法的 2-着色方案",
        shots=3
    )

    compiler = QSLCompiler(verbose=True)
    results = compiler.compile_and_run(program)
    return results


def demo_5_larger_search():
    """
    示例 5：较大搜索空间 (n=6) 的量子搜索
    
    搜索满足以下约束的 6 位二进制数：
        - 前 3 位的奇偶性为偶（即 x0⊕x1⊕x2 = 0）
        - 中 2 位不为全 0
        - 第 6 位 (x5) = 1
    
    这个搜索空间有 64 个状态，经典需要检查最多 64 个。
    量子搜索约需 √64 = 8 次查询。
    
    展示量子加速的实际效果。
    """
    program = QSLProgram(
        name="大空间搜索 (n=6, N=64)",
        n_qubits=6,
        premises=[
            "~(x0 ^ x1 ^ x2)",   # 前3位偶校验 → x0⊕x1⊕x2 = 0 即 ~(异或)
            "x3 | x4",           # 中2位不全为0
            "x5",                # 第6位必须为1
        ],
        question="在大搜索空间中寻找满足约束的状态",
        shots=3
    )

    compiler = QSLCompiler(verbose=True)
    results = compiler.compile_and_run(program)
    return results


def demo_6_number_partition():
    """
    示例 6：数字划分问题（Number Partitioning）
    
    给定集合 S = {1, 2, 3}，是否存在子集 A ⊆ S 使得 sum(A) = sum(S\A)？
    这等价于找到赋值使得 sum(x[i] * S[i]) = total/2。
    
    用 4 个量子比特表示（x0,x1,x2 表示选择，x3 为辅助）。
    
    前提：1·x0 + 2·x1 + 3·x2 = 3
    
    数学分析：
        total = 1+2+3 = 6, target = 3
        n = 4, N = 16
        解：(0,0,1,?) or (1,1,0,?)
        但我们需要等号约束，用布尔表达式表示有限和比较困难
    
    简化：搜索满足 (x0 ∧ ¬x1 ∧ x2) ∨ (¬x0 ∧ x1 ∧ ¬x2) 的状态
    即恰好选 1 和选 3 = 4... 实际 = {3}（值3≠3的一半）
    
    重新设计：选或不选的和为 3
    {1,2}: x0=1, x1=1, x2=0 → sum=3 ✓
    {3}: x0=0, x1=0, x2=1 → sum=3 ✓
    """
    # 用布尔表达式编码 sum = 3 的约束
    # 对于小规模，可以直接枚举所有有效赋值
    program = QSLProgram(
        name="数字划分 (S={1,2,3}, target=3)",
        n_qubits=4,
        premises=[
            # (选1 ∧ 选2 ∧ ¬选3) ∨ (¬选1 ∧ ¬选2 ∧ 选3)
            # 即 (x0 & x1 & ~x2) | (~x0 & ~x1 & x2)
            "(x0 & x1 & ~x2) | (~x0 & ~x1 & x2)",
            # 辅助位x3可以在任意状态
        ],
        question="找到使 sum = 3 的子集",
        shots=5
    )

    compiler = QSLCompiler(verbose=True)
    results = compiler.compile_and_run(program)
    return results


def demo_7_binary_to_quantum():
    """
    示例 7：演示"前提完全转化为二进制"
    
    展示如何将任意的二进制约束编译为量子 Oracle。
    
    前提定义：
        一个 4 位二进制数 x 满足：
        - x 是奇数（x0 = 1）
        - x 在 5 到 12 之间
        - x 的二进制表示中恰好有 2 个 1
    
    这些"人类可读"的前提被编译为布尔表达式，
    再编译为量子 Oracle 的标记集。
    
    这就是 QSL 核心思想的完整展示：
    前提(自然语言) → 布尔表达式 → 量子Oracle → Grover搜索 → 结果
    """
    program = QSLProgram(
        name="二进制约束搜索",
        n_qubits=4,
        premises=[
            "x0",                           # 奇数 (LSB=1)
            "x2 | x3",                      # x ≥ 8 或 x ≥ 4（实际上 ≥ 5）
            # x ≥ 5 (0101) 且 x ≤ 12 (1100)
            # 不等式的布尔编码
            "(x2 & ~x1 & ~x0) | (x2 & x1) | x3",  # x ≥ 5
            "~x3 | ~x2 | ~x1",              # x ≤ 14... 简化
            # 恰好2个1
            # 用 XOR 和 AND 编码... 对于小规模，直接列出
        ],
        question="找到满足二进制约束的所有数",
        shots=3
    )

    # 使用更精确的编码
    program.premises = [
        "x0",  # 奇数
    ]

    compiler = QSLCompiler(verbose=True)

    # 手动构建更精确的 oracle
    def precise_oracle(x: int) -> bool:
        # x 是奇数
        if (x & 1) == 0:
            return False
        # x 在 5 到 12 之间
        if x < 5 or x > 12:
            return False
        # x 恰好有 2 个 1
        if bin(x).count('1') != 2:
            return False
        return True

    grover = GroverSearch(4, verbose=True)
    results = grover.search(precise_oracle, shots=3)
    return results


# ╔══════════════════════════════════════════════════════════════╗
# ║  PART 7: 主程序入口                                          ║
# ╚══════════════════════════════════════════════════════════════╝

def print_banner():
    print(r"""
    ╔══════════════════════════════════════════════════════════╗
    ║     QSL - Quantum Search Language                       ║
    ║     量子搜索语言 v1.0                                    ║
    ║                                                          ║
    ║     基于"前提-工具-问题-主函数"框架                      ║
    ║     Premise → Tools → Question → Main                   ║
    ║                                                          ║
    ║     数学核心：Grover 振幅放大算法                        ║
    ║     |ψ_t⟩ = G^t|ψ₀⟩ = sin((2t+1)θ)|sol⟩ + ...          ║
    ║     O(√N) 量子加速 vs O(N) 经典搜索                     ║
    ╚══════════════════════════════════════════════════════════╝
    """)


def main():
    print_banner()

    demos = [
        ("SAT 求解器", demo_1_sat_solver),
        ("QSL DSL 语法", demo_2_sat_with_dsl),
        ("恰好一个 1", demo_3_exact_one),
        ("图着色", demo_4_graph_coloring),
        ("大空间搜索 (n=6)", demo_5_larger_search),
        ("数字划分", demo_6_number_partition),
        ("二进制约束", demo_7_binary_to_quantum),
    ]

    print("\n  可用示例:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"    {i}. {name}")
    print(f"    0. 运行所有示例")
    print()

    try:
        choice = input("  请选择示例编号 [0-7] (默认 0): ").strip()
        if not choice:
            choice = "0"
        choice = int(choice)
    except (ValueError, EOFError):
        choice = 0

    if choice == 0:
        for name, demo_fn in demos:
            print(f"\n{'~'*60}")
            print(f"  运行示例: {name}")
            print(f"{'~'*60}")
            try:
                demo_fn()
            except Exception as e:
                print(f"  ⚠ 示例执行出错: {e}")
            print()
    elif 1 <= choice <= len(demos):
        name, demo_fn = demos[choice - 1]
        demo_fn()
    else:
        print("  无效选择。")

    print("\n" + "=" * 60)
    print("  QSL 量子搜索语言演示完成。")
    print("  核心思想：将搜索问题编码为量子 Oracle，")
    print("  利用 Grover 算法实现 √N 倍的量子加速。")
    print("=" * 60)


if __name__ == '__main__':
    main()
