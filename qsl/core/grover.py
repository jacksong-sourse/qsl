from __future__ import annotations

"""
Grover 量子搜索算法实现。

算法流程:
    1. 初始化: |psi_0> = H^{⊗n} |0>^{⊗n} = 1/sqrt(N) * sum |x>
    2. 重复 t_opt 次:
       a. Oracle: 对解状态施加 -1 相位
       b. Diffusion: 关于 |psi_0> 的反射
    3. 测量

参数推导:
    设 M 个解, N = 2^n, sin(theta) = sqrt(M/N)
    则 Grover 算子 G 的效果是在 {|sol>, |bad>} 二维子空间中旋转 2*theta
    初始态: |psi_0> = cos(theta)|bad> + sin(theta)|sol>
    迭代 t 次: |psi_t> = sin((2t+1)*theta)|sol> + cos((2t+1)*theta)|bad>
    最优 t: (2t+1)*theta ≈ pi/2 => t_opt = round((pi/2 - theta) / (2*theta))
    成功概率: P = sin^2((2*t_opt+1)*theta)

失败模式分析:
    1. 零解: M = 0, 算法无法放大振幅
    2. 全解: M = N, 初始态就是目标态 (退化情况)
    3. theta ≈ 0: M << N 时最优 t 计算正确
    4. theta ≈ pi/2: M ≈ N 时算法退化为几乎无意义的搜索
    5. 数值不稳定: M 极小时 sin(theta) 可能下溢
    6. t_opt = 0: 当 M > N/2 时 (此时应增加量子比特)
    7. 测量得到非解: 概率小于 1, 多次 shot 可缓解
"""

import math
from typing import Callable, List, Optional, Tuple, Set

from .state import QuantumState, MAX_QUBITS
from ..utils.validation import validate_n_qubits, validate_shots
from ..utils.exceptions import NoSolutionError


class GroverSearch:
    """
    Grover 量子搜索算法。

    封装完整的 Grover 迭代流程，包括:
    - 自动计算最优迭代次数
    - 自动构建 Oracle (从条件函数)
    - 可选 verbose 日志
    """

    def __init__(self, n_qubits: int, verbose: bool = False):
        """
        初始化 Grover 搜索。

        参数:
            n_qubits: 量子比特数 (搜索空间 N = 2^n)
            verbose: 是否输出详细计算日志

        失败模式:
            - n_qubits < 1 或 > MAX_QUBITS: 抛出异常
        """
        validate_n_qubits(n_qubits, MAX_QUBITS)
        self.n = n_qubits
        self.N = 1 << n_qubits
        self.verbose = verbose
        self._search_result: Optional['GroverResult'] = None
        # Cache marked sets by condition id() to avoid O(N) re-enumeration
        self._marked_cache: dict = {}

    # ----------------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------------

    @staticmethod
    def _compute_optimal_iterations(N: int, M: int) -> Tuple[int, float, float]:
        """
        计算 Grover 搜索的最优迭代次数。

        参数:
            N: 搜索空间大小
            M: 解的数量

        返回:
            (t_opt, theta, success_prob)

        失败模式:
            - M = 0: 除零错误, 应由调用方提前处理
            - M >= N: t_opt = 0 (初始态已是最优)
            - 数值不稳定: sqrt(M/N) 可能下溢
        """
        if M <= 0:
            return 0, 0.0, 0.0
        if M >= N:
            return 0, math.pi / 2, 1.0

        # sin(theta) = sqrt(M/N)
        sin_theta = math.sqrt(M / N)
        # 防止浮点精度导致 theta 计算为 0
        if sin_theta < 1e-15:
            sin_theta = 1e-15
        theta = math.asin(sin_theta)

        t_opt = round((math.pi / 2 - theta) / (2 * theta))
        t_opt = max(0, t_opt)

        if t_opt == 0 and M > 0 and M < N:
            ratio = M / N
            if ratio >= 0.5:
                import warnings
                warnings.warn(
                    f"M/N = {ratio:.4f} >= 1/2, 初始态已足够好，"
                    f"Grover无法提供额外加速。当前成功率约{ratio*100:.1f}%"
                )

        success_prob = math.sin((2 * t_opt + 1) * theta) ** 2

        return t_opt, theta, success_prob

    def _build_marked_set(self,
                           condition: Callable[[int], bool]) -> Set[int]:
        """
        从条件函数构建标记集合（带缓存）。

        失败模式:
            - 全集查询 O(N): 对大量子比特数可能很慢
        """
        cond_id = id(condition)
        if cond_id in self._marked_cache:
            return self._marked_cache[cond_id]
        marked = set()
        for x in range(self.N):
            if condition(x):
                marked.add(x)
        self._marked_cache[cond_id] = marked
        return marked

    def _count_solutions(self, condition: Callable[[int], bool]) -> int:
        """统计解的数量 (O(N) 全枚举)。"""
        return sum(1 for x in range(self.N) if condition(x))

    # ----------------------------------------------------------------
    # 搜索执行
    # ----------------------------------------------------------------

    def search(self,
               condition: Callable[[int], bool],
               num_solutions: Optional[int] = None,
               shots: int = 1) -> 'GroverResult':
        """
        执行 Grover 搜索。

        参数:
            condition: Oracle 函数 f(x) -> bool, True 表示 x 是解
            num_solutions: 已知解的数量 (None 则自动统计, O(N) 开销)
            shots: 测量次数

        返回:
            GroverResult 包含搜索的全部结果和元数据

        失败模式:
            - 搜索空间无解: 抛出 NoSolutionError
            - shots < 1: 验证层抛出 ProgramValidationError
            - M = N (全解): t_opt = 0, 直接测量初始态
        """
        validate_shots(shots)

        # 步骤 1: 统计解的数量
        if num_solutions is None:
            M = self._count_solutions(condition)
        else:
            M = num_solutions

        if M == 0:
            raise NoSolutionError(
                premises=["<oracle function>"],
                n_qubits=self.n
            )

        # 步骤 2: 计算最优迭代
        t_opt, theta, theory_prob = self._compute_optimal_iterations(self.N, M)

        if self.verbose:
            self._log_param_summary(M, t_opt, theta, theory_prob)

        # 步骤 3: 初始均匀叠加态
        state = QuantumState(self.n)
        for q in range(self.n):
            state.h(q)

        # 步骤 4: 构建标记集并执行 Grover 迭代
        marked = self._build_marked_set(condition) if t_opt > 0 else set()

        for it in range(1, t_opt + 1):
            state.phase_oracle(marked)
            state.diffusion_operator()
            if self.verbose:
                current_prob = math.sin((2 * it + 1) * theta) ** 2
                print(f"  [迭代 {it}/{t_opt}] 成功概率 = {current_prob:.4%}")

        # 步骤 5: 测量
        results = []
        for shot in range(shots):
            result_int, prob = state.measure()
            is_solution = condition(result_int)
            results.append((result_int, prob, is_solution))

        result = GroverResult(
            n_qubits=self.n,
            num_solutions=M,
            iterations=t_opt,
            theta=theta,
            theory_success_prob=theory_prob,
            measurements=results,
        )

        self._search_result = result

        if self.verbose:
            self._log_results(result)

        return result

    def search_with_oracle_set(self,
                                marked_states: Set[int],
                                shots: int = 1) -> 'GroverResult':
        """
        使用预标记的状态集合执行 Grover 搜索。

        这是 search() 的高效变体，跳过条件函数枚举步骤。

        参数:
            marked_states: 解的基态索引集合
            shots: 测量次数

        返回:
            GroverResult

        失败模式:
            - marked_states 为空: 抛出 NoSolutionError
            - 标记状态越界: 静默忽略
        """
        # 过滤越界的标记
        valid_marked = {x for x in marked_states if 0 <= x < self.N}
        M = len(valid_marked)

        if M == 0:
            raise NoSolutionError(
                premises=["<marked set>"],
                n_qubits=self.n
            )

        validate_shots(shots)

        t_opt, theta, theory_prob = self._compute_optimal_iterations(self.N, M)

        if self.verbose:
            self._log_param_summary(M, t_opt, theta, theory_prob)

        state = QuantumState(self.n)
        for q in range(self.n):
            state.h(q)

        for it in range(1, t_opt + 1):
            state.phase_oracle(valid_marked)
            state.diffusion_operator()
            if self.verbose:
                current_prob = math.sin((2 * it + 1) * theta) ** 2
                print(f"  [迭代 {it}/{t_opt}] 成功概率 = {current_prob:.4%}")

        results = []
        for shot in range(shots):
            result_int, prob = state.measure()
            is_solution = result_int in valid_marked
            results.append((result_int, prob, is_solution))

        result = GroverResult(
            n_qubits=self.n,
            num_solutions=M,
            iterations=t_opt,
            theta=theta,
            theory_success_prob=theory_prob,
            measurements=results,
        )

        self._search_result = result

        if self.verbose:
            self._log_results(result)

        return result

    # ----------------------------------------------------------------
    # 日志方法
    # ----------------------------------------------------------------

    def _log_param_summary(self, M: int, t_opt: int,
                            theta: float, theory_prob: float):
        print(f"\n{'='*60}")
        print(f"  Grover 量子搜索")
        print(f"{'='*60}")
        print(f"  量子比特数 n   = {self.n}")
        print(f"  搜索空间 N     = 2^{self.n} = {self.N}")
        print(f"  解的数量 M     = {M}")
        print(f"  sin^2(theta)  = M/N = {M}/{self.N} = {M/self.N:.6f}")
        print(f"  theta          = arcsin(sqrt({M/self.N:.6f})) = {theta:.6f} rad")
        print(f"  最优迭代 t_opt = {t_opt}")
        print(f"  理论成功概率    = {theory_prob:.4%}")
        print(f"  经典查询次数    = O({self.N})")
        print(f"  量子查询次数    = O({int(math.sqrt(self.N))})")
        print(f"  加速比          ≈ sqrt(N) = {math.sqrt(self.N):.1f}x")

    def _log_results(self, result: 'GroverResult'):
        print(f"\n  {'─'*50}")
        print(f"  测量结果:")
        for i, (bits_int, prob, is_sol) in enumerate(result.measurements):
            bits = format(bits_int, f'0{self.n}b')
            status = "V" if is_sol else "X"
            print(f"    Shot {i+1}: |{bits}> (int={bits_int}), "
                  f"p={prob:.4%} [{status}]")
        best_int, best_prob = result.best_measurement()
        best_bits = format(best_int, f'0{self.n}b')
        print(f"\n  最优: |{best_bits}> (int={best_int}), p={best_prob:.4%}")
        print(f"  {'='*60}\n")


# ----------------------------------------------------------------
# 搜索结果数据结构
# ----------------------------------------------------------------

class GroverResult:
    """
    Grover 搜索的完整结果。

    属性:
        n_qubits: 量子比特数
        num_solutions: 解的数量 M
        iterations: 实际执行的 Grover 迭代次数
        theta: 理论角度 (弧度)
        theory_success_prob: 理论成功概率
        measurements: [(结果整数, 概率, 是否为解), ...]
    """

    def __init__(self,
                 n_qubits: int,
                 num_solutions: int,
                 iterations: int,
                 theta: float,
                 theory_success_prob: float,
                 measurements: List[Tuple[int, float, bool]]):
        self.n_qubits = n_qubits
        self.num_solutions = num_solutions
        self.iterations = iterations
        self.theta = theta
        self.theory_success_prob = theory_success_prob
        self.measurements = measurements

    @property
    def success_count(self) -> int:
        """成功测量到解的次数。"""
        return sum(1 for _, _, is_sol in self.measurements if is_sol)

    @property
    def empirical_success_rate(self) -> float:
        """经验成功概率 (= 成功次数 / 总测量次数)。"""
        if not self.measurements:
            return 0.0
        return self.success_count / len(self.measurements)

    @property
    def shots(self) -> int:
        """总测量次数。"""
        return len(self.measurements)

    def best_measurement(self) -> Tuple[int, float]:
        """
        返回概率最大的测量结果。

        返回:
            (基态整数, 概率)
        """
        if not self.measurements:
            return 0, 0.0
        best = max(self.measurements, key=lambda m: m[1])
        return best[0], best[1]

    def get_solutions(self) -> List[int]:
        """
        返回所有测量到的解。

        返回:
            去重后的解列表
        """
        return list(set(
            r[0] for r in self.measurements if r[2]
        ))

    def get_measurement_counts(self) -> dict:
        """
        返回测量结果的计数分布。

        返回:
            {基态整数: 出现次数}
        """
        counts = {}
        for result_int, _, _ in self.measurements:
            counts[result_int] = counts.get(result_int, 0) + 1
        return counts

    def summary(self) -> str:
        """生成单行结果摘要。"""
        solutions = self.get_solutions()
        return (
            f"GroverResult(n={self.n_qubits}, M={self.num_solutions}, "
            f"iter={self.iterations}, "
            f"theory_P={self.theory_success_prob:.4%}, "
            f"empirical_P={self.empirical_success_rate:.4%}, "
            f"found={solutions})"
        )

    def __repr__(self) -> str:
        return self.summary()


# ----------------------------------------------------------------
# SAT Solver (extends GroverSearch)
# ----------------------------------------------------------------

def solve_sat(cnf_clauses: list[list[int]],
              n_qubits: int,
              shots: int = 100,
              verbose: bool = False) -> GroverResult:
    """
    Solve a SAT problem in CNF form using Grover's search.

    Each clause is a list of literals where positive integers
    represent variables and negative integers represent negated variables.

    Example:
        cnf_clauses = [[1, -2, 3], [-1, 2], [-3]]
        means (x1 OR NOT x2 OR x3) AND (NOT x1 OR x2) AND (NOT x3)

    Args:
        cnf_clauses: List of clauses, each clause is a list of literals
        n_qubits: Number of Boolean variables
        shots: Number of measurements
        verbose: Whether to print detailed progress

    Returns:
        GroverResult containing the search results
    """
    N = 1 << n_qubits

    def _eval_literal(assignment: int, literal: int) -> bool:
        """Evaluate a single literal against a bit assignment."""
        var_idx = abs(literal) - 1
        if var_idx >= n_qubits:
            raise ValueError(f"CNF 变量索引 {literal} 超出 n_qubits={n_qubits}")
        bit_val = bool((assignment >> var_idx) & 1)
        return bit_val if literal > 0 else not bit_val

    def _eval_clause(assignment: int, clause: list[int]) -> bool:
        """Evaluate one clause: OR of its literals."""
        return any(_eval_literal(assignment, lit) for lit in clause)

    def oracle(assignment: int) -> bool:
        """SAT oracle: AND of all clauses."""
        return all(_eval_clause(assignment, clause) for clause in cnf_clauses)

    # Count solutions O(N)
    M = sum(1 for x in range(N) if oracle(x))

    if M == 0:
        raise NoSolutionError(
            premises=[f"CNF: {cnf_clauses}"],
            n_qubits=n_qubits
        )

    search = GroverSearch(n_qubits, verbose=verbose)
    return search.search(condition=oracle, num_solutions=M, shots=shots)
