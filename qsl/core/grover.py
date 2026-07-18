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
import random
from typing import Callable, List, Optional, Tuple, Set

from .state import QuantumState, MAX_QUBITS
from .oracle import OracleCircuit, compile_phase_oracle
from .parser import BooleanExpr, VarExpr, NotExpr, OrExpr, AndExpr
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
    # 量子 Oracle 电路 (布尔表达式直接编译, 无经典枚举)
    # ----------------------------------------------------------------

    @staticmethod
    def _apply_oracle_circuit(state: QuantumState, circ: OracleCircuit):
        """将编译后的相位 Oracle 电路作用到量子态上。"""
        for gate, qs in circ.gates:
            if gate == "X":
                state.x(qs[0])
            elif gate == "Z":
                state.z(qs[0])
            elif gate == "CNOT":
                state.cnot(qs[0], qs[1])
            elif gate == "TOFFOLI":
                state.toffoli(qs[0], qs[1], qs[2])
            else:
                raise ValueError(f"未知的 Oracle 门: {gate}")

    @staticmethod
    def _apply_diffusion_main(state: QuantumState, n_main: int):
        """仅在主寄存器上应用 Grover 扩散算子 (ancilla 不受影响)。"""
        qs = list(range(n_main))
        for q in qs:
            state.h(q)
            state.x(q)
        state.mcz(qs)
        for q in qs:
            state.x(q)
            state.h(q)

    def _run_grover_iterations(self, iterations: int,
                                circ: OracleCircuit) -> QuantumState:
        """制备叠加态并执行指定次数的 Grover 迭代 (电路 Oracle)。"""
        total_qubits = self.n + circ.n_ancilla
        state = QuantumState(total_qubits)
        for q in range(self.n):
            state.h(q)
        for _ in range(iterations):
            self._apply_oracle_circuit(state, circ)
            self._apply_diffusion_main(state, self.n)
        return state

    def search_expressions(self,
                           expressions: List[BooleanExpr],
                           num_solutions: Optional[int] = None,
                           shots: int = 1) -> 'GroverResult':
        """
        使用量子 Oracle 电路执行 Grover 搜索 (无经典枚举)。

        Oracle 由布尔表达式 AST 直接编译为 X/CNOT/Toffoli/Z 可逆电路,
        不遍历 2^n 个基态。当 num_solutions 未知 (None) 时,
        使用 BBHT 指数搜索自动确定迭代次数, 期望查询复杂度 O(√(N/M)),
        无需任何 O(N) 经典预处理。

        参数:
            expressions: BooleanExpr 列表 (逻辑与关系)
            num_solutions: 已知解数量 (None 则使用 BBHT 搜索)
            shots: 测量次数

        返回:
            GroverResult (BBHT 路径下 num_solutions 为 None,
            quantum_queries 记录实际 Oracle 查询次数)

        失败模式:
            - 无解: BBHT 耗尽预算后抛出 NoSolutionError
            - 表达式变量索引越界: 抛出 ValueError
        """
        validate_shots(shots)
        if not expressions:
            raise ValueError("expressions 不能为空")

        circ = compile_phase_oracle(expressions, self.n)
        total_qubits = self.n + circ.n_ancilla
        if total_qubits > MAX_QUBITS:
            raise ValueError(
                f"表达式需要 {circ.n_ancilla} 个 ancilla, 总量子比特数 "
                f"{total_qubits} 超过模拟上限 {MAX_QUBITS}"
            )

        def _is_solution(x: int) -> bool:
            return all(e.evaluate(x) for e in expressions)

        if num_solutions is not None:
            # 已知 M: 直接计算最优迭代次数
            M = num_solutions
            if M == 0:
                raise NoSolutionError(
                    premises=[e.to_string() for e in expressions],
                    n_qubits=self.n,
                )
            t_opt, theta, theory_prob = self._compute_optimal_iterations(
                self.N, M)
            state = self._run_grover_iterations(t_opt, circ)
            measurements = self._measure_state(state, shots, _is_solution)
            result = GroverResult(
                n_qubits=self.n,
                num_solutions=M,
                iterations=t_opt,
                theta=theta,
                theory_success_prob=theory_prob,
                measurements=measurements,
                quantum_queries=t_opt,
            )
        else:
            # 未知 M: BBHT 指数搜索
            measurements, queries = self._bbht_search(
                circ, _is_solution, shots, expressions)
            result = GroverResult(
                n_qubits=self.n,
                num_solutions=None,
                iterations=None,
                theta=None,
                theory_success_prob=None,
                measurements=measurements,
                quantum_queries=queries,
            )

        self._search_result = result
        if self.verbose:
            self._log_results(result)
        return result

    def _bbht_search(self, circ: OracleCircuit,
                     is_solution: Callable[[int], bool],
                     shots: int,
                     expressions: List[BooleanExpr],
                     lam: float = 1.34) -> Tuple[List[Tuple[int, float, bool]], int]:
        """
        BBHT 指数搜索 (Boyer-Brassard-Høyer-Tapp), 用于解数量未知的情形。

        每轮从 [0, m) 均匀随机选择迭代次数 t, 执行 t 次 Grover 迭代后测量;
        若找到解则停止, 否则将 m 乘以因子 lam 继续, 直到 m 超过 √N。
        期望 Oracle 查询次数为 O(√(N/M)), 无需预先知道 M。

        返回:
            (measurements, total_oracle_queries)
        """
        sqrt_N = math.sqrt(self.N)
        m = 1.0
        total_queries = 0

        while m <= sqrt_N * lam:
            t = random.randint(0, max(0, int(m)))
            state = self._run_grover_iterations(t, circ)
            total_queries += t

            result_int, prob = state.measure()
            result_int &= (self.N - 1)  # 屏蔽 ancilla 位 (ancilla 已逆计算为 |0>)

            if is_solution(result_int):
                # 找到解: 在同一放大后的态上完成剩余 shots
                measurements = [(result_int, prob, True)]
                if shots > 1:
                    measurements.extend(
                        self._measure_state(state, shots - 1, is_solution))
                return measurements, total_queries

            m = min(m * lam, sqrt_N * lam + 1.0)

        raise NoSolutionError(
            premises=[e.to_string() for e in expressions],
            n_qubits=self.n,
        )

    def _measure_state(self, state: QuantumState, shots: int,
                       is_solution: Callable[[int], bool]
                       ) -> List[Tuple[int, float, bool]]:
        """对量子态测量 shots 次, 屏蔽 ancilla 位并验证解。"""
        measurements = []
        for _ in range(shots):
            result_int, prob = state.measure()
            result_int &= (self.N - 1)
            measurements.append((result_int, prob, is_solution(result_int)))
        return measurements

    # ----------------------------------------------------------------
    # 搜索执行
    # ----------------------------------------------------------------

    def search(self,
               condition: Callable[[int], bool],
               num_solutions: Optional[int] = None,
               shots: int = 1) -> 'GroverResult':
        """
        执行 Grover 搜索 (黑盒 Oracle 路径)。

        注意: 任意 Python 可调用对象无法编译为量子电路 (即使真实量子
        计算机也必须先将其表达为布尔电路), 因此本路径在模拟器中通过
        一次性的 O(N) 标记集构建来模拟 Oracle —— 这是经典模拟的固有
        开销, 而非算法的查询复杂度 (Grover 迭代仍为 O(√(N/M)) 次
        Oracle 查询)。若条件可表达为布尔表达式, 请使用
        search_expressions(), 它直接从 AST 编译 Oracle 电路,
        不做任何经典枚举。

        参数:
            condition: Oracle 函数 f(x) -> bool, True 表示 x 是解
            num_solutions: 已知解的数量 (None 则自动统计, O(N) 模拟开销)
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
            quantum_queries=t_opt,
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
            quantum_queries=t_opt,
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
        num_solutions: 解的数量 M (BBHT 路径下未知, 为 None)
        iterations: 实际执行的 Grover 迭代次数 (BBHT 路径为 None)
        theta: 理论角度 (弧度, BBHT 路径为 None)
        theory_success_prob: 理论成功概率 (BBHT 路径为 None)
        measurements: [(结果整数, 概率, 是否为解), ...]
        quantum_queries: 实际 Oracle 查询次数 (量子查询复杂度)
    """

    def __init__(self,
                 n_qubits: int,
                 num_solutions: Optional[int],
                 iterations: Optional[int],
                 theta: Optional[float],
                 theory_success_prob: Optional[float],
                 measurements: List[Tuple[int, float, bool]],
                 quantum_queries: Optional[int] = None):
        self.n_qubits = n_qubits
        self.num_solutions = num_solutions
        self.iterations = iterations
        self.theta = theta
        self.theory_success_prob = theory_success_prob
        self.measurements = measurements
        self.quantum_queries = quantum_queries

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
        theory = (f"{self.theory_success_prob:.4%}"
                  if self.theory_success_prob is not None else "N/A")
        return (
            f"GroverResult(n={self.n_qubits}, M={self.num_solutions}, "
            f"iter={self.iterations}, "
            f"theory_P={theory}, "
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

    The CNF formula is compiled directly into a quantum phase-oracle
    circuit (X/CNOT/Toffoli/Z gates over ancilla qubits) — no classical
    enumeration of the 2^n assignments is performed. Since the number
    of solutions M is unknown, BBHT exponential search is used, giving
    an expected O(√(N/M)) oracle query complexity.

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
    if not cnf_clauses:
        raise ValueError("cnf_clauses 不能为空")

    # CNF -> BooleanExpr AST: 每个子句是文字的 OR, 整体是子句的 AND
    expressions: List[BooleanExpr] = []
    for clause in cnf_clauses:
        if not clause:
            raise ValueError("CNF 中存在空子句 (永假)")
        clause_expr: Optional[BooleanExpr] = None
        for literal in clause:
            var_idx = abs(literal) - 1
            if var_idx >= n_qubits:
                raise ValueError(
                    f"CNF 变量索引 {literal} 超出 n_qubits={n_qubits}")
            lit_expr: BooleanExpr = VarExpr(var_idx)
            if literal < 0:
                lit_expr = NotExpr(lit_expr)
            clause_expr = (lit_expr if clause_expr is None
                           else OrExpr(clause_expr, lit_expr))
        expressions.append(clause_expr)

    search = GroverSearch(n_qubits, verbose=verbose)
    return search.search_expressions(
        expressions=expressions, num_solutions=None, shots=shots)
