"""
QSL 编译器。

将 QSLProgram 编译为量子搜索任务并调度到指定后端执行。

编译流程:
    QSLProgram
      -> 解析布尔表达式
      -> 构建组合 Oracle
      -> 验证解存在性
      -> 委托后端执行 Grover 搜索
      -> 返回 GroverResult

失败模式分析:
    1. 前提矛盾 (无解): 抛出 NoSolutionError
    2. 后端不可用: 抛出 BackendNotAvailableError
    3. 后端执行失败: 抛出 BackendJobError
    4. 布尔表达式语法错误: 抛出 BooleanParseError (由 parser 层)
    5. 量子比特数不足: 前提引用的变量索引 >= n_qubits
"""

import numpy as np
from typing import List, Tuple, Optional

from .program import QSLProgram
from ..core.parser import parse_bool, build_oracle_function, BooleanExpr
from ..core.grover import GroverResult
from ..algorithms.shor import ShorSolver
from ..algorithms.qaoa import QAOA
from ..algorithms.vqe import VQE
from ..backends import get_backend, AbstractBackend
from ..utils.exceptions import (
    NoSolutionError,
    CompilerError,
    ProgramValidationError,
)
from .optimizer import gate_fusion, commutation_optimization, depth_reduction


class AlgorithmResult:
    """
    Unified result wrapper for all quantum algorithms.

    Provides a consistent interface regardless of the underlying algorithm.
    """
    def __init__(self, algorithm: str, data: dict):
        self.algorithm = algorithm
        self._data = data

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(
            f"AlgorithmResult for '{self.algorithm}' has no attribute '{name}'. "
            f"Available: {list(self._data.keys())}"
        )

    def summary(self) -> str:
        if self.algorithm == "grover":
            solutions = self._data.get("solutions", [])
            return f"Grover: found {len(solutions)} solution(s): {solutions}"
        elif self.algorithm == "shor":
            factors = self._data.get("factors", [])
            return f"Shor: factors = {factors}"
        elif self.algorithm == "qaoa":
            energy = self._data.get("optimal_energy", float('nan'))
            return f"QAOA: optimal energy = {energy:.4f}"
        elif self.algorithm == "vqe":
            energy = self._data.get("ground_energy", float('nan'))
            return f"VQE: ground energy = {energy:.4f}"
        return f"AlgorithmResult({self.algorithm})"

    def get_solutions(self):
        """Backward compatibility: return solution list for Grover."""
        return self._data.get("solutions", [])

    def __repr__(self):
        return f"AlgorithmResult(algorithm={self.algorithm!r}, data={self._data!r})"


class QSLCompiler:
    """
    QSL 编译器。

    将声明式的 QSLProgram 编译为可执行的量子搜索并运行。

    使用方式:
        >>> program = QSLProgram(name="test", n_qubits=3, premises=["x0 & x1"])
        >>> compiler = QSLCompiler(backend="simulator")
        >>> result = compiler.compile_and_run(program)
    """

    def __init__(self,
                 backend: str = "simulator",
                 backend_options: Optional[dict] = None,
                 verbose: bool = False):
        """
        初始化编译器。

        参数:
            backend: 后端名称 ("simulator" 或 IBM 后端名)
            backend_options: 传递给后端的额外选项
            verbose: 是否输出详细编译信息

        失败模式:
            - 无效的后端名称: 在首次 compile_and_run 时由 get_backend 报错
        """
        self._backend_name = backend
        self._backend_options = backend_options or {}
        self.verbose = verbose
        self._backend: Optional[AbstractBackend] = None

    # ----------------------------------------------------------------
    # 核心: 编译并执行
    # ----------------------------------------------------------------

    def compile_and_run(self,
                         program: QSLProgram,
                         backend: Optional[str] = None,
                         **run_options) -> GroverResult:
        """
        编译 QSLProgram 并在后端执行。

        参数:
            program: QSL 程序定义
            backend: 覆盖默认后端 (None 则使用初始化时指定的)
            **run_options: 传递给 backend.run() 的额外选项

        返回:
            GroverResult 或其他算法结果

        失败模式:
            - 程序验证失败: 抛出异常
            - 前提无解: 抛出 NoSolutionError
            - 后端错误: 抛出 BackendError
            - 不支持的算法: 抛出 ProgramValidationError
        """
        # 1. 验证程序
        self._validate_program(program)

        algorithm = program.main_algorithm

        if algorithm == "grover":
            return self._run_grover(program, backend, **run_options)
        elif algorithm == "shor":
            result = self._run_shor(program, **run_options)
            return AlgorithmResult("shor", {"factors": result, "N": result[0] if result else 0})
        elif algorithm == "qaoa":
            result = self._run_qaoa(program, **run_options)
            return AlgorithmResult("qaoa", {"params": result[0], "optimal_energy": result[1]})
        elif algorithm == "vqe":
            result = self._run_vqe(program, **run_options)
            return AlgorithmResult("vqe", {"ground_energy": result[0], "ground_state": result[1]})
        else:
            raise ProgramValidationError(
                "main_algorithm", algorithm,
                f"不支持的算法: {algorithm}。支持的算法: grover, shor, qaoa, vqe"
            )

    def _run_grover(self, program: QSLProgram, backend: Optional[str] = None, **run_options):
        """执行 Grover 搜索。

        布尔前提直接传递给后端编译为量子 Oracle 电路, 不在经典侧
        枚举 2^n 个基态。解数量未知, 由后端使用 BBHT 指数搜索
        (期望查询复杂度 O(√(N/M))); 若无解, 后端抛出 NoSolutionError。
        """
        backend_name = backend or self._backend_name
        backend_instance = self._get_backend(backend_name)

        parsed_premises = self._parse_premises(program)
        oracle = build_oracle_function(parsed_premises)

        if self.verbose:
            self._print_compilation_info(program, parsed_premises, None)

        gate_sequence = []
        for premise in program.premises:
            gate_sequence.append({
                'gate': 'ORACLE',
                'targets': list(range(program.n_qubits)),
                'params': {'expression': premise}
            })

        gate_sequence, original_depth, optimized_depth = depth_reduction(
            commutation_optimization(gate_fusion(gate_sequence))
        )

        if self.verbose and gate_sequence:
            print(f"  Circuit optimization: depth reduced from {original_depth} to {optimized_depth}")

        shots = program.shots
        options = {**self._backend_options, **run_options}
        options.pop("oracle_expressions", None)

        result = backend_instance.run_grover_search(
            n_qubits=program.n_qubits,
            oracle=oracle,
            num_solutions=None,
            shots=shots,
            verbose=self.verbose,
            oracle_expressions=parsed_premises,
            **options,
        )

        return result

    def _run_shor(self, program: QSLProgram, **run_options):
        """执行 Shor 因子分解算法。"""
        if self.verbose:
            print(f"\n{'#'*60}")
            print(f"  Shor 算法: {program.name}")
            print(f"{'#'*60}")

        N = run_options.get('N', None)
        
        if N is None:
            import re
            digits = re.findall(r'\b(\d+)\b', program.name)
            if digits:
                N = int(digits[-1])
            else:
                N = 1 << program.n_qubits
        
        if N < 4:
            raise ValueError(
                f"Shor算法需要N >= 4，当前N={N}。"
                f"请通过compiler.run(N=your_number)指定要分解的整数。"
            )

        solver = ShorSolver(N)
        factors = solver.factor()

        if self.verbose:
            print(f"  N = {N}")
            print(f"  因子分解结果: {factors}")
            print(f"  验证: {np.prod(factors) if factors else '无'} = {N}")

        return factors

    def _run_qaoa(self, program: QSLProgram, **run_options):
        """执行 QAOA 算法。"""
        if self.verbose:
            print(f"\n{'#'*60}")
            print(f"  QAOA 算法: {program.name}")
            print(f"{'#'*60}")

        n_qubits = program.n_qubits

        cost_matrix = np.zeros((n_qubits, n_qubits))
        for i, premise in enumerate(program.premises):
            try:
                cost_matrix[i % n_qubits, (i + 1) % n_qubits] = float(premise)
            except ValueError:
                pass

        qaoa = QAOA(n_qubits, cost_matrix)
        params, energy = qaoa.optimize()

        if self.verbose:
            print(f"  量子比特数: {n_qubits}")
            print(f"  最优能量: {energy:.4f}")
            print(f"  最优参数: {params}")

        return params, energy

    def _run_vqe(self, program: QSLProgram, **run_options):
        """执行 VQE 算法。"""
        if self.verbose:
            print(f"\n{'#'*60}")
            print(f"  VQE 算法: {program.name}")
            print(f"{'#'*60}")

        n_qubits = program.n_qubits

        if program.premises:
            molecule = program.premises[0]
        else:
            molecule = "h2"

        if molecule.lower() == "h2":
            hamiltonian = VQE.h2_hamiltonian()
        else:
            raise ProgramValidationError(
                "premises", molecule,
                f"暂不支持分子 '{molecule}'。当前仅内置 H₂ 的 STO-3G "
                f"Pauli 哈密顿量; 其他分子请直接构造 "
                f"hamiltonian_pauli_terms 并使用 qsl.algorithms.VQE。"
            )

        vqe = VQE(n_qubits=min(n_qubits, 4), hamiltonian_pauli_terms=hamiltonian)
        result = vqe.optimize()

        if self.verbose:
            print(f"  量子比特数: {n_qubits}")
            print(f"  分子: {molecule}")
            print(f"  基态能量: {result[0]:.4f}")

        return result

    # ----------------------------------------------------------------
    # 编译分析 (不执行)
    # ----------------------------------------------------------------

    def compile_and_analyze(self, program: QSLProgram) -> dict:
        """
        编译并分析程序，但不执行搜索。

        返回包含编译元数据的字典:
            - n_qubits: 量子比特数
            - search_space_size: 搜索空间大小
            - num_solutions: 解的数量
            - solution_fraction: M/N 比例
            - optimal_iterations: 理论最优迭代次数
            - theoretical_success_prob: 理论成功概率
            - oracle_expressions: 解析后的表达式字符串
            - solution_states: 解的基态列表

        失败模式:
            - 前提无解: 抛出 NoSolutionError
            - 相同失败模式同 compile_and_run
        """
        self._validate_program(program)
        parsed_premises = self._parse_premises(program)
        oracle = build_oracle_function(parsed_premises)

        N = 1 << program.n_qubits
        solutions = [x for x in range(N) if oracle(x)]
        M = len(solutions)

        if M == 0:
            raise NoSolutionError(
                premises=program.premises,
                n_qubits=program.n_qubits
            )

        import math
        # 计算理论参数
        if M > 0 and M < N:
            theta = math.asin(math.sqrt(M / N))
            t_opt = round((math.pi / 2 - theta) / (2 * theta))
            t_opt = max(1, t_opt)
            theory_prob = math.sin((2 * t_opt + 1) * theta) ** 2
        elif M == N:
            theta = math.pi / 2
            t_opt = 0
            theory_prob = 1.0
        else:
            theta = 0.0
            t_opt = 0
            theory_prob = 0.0

        return {
            "n_qubits": program.n_qubits,
            "search_space_size": N,
            "num_solutions": M,
            "solution_fraction": M / N if N > 0 else 0.0,
            "optimal_iterations": t_opt,
            "theta": theta,
            "theoretical_success_prob": theory_prob,
            "oracle_expressions": [
                p.to_string() for p in parsed_premises
            ],
            "solution_states": solutions,
        }

    # ----------------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------------

    def _validate_program(self, program: QSLProgram):
        """验证程序定义。"""
        if not isinstance(program, QSLProgram):
            raise ProgramValidationError(
                "program", program,
                f"必须是 QSLProgram 实例，当前类型: {type(program).__name__}"
            )
        program.validate()

    def _get_backend(self, name: str) -> AbstractBackend:
        """延迟初始化后端 (仅在首次执行时创建)。"""
        if self._backend is None or self._backend_name != name:
            self._backend = get_backend(name, **self._backend_options)
            self._backend_name = name
        return self._backend

    def _parse_premises(self,
                         program: QSLProgram) -> List[BooleanExpr]:
        """解析所有前提表达式。"""
        parsed = []
        for p in program.premises:
            try:
                expr = parse_bool(p)
                parsed.append(expr)
            except Exception as e:
                raise CompilerError(
                    f"解析前提表达式 '{p}' 失败: {e}"
                )
        return parsed

    def _print_compilation_info(self, program: QSLProgram,
                                 parsed: List[BooleanExpr], M):
        """打印编译信息。"""
        N = 1 << program.n_qubits
        print(f"\n{'#'*60}")
        print(f"  QSL 编译: {program.name}")
        print(f"{'#'*60}")
        print(f"  量子比特数: {program.n_qubits} (空间: 2^{program.n_qubits} = {N})")
        print(f"  后端:       {self._backend_name}")
        print(f"  前提:")
        for p, expr in zip(program.premises, parsed):
            print(f"    {p}  ->  {expr.to_string()}")
        if M is not None:
            print(f"  满足前提:   {M}/{N} ({M/N*100:.2f}%)")
        else:
            print(f"  解数量:     未知 (BBHT 指数搜索, 无经典枚举)")
        print()


# ----------------------------------------------------------------
# 便捷函数
# ----------------------------------------------------------------

def compile_and_run(program,
                     backend: str = "simulator",
                     verbose: bool = False,
                     **options):
    """
    编译并执行的一次性便捷函数。

    支持两种输入:
        - ``QSLProgram``: 走 DSL 编译流程 (Grover/Shor/QAOA/VQE),
          等同于 ``QSLCompiler(backend).compile_and_run(program)``。
        - ``QuantumCircuit``: 直接执行电路并返回 ``ExecutionResult``,
          免去先学 DSL 的负担, 降低新用户学习曲线。

    参数:
        program: ``QSLProgram`` 或 ``QuantumCircuit``
        backend: 后端名称 (仅对 ``QSLProgram`` 有意义; 电路走本地模拟器)
        verbose: 是否详细输出
        **options: 额外选项; 对电路则透传给 ``QuantumCircuit.execute()``
                   (如 shots / seed / initial_state)

    返回:
        ``GroverResult`` / ``AlgorithmResult`` (QSLProgram) 或
        ``ExecutionResult`` (QuantumCircuit)

    示例:
        >>> from qsl import QuantumCircuit, compile_and_run
        >>> qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1)
        >>> res = compile_and_run(qc, shots=512)
        >>> res.counts   # {0: ~256, 3: ~256}
    """
    # 避免顶层循环导入, 延迟到调用时引入
    from ..circuit.circuit import QuantumCircuit
    if isinstance(program, QuantumCircuit):
        return program.execute(**options)

    if not isinstance(program, QSLProgram):
        raise ProgramValidationError(
            "program", type(program).__name__,
            "compile_and_run 仅接受 QSLProgram 或 QuantumCircuit"
        )

    compiler = QSLCompiler(backend=backend, verbose=verbose)
    return compiler.compile_and_run(program, **options)


def analyze(program: QSLProgram) -> dict:
    """
    分析 QSL 程序 (不执行搜索)。

    返回编译分析结果字典。

    参数:
        program: QSL 程序

    返回:
        分析结果字典
    """
    compiler = QSLCompiler()
    return compiler.compile_and_analyze(program)
