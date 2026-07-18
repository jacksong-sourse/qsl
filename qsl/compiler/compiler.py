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

from typing import List, Tuple, Optional

from .program import QSLProgram
from ..core.parser import parse_bool, build_oracle_function, BooleanExpr
from ..core.grover import GroverResult
from ..backends import get_backend, AbstractBackend
from ..utils.exceptions import (
    NoSolutionError,
    CompilerError,
    ProgramValidationError,
)
from .optimizer import gate_fusion, commutation_optimization, depth_reduction


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
            GroverResult

        失败模式:
            - 程序验证失败: 抛出异常
            - 前提无解: 抛出 NoSolutionError
            - 后端错误: 抛出 BackendError
        """
        # 1. 验证程序
        self._validate_program(program)

        # 2. 确定后端
        backend_name = backend or self._backend_name
        backend_instance = self._get_backend(backend_name)

        # 3. 解析前提 -> 组合 Oracle
        parsed_premises = self._parse_premises(program)

        # 4. 构建 Oracle 函数
        oracle = build_oracle_function(parsed_premises)

        # 5. 统计解的数量 (O(N) 经典预检)
        N = 1 << program.n_qubits
        M = sum(1 for x in range(N) if oracle(x))

        if M == 0:
            raise NoSolutionError(
                premises=program.premises,
                n_qubits=program.n_qubits
            )

        if self.verbose:
            self._print_compilation_info(program, parsed_premises, M)

        # 6. Apply circuit optimization passes
        # (Placeholder: future versions will build gate sequence from premises
        #  and run gate_fusion / commutation_optimization / depth_reduction.
        #  Currently QSLProgram.premises are used directly to build oracle.)

        # 7. 在后端执行搜索
        shots = program.shots
        options = {**self._backend_options, **run_options}

        result = backend_instance.run_grover_search(
            n_qubits=program.n_qubits,
            oracle=oracle,
            num_solutions=M,
            shots=shots,
            verbose=self.verbose,
            **options,
        )

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
                                 parsed: List[BooleanExpr], M: int):
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
        print(f"  满足前提:   {M}/{N} ({M/N*100:.2f}%)")
        print()


# ----------------------------------------------------------------
# 便捷函数
# ----------------------------------------------------------------

def compile_and_run(program: QSLProgram,
                     backend: str = "simulator",
                     verbose: bool = False,
                     **options) -> GroverResult:
    """
    编译并执行 QSL 程序的一次性便捷函数。

    等同于:
        >>> compiler = QSLCompiler(backend=backend)
        >>> result = compiler.compile_and_run(program, **options)

    参数:
        program: QSL 程序
        backend: 后端名称
        verbose: 是否详细输出
        **options: 额外后端选项

    返回:
        GroverResult
    """
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
