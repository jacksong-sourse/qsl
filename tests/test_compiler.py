"""
测试编译器模块 (QSLProgram, QSLCompiler, DSL)。

覆盖:
    - QSLProgram 验证
    - DSL 解析 (parse_qsl)
    - 编译器编译执行
    - 编译器分析
    - 错误处理
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsl.compiler.program import QSLProgram
from qsl.compiler.compiler import QSLCompiler, compile_and_run, analyze
from qsl.compiler.dsl import parse_qsl
from qsl.utils.exceptions import (
    ProgramValidationError,
    NoSolutionError,
    DSLParseError,
)


class TestQSLProgram:
    """测试 QSLProgram 数据结构。"""

    def test_valid_program(self):
        """合法程序定义。"""
        p = QSLProgram(name="test", n_qubits=3, premises=["x0 & x1"], shots=5)
        assert p.name == "test"
        assert p.n_qubits == 3
        assert len(p.premises) == 1

    def test_invalid_n_qubits(self):
        """无效量子比特数。"""
        try:
            p = QSLProgram(name="test", n_qubits=0, premises=["x0"])
            p.validate()  # 手动触发验证
            assert False
        except Exception:
            pass

    def test_empty_name(self):
        """空名称。"""
        try:
            QSLProgram(name="", n_qubits=2, premises=["x0"])
            assert False
        except (ValueError, ProgramValidationError):
            pass

    def test_invalid_shots(self):
        """无效测量次数。"""
        try:
            QSLProgram(name="test", n_qubits=2, premises=["x0"], shots=0)
            assert False
        except (ProgramValidationError, ValueError):
            pass

    def test_empty_premises_ok(self):
        """空前提合法 (搜索全空间)。"""
        p = QSLProgram(name="test", n_qubits=2)
        assert p.premises == []

    def test_invalid_algorithm(self):
        """不支持的算法。"""
        try:
            QSLProgram(name="test", n_qubits=2, premises=["x0"],
                       main_algorithm="invalid_algo")
            assert False
        except ValueError:
            pass

    def test_to_dict_and_copy(self):
        """to_dict 和 copy_with。"""
        p = QSLProgram(name="test", n_qubits=3, premises=["x0"], shots=10)
        d = p.to_dict()
        assert d["name"] == "test"
        p2 = p.copy_with(name="new_name", shots=20)
        assert p2.name == "new_name"
        assert p2.shots == 20


class TestDSLParser:
    """测试 DSL 文本解析。"""

    def test_basic_dsl(self):
        """基本 DSL 解析。"""
        source = '''
        program "Test" {
            qubits: 3

            premise {
                x0 & x1
                ~x2 | x0
            }

            main {
                algorithm: grover
                shots: 5
            }
        }
        '''
        program = parse_qsl(source)
        assert program.name == "Test"
        assert program.n_qubits == 3
        assert len(program.premises) == 2
        assert program.shots == 5

    def test_dsl_with_comments(self):
        """DSL 含注释。"""
        source = '''
        // 这是一个注释
        program "Comment Test" {
            qubits: 4    # 行尾注释
            premise {
                x0 & x1 & x2
            }
            main {
                algorithm: grover
                shots: 3
            }
        }
        '''
        program = parse_qsl(source)
        assert program.name == "Comment Test"
        assert program.n_qubits == 4

    def test_dsl_missing_program(self):
        """缺少 program 声明。"""
        try:
            parse_qsl("qubits: 3")
            assert False
        except DSLParseError:
            pass

    def test_dsl_missing_qubits(self):
        """缺少 qubits。"""
        source = '''
        program "NoQubits" {
            premise {
                x0
            }
        }
        '''
        try:
            parse_qsl(source)
            assert False
        except DSLParseError:
            pass

    def test_dsl_empty_source(self):
        """空源码。"""
        try:
            parse_qsl("")
            assert False
        except DSLParseError:
            pass

    def test_dsl_with_backend(self):
        """DSL 指定后端。"""
        source = '''
        program "BackendTest" {
            qubits: 3
            premise { x0 & x1 }
            main {
                algorithm: grover
                shots: 3
                backend: simulator
            }
        }
        '''
        program = parse_qsl(source)
        assert program.backend == "simulator"


class TestCompiler:
    """测试编译器。"""

    def test_compile_and_run_simulator(self):
        """在模拟器上编译运行。"""
        program = QSLProgram(
            name="SAT Test",
            n_qubits=3,
            premises=["x0 | ~x1", "x1 | x2", "~x0 | ~x2"],
            shots=10,
        )
        compiler = QSLCompiler(verbose=False)
        result = compiler.compile_and_run(program)
        assert result.success_count > 0
        # BBHT 路径下解数量未知 (num_solutions=None), 不断言具体 M
        assert len(result.get_solutions()) > 0

    def test_compile_and_analyze(self):
        """编译分析不执行。"""
        program = QSLProgram(
            name="Analysis Test",
            n_qubits=3,
            premises=["x0 & x1"],
            shots=5,
        )
        compiler = QSLCompiler(verbose=False)
        analysis = compiler.compile_and_analyze(program)
        assert analysis["n_qubits"] == 3
        assert analysis["num_solutions"] == 2  # x0&x1: |011>|111> = 3,7
        assert analysis["search_space_size"] == 8
        assert "optimal_iterations" in analysis

    def test_compile_no_solution(self):
        """矛盾前提 -> NoSolutionError。"""
        program = QSLProgram(
            name="Impossible",
            n_qubits=2,
            premises=["x0", "~x0"],  # x0 & ~x0 永远为假
            shots=5,
        )
        compiler = QSLCompiler(verbose=False)
        try:
            compiler.compile_and_run(program)
            assert False
        except NoSolutionError:
            pass

    def test_compile_sat_solutions_correct(self):
        """SAT 解的正确性。"""
        program = QSLProgram(
            name="SAT",
            n_qubits=3,
            premises=["x0 | ~x1", "x1 | x2", "~x0 | ~x2"],
            shots=30,
        )
        compiler = QSLCompiler(verbose=False)
        result = compiler.compile_and_run(program)
        # 验证所有找到的解确实满足前提
        for rst, _, is_sol in result.measurements:
            if is_sol:
                # 手动检查
                b0 = (rst >> 0) & 1
                b1 = (rst >> 1) & 1
                b2 = (rst >> 2) & 1
                assert (b0 or not b1)  # x0 | ~x1
                assert (b1 or b2)       # x1 | x2
                assert (not b0 or not b2)  # ~x0 | ~x2

    def test_convenience_function(self):
        """便捷函数 compile_and_run。"""
        program = QSLProgram(
            name="Convenience",
            n_qubits=2,
            premises=["x0 & x1"],
            shots=5,
        )
        result = compile_and_run(program, verbose=False)
        assert 3 in result.get_solutions()  # |11> = 3

    def test_convenience_analyze(self):
        """便捷函数 analyze。"""
        program = QSLProgram(
            name="Analyze",
            n_qubits=2,
            premises=["x0"],
            shots=5,
        )
        analysis = analyze(program)
        assert analysis["num_solutions"] == 2  # |01>|11> = 1,3

    def test_invalid_program_type(self):
        """非法程序类型。"""
        compiler = QSLCompiler(verbose=False)
        try:
            compiler.compile_and_run("not a program")
            assert False
        except (ProgramValidationError, AttributeError, TypeError):
            pass
