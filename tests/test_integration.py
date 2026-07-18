"""
集成测试 - 端到端场景。

覆盖:
    - Python API -> 模拟器 完整流程
    - DSL 文本 -> 解析 -> 编译 -> 执行 完整流程
    - 多前提组合搜索
    - 图着色搜索
    - 异常栈完整性
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsl import QSLProgram, QSLCompiler, parse_qsl, compile_and_run, analyze


class TestEndToEnd:
    """端到端测试。"""

    def test_python_api_flow(self):
        """Python API 完整流程: QSLProgram -> QSLCompiler -> GroverResult。"""
        program = QSLProgram(
            name="3-SAT",
            n_qubits=3,
            premises=["x0 | ~x1", "x1 | x2", "~x0 | ~x2"],
            shots=20,
        )
        compiler = QSLCompiler(verbose=False)
        result = compiler.compile_and_run(program)
        assert result.success_count > 0
        solutions = result.get_solutions()
        for s in solutions:
            b0 = (s >> 0) & 1
            b1 = (s >> 1) & 1
            b2 = (s >> 2) & 1
            assert (b0 or not b1)
            assert (b1 or b2)
            assert (not b0 or not b2)

    def test_dsl_flow(self):
        """DSL 文本完整流程: parse_qsl -> QSLCompiler -> GroverResult。"""
        source = '''
        program "Graph Coloring" {
            qubits: 3

            premise {
                x0 ^ x1     // A != B
                x1 ^ x2     // B != C
            }

            main {
                algorithm: grover
                shots: 15
            }
        }
        '''
        program = parse_qsl(source)
        assert program.name == "Graph Coloring"
        compiler = QSLCompiler(verbose=False)
        result = compiler.compile_and_run(program)
        assert result.success_count > 0

    def test_multiple_premises(self):
        """多前提组合: 恰好一个 1。"""
        program = QSLProgram(
            name="Exactly One",
            n_qubits=3,
            premises=[
                "x0 | x1 | x2",      # at least one 1
                "~x0 | ~x1",          # not both 0 and 1
                "~x0 | ~x2",          # not both 0 and 2
                "~x1 | ~x2",          # not both 1 and 2
            ],
            shots=20,
        )
        result = compile_and_run(program, verbose=False)
        solutions = result.get_solutions()
        valid = {1, 2, 4}  # |001>, |010>, |100>
        for s in solutions:
            assert s in valid

    def test_convenience_function_flow(self):
        """便捷函数端到端。"""
        program = QSLProgram(
            name="Quick",
            n_qubits=2,
            premises=["x0 ^ x1"],  # 恰好一个1
            shots=10,
        )
        result = compile_and_run(program, verbose=False)
        solutions = result.get_solutions()
        valid = {1, 2}
        for s in solutions:
            assert s in valid

    def test_analyze_flow(self):
        """分析流程。"""
        program = QSLProgram(
            name="Analyze",
            n_qubits=4,
            premises=["x0 & x1", "x2 | x3"],
            shots=5,
        )
        analysis = analyze(program)
        assert analysis["search_space_size"] == 16
        # x0&x1: bits 0,1 are 1 -> 4 states (3,7,11,15)
        # x2|x3: among those, state 3 (x2=0,x3=0) fails, 7,11,15 pass -> 3
        assert analysis["num_solutions"] == 3
        assert analysis["optimal_iterations"] >= 1

    def test_large_dsl_flow(self):
        """更大规模 DSL 端到端。"""
        source = '''
        program "Large Search" {
            qubits: 6

            premise {
                x5
                x3 | x4
                ~(x0 ^ x1 ^ x2)
            }

            main {
                algorithm: grover
                shots: 5
            }
        }
        '''
        program = parse_qsl(source)
        result = compile_and_run(program, verbose=False)
        assert result.success_count > 0
        # 验证所有解满足 x5=1
        for rst, _, is_sol in result.measurements:
            if is_sol:
                assert ((rst >> 5) & 1) == 1

    def test_exception_chain(self):
        """异常栈完整性: 确保错误信息有上下文。"""
        # 矛盾前提
        program = QSLProgram(
            name="Bad",
            n_qubits=2,
            premises=["x0 & ~x0"],
            shots=5,
        )
        compiler = QSLCompiler(verbose=False)
        try:
            compiler.compile_and_run(program)
            assert False, "应该抛出 NoSolutionError"
        except Exception as e:
            msg = str(e)
            assert "x0 & ~x0" in msg or "解" in msg or "premises" in msg.lower()
