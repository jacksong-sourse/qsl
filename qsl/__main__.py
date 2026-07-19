"""
QSL 命令行入口。

使用:
    python -m qsl              # 交互式选择示例
    python -m qsl --demo 1     # 运行指定示例
    python -m qsl --version    # 查看版本
"""

import sys


def main():
    """命令行主入口。"""
    args = sys.argv[1:]

    if not args:
        run_interactive()
        return

    cmd = args[0]
    if cmd in ("-h", "--help"):
        print_help()
    elif cmd in ("-V", "--version"):
        from qsl import __version__
        print(f"qsl-quantum {__version__}")
    elif cmd == "--demo":
        choice = args[1] if len(args) > 1 else None
        run_demo(choice)
    elif cmd == "--file":
        run_file(args[1] if len(args) > 1 else None)
    elif cmd == "--solve":
        run_solve(args[1:])
    elif cmd == "--list-demos":
        _list_ai_demos()
    elif cmd == "--ai-demo":
        _run_ai_demo(args[1] if len(args) > 1 else None)
    else:
        print(f"未知参数: {cmd}")
        print_help()


def _list_ai_demos():
    """列出内置中文 AI 演示模板。"""
    try:
        from qsl.ai.demos import list_demos
        for d in list_demos():
            print(f"  {d['id']:2d}. {d['name']}  —  {d['desc']}")
    except Exception as e:
        print(f"无法加载演示列表: {e}")


def _run_ai_demo(key):
    """运行一个中文 AI 演示。"""
    try:
        from qsl.ai.demos import run_demo
        if key is None:
            _list_ai_demos()
            print("\n用法: python -m qsl --ai-demo <编号>")
            return
        try:
            idx = int(key)
        except ValueError:
            idx = key
        report = run_demo(idx, verbose=True)
        print("\n" + report.to_markdown())
    except Exception as e:
        print(f"演示运行错误: {e}")


def run_file(filepath: str = None):
    """从文件运行 QSL 程序。"""
    if filepath is None:
        print("用法: python -m qsl --file <path.qsl>")
        return
    from qsl import QSLCompiler, parse_qsl
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        program = parse_qsl(content)
        compiler = QSLCompiler(verbose=True)
        result = compiler.compile_and_run(program)
        print(f"\n结果: {result.summary()}")
    except FileNotFoundError:
        print(f"文件未找到: {filepath}")
    except Exception as e:
        print(f"错误: {e}")


def run_solve(args: list):
    """从命令行参数运行 SAT 求解。"""
    if not args:
        print("用法: python -m qsl --solve <n_qubits> <premise1> [premise2] ...")
        print("示例: python -m qsl --solve 3 \"x0 | ~x1\" \"x1 | x2\" \"~x0 | ~x2\"")
        return
    try:
        n_qubits = int(args[0])
        premises = args[1:]
        if not premises:
            print("错误: 至少需要一个前提表达式")
            return
        from qsl import QSLProgram, QSLCompiler
        program = QSLProgram(
            name="命令行SAT求解",
            n_qubits=n_qubits,
            premises=premises,
            shots=10
        )
        compiler = QSLCompiler(verbose=True)
        result = compiler.compile_and_run(program)
        print(f"\n结果: {result.summary()}")
    except ValueError:
        print(f"量子比特数必须是整数: {args[0]}")
    except Exception as e:
        print(f"错误: {e}")


def print_help():
    from qsl import __version__
    print(f"""
QSL - Quantum Search Language v{__version__}
=============================================

用法:
    python -m qsl                    交互式模式
    python -m qsl --demo [N]         运行 Grover 演示 (N=1-4, 0=全部)
    python -m qsl --file <path>      从 .qsl 文件运行
    python -m qsl --solve N <expr>   命令行 SAT 求解
    python -m qsl --list-demos       列出中文 AI 演示模板
    python -m qsl --ai-demo <N>      运行中文 AI 演示
    python -m qsl --version          查看版本
    python -m qsl --help             显示此帮助

Python API 速览:
    >>> from qsl import QuantumCircuit, QuantumState
    >>> qc = QuantumCircuit(2); qc.h(0); qc.cx(0,1)
    >>> counts = qc.measure_all(shots=1000)
    >>> from qsl import solve_sat, GroverSearch, ShorSolver
""")


def run_demo(choice=None):
    from qsl import __version__
    print("\n" + "=" * 60)
    print(f"  QSL - Quantum Search Language v{__version__}")
    print("  Grover 量子搜索演示")
    print("=" * 60)

    from qsl import QSLProgram, QSLCompiler

    demos = {
        "1": ("SAT 求解 (n=3)", QSLProgram(
            name="3-SAT 求解",
            n_qubits=3,
            premises=["x0 | ~x1", "x1 | x2", "~x0 | ~x2"],
            shots=5,
        )),
        "2": ("图着色 (3顶点2色)", QSLProgram(
            name="图着色",
            n_qubits=3,
            premises=["x0 ^ x1", "x1 ^ x2"],
            shots=5,
        )),
        "3": ("恰好一个1 (n=3)", QSLProgram(
            name="恰好一个1",
            n_qubits=3,
            premises=["x0 | x1 | x2", "~x0 | ~x1", "~x0 | ~x2", "~x1 | ~x2"],
            shots=5,
        )),
        "4": ("大空间搜索 (n=6)", QSLProgram(
            name="大空间搜索",
            n_qubits=6,
            premises=["~(x0 ^ x1 ^ x2)", "x3 | x4", "x5"],
            shots=3,
        )),
    }

    compiler = QSLCompiler(verbose=True)

    def _run_one(key):
        desc, program = demos[key]
        print(f"\n{'─'*60}")
        print(f"  示例 {key}: {desc}")
        print(f"{'─'*60}")
        try:
            result = compiler.compile_and_run(program)
            print(f"  结果: {result.summary()}")
        except Exception as e:
            print(f"  错误: {e}")

    if choice is None:
        print("\n可用示例:")
        for key, (desc, _) in demos.items():
            print(f"  {key}. {desc}")
        print(f"  0. 运行所有示例")
        try:
            choice = input("\n请选择 [0-4]: ").strip()
            if not choice:
                choice = "0"
        except (EOFError, KeyboardInterrupt):
            print()
            return

    if choice == "0":
        for key in demos:
            _run_one(key)
    elif choice in demos:
        _run_one(choice)
    else:
        print(f"无效选择: {choice}")
        print(f"可用编号: 0 (全部), {', '.join(demos.keys())}")


def run_interactive():
    from qsl import __version__
    print("\n" + "=" * 60)
    print(f"  QSL - Quantum Search Language v{__version__}")
    print("  Quantum Search Language")
    print("=" * 60)
    print()
    print("  快速开始:")
    print("    python -m qsl --demo       运行 Grover 演示")
    print("    python -m qsl --list-demos 列出中文 AI 演示")
    print("    python -m qsl --help       查看帮助")
    print()


if __name__ == "__main__":
    main()
