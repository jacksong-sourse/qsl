"""
QSL 命令行入口。

使用:
    python -m qsl              # 交互式选择示例
    python -m qsl --demo 1     # 运行指定示例
"""

import sys


def main():
    """命令行主入口。"""
    from qsl import QSLProgram, QSLCompiler, parse_qsl

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo()
    elif len(sys.argv) > 1 and sys.argv[1] == "--help":
        print_help()
    elif len(sys.argv) > 1 and sys.argv[1] == "--file":
        run_file(sys.argv[2] if len(sys.argv) > 2 else None)
    elif len(sys.argv) > 1 and sys.argv[1] == "--solve":
        run_solve(sys.argv[2:])
    else:
        run_interactive()


def run_file(filepath: str = None):
    """从文件运行 QSL 程序。"""
    if filepath is None:
        print("用法: python -m qsl --file <path.qsl>")
        return
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
    print("""
QSL - Quantum Search Language
=============================

用法:
    python -m qsl             交互式模式
    python -m qsl --demo 1    运行指定示例
    python -m qsl --help      显示此帮助
""")
    print("""
Python API:
    >>> from qsl import QSLProgram, QSLCompiler
    >>> program = QSLProgram(
    ...     name="我的搜索问题",
    ...     n_qubits=4,
    ...     premises=["x0 & x1", "~x2 | x3"],
    ...     shots=10
    ... )
    >>> compiler = QSLCompiler(backend="simulator")
    >>> result = compiler.compile_and_run(program)
""")


def run_demo():
    from qsl import __version__
    print("\n" + "=" * 60)
    print(f"  QSL - Quantum Search Language v{__version__}")
    print("  Grover 量子搜索演示")
    print("=" * 60)

    from qsl import QSLProgram, QSLCompiler

    # 内嵌示例
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

    compiler = QSLCompiler(verbose=True)

    if choice == "0":
        for key, (desc, program) in demos.items():
            print(f"\n{'─'*60}")
            print(f"  示例 {key}: {desc}")
            print(f"{'─'*60}")
            try:
                result = compiler.compile_and_run(program)
                print(f"  结果: {result.summary()}")
            except Exception as e:
                print(f"  错误: {e}")
    elif choice in demos:
        desc, program = demos[choice]
        print(f"\n{'─'*60}")
        print(f"  示例 {choice}: {desc}")
        print(f"{'─'*60}")
        try:
            result = compiler.compile_and_run(program)
            print(f"  结果: {result.summary()}")
        except Exception as e:
            print(f"  错误: {e}")
    else:
        print(f"无效选择: {choice}")


def run_interactive():
    from qsl import __version__
    print("\n" + "=" * 60)
    print(f"  QSL - Quantum Search Language v{__version__}")
    print("  Quantum Search Language")
    print("=" * 60)
    print("\n用法: python -m qsl --demo    运行演示")
    print("      python -m qsl --help    查看帮助")
    print()


if __name__ == "__main__":
    main()
