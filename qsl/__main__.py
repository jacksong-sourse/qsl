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
    else:
        run_interactive()


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
