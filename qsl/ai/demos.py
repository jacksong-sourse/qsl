"""
免 API key 的中文演示模板。

10 个可离线运行的端到端演示, 每个演示:
    1. 构造一个小规模量子问题
    2. 用量子算法求解 (Shor / Grover / QAOA / 直接模拟)
    3. 用 qsl.ai.verifier 做经典交叉验证
    4. 用 qsl.ai.report 生成中文 Markdown 报告

注册表:
    DEMOS: {名称: demo 函数}
    list_demos() -> [(名称, 中文简介), ...]
    run_demo(name) -> {"task", "algorithm", "result", "verified", "report_markdown"}
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

import numpy as np

from .report import AgentReport
from .verifier import (
    VerificationResult,
    verify_grover,
    verify_qaoa,
    verify_sat,
    verify_shor,
)


def _finalize(task: str,
              algorithm: str,
              reason: str,
              circuit_text: str,
              result_summary: str,
              verification: VerificationResult,
              result_payload,
              decision_chain: List[dict],
              verbose: bool) -> dict:
    """汇总演示产物: 生成报告并返回统一字典。"""
    report = AgentReport(
        task=task,
        algorithm=algorithm,
        algorithm_reason=reason,
        backend="simulator",
        circuit_text=circuit_text,
        result_summary=result_summary,
        verification=verification,
        iterations=len(decision_chain),
        decision_chain=decision_chain,
    )
    out = {
        "task": task,
        "algorithm": algorithm,
        "result": result_payload,
        "verified": bool(verification.passed),
        "report_markdown": report.to_markdown(),
    }
    if verbose:
        icon = "✅" if verification.passed else "❌"
        print(f"[{algorithm}] {task}")
        print(f"  结果: {result_summary}")
        print(f"  验证: {icon} {verification.message}")
    return out


# ----------------------------------------------------------------
# 1. Shor 整数分解
# ----------------------------------------------------------------

def demo_factor(verbose: bool = True) -> dict:
    """分解整数 15 (Shor 算法 + 回乘/素性校验)"""
    from ..algorithms.shor import ShorSolver

    random.seed(42)
    np.random.seed(42)
    N = 15
    verification = None
    factors: List[int] = []
    attempts = 0
    for attempts in range(1, 4):
        factors = ShorSolver(N).factor()
        verification = verify_shor(N, factors)
        if verification.passed:
            break

    chain = [
        {"round": 1, "action": "解析任务: 整数分解 N=15", "outcome": "选择 Shor"},
        {"round": 2, "action": f"量子周期查找 (尝试 {attempts} 轮)",
         "outcome": f"因子 {sorted(factors)}"},
        {"round": 3, "action": "经典回乘校验 p*q==N 与素性",
         "outcome": "通过" if verification.passed else "失败"},
    ]
    circuit_text = (
        "Shor(N=15): 控制寄存器 6 比特 (均匀叠加) + 目标寄存器 3 比特 |1>\n"
        "  受控模幂 U_a^{2^k} (稀疏精确模拟) -> 逆 QFT -> 连分数恢复周期 r\n"
        "  经典后处理: gcd(a^{r/2} ± 1, N)"
    )
    return _finalize(
        task="分解整数 15，找到它的素因子",
        algorithm="shor",
        reason="整数分解是 Shor 算法的标志性应用 (RSA 破解原理)",
        circuit_text=circuit_text,
        result_summary=f"15 = {' * '.join(map(str, sorted(factors)))}",
        verification=verification,
        result_payload={"N": N, "factors": sorted(factors)},
        decision_chain=chain,
        verbose=verbose,
    )


# ----------------------------------------------------------------
# 2. 3-SAT 求解 (QSLProgram + Grover)
# ----------------------------------------------------------------

def demo_sat(verbose: bool = True) -> dict:
    """3-SAT 布尔可满足性求解 (Grover/BBHT + 子句代回校验)"""
    from ..compiler.compiler import QSLCompiler
    from ..compiler.program import QSLProgram

    random.seed(7)
    np.random.seed(7)
    premises = ["x0 | x1", "~x0 | x1", "x0 | ~x2"]
    solutions: List[int] = []
    verification = None
    assignment: Dict[str, bool] = {}
    result = None
    for _ in range(3):
        prog = QSLProgram(
            name="3-SAT 求解演示", n_qubits=3, premises=premises, shots=20)
        result = QSLCompiler(backend="simulator").compile_and_run(prog)
        solutions = result.get_solutions()
        if solutions:
            sol = solutions[0]
            assignment = {f"x{i}": bool((sol >> i) & 1) for i in range(3)}
            verification = verify_sat(premises, assignment)
            if verification.passed:
                break

    chain = [
        {"round": 1, "action": "解析任务: SAT 约束满足", "outcome": "选择 Grover"},
        {"round": 2, "action": "布尔子句编译为相位 Oracle 电路",
         "outcome": f"{len(premises)} 个子句, BBHT 搜索"},
        {"round": 3, "action": "代回全部子句逐一求值",
         "outcome": "通过" if verification.passed else "失败"},
    ]
    circuit_text = "premises:\n" + "\n".join(f"  {p}" for p in premises)
    return _finalize(
        task="求解 3 变量 SAT 问题 (3 个子句)",
        algorithm="grover",
        reason="SAT 是典型 NP 完全搜索问题, Grover 提供平方级加速",
        circuit_text=circuit_text,
        result_summary=f"满足赋值: {assignment} (x2x1x0 = "
                       f"{format(solutions[0] if solutions else 0, '03b')})",
        verification=verification,
        result_payload={"solutions": solutions, "assignment": assignment},
        decision_chain=chain,
        verbose=verbose,
    )


# ----------------------------------------------------------------
# 3. 2x2 迷你数独
# ----------------------------------------------------------------

def demo_sudoku(verbose: bool = True) -> dict:
    """2x2 迷你数独 (SAT 编码 + Grover 搜索)"""
    from ..compiler.compiler import QSLCompiler
    from ..compiler.program import QSLProgram

    random.seed(11)
    np.random.seed(11)
    # 格子 a b / c d, 每个取 {1,2}, 用 1 个比特表示 (0->1, 1->2)
    # 约束: 每行/每列两格不同 +  clue: a = 1
    premises = ["x0 ^ x1", "x2 ^ x3", "x0 ^ x2", "x1 ^ x3", "x0"]
    solutions: List[int] = []
    verification = None
    assignment: Dict[str, bool] = {}
    for _ in range(3):
        prog = QSLProgram(
            name="2x2 迷你数独", n_qubits=4, premises=premises, shots=10)
        result = QSLCompiler(backend="simulator").compile_and_run(prog)
        solutions = result.get_solutions()
        if solutions:
            sol = solutions[0]
            assignment = {f"x{i}": bool((sol >> i) & 1) for i in range(4)}
            verification = verify_sat(premises, assignment)
            if verification.passed:
                break

    sol = solutions[0] if solutions else 0
    grid = [[(sol >> 0) & 1, (sol >> 1) & 1], [(sol >> 2) & 1, (sol >> 3) & 1]]
    grid_text = f"[{grid[0][0]+1} {grid[0][1]+1}]\n[{grid[1][0]+1} {grid[1][1]+1}]"
    chain = [
        {"round": 1, "action": "数独 -> SAT 编码 (4 比特, 5 子句)",
         "outcome": "选择 Grover"},
        {"round": 2, "action": "Grover 搜索唯一解", "outcome": f"解 {sol}"},
        {"round": 3, "action": "行/列/宫约束代回校验",
         "outcome": "通过" if verification.passed else "失败"},
    ]
    return _finalize(
        task="求解 2x2 迷你数独 (提示数: 左上 = 1)",
        algorithm="grover",
        reason="数独是约束满足问题, 编码为 SAT 后可用 Grover 平方加速",
        circuit_text="premises:\n" + "\n".join(f"  {p}" for p in premises),
        result_summary=f"数独解:\n{grid_text}",
        verification=verification,
        result_payload={"grid": grid, "solution_int": sol},
        decision_chain=chain,
        verbose=verbose,
    )


# ----------------------------------------------------------------
# 4. MaxCut (QAOA)
# ----------------------------------------------------------------

def demo_maxcut(verbose: bool = True) -> dict:
    """4 节点环图最大割 (QAOA + 全枚举基线校验)"""
    from ..algorithms.qaoa import QAOA

    np.random.seed(3)
    # 4 节点环 C4: 边 (0-1), (1-2), (2-3), (3-0)
    adj = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ], dtype=float)

    verification = None
    best, best_cost = 0, 0.0
    energy = float("nan")
    attempts = 0
    for attempts in range(1, 5):
        qaoa = QAOA(4, adj, p=2, encoding="ising")
        _, energy = qaoa.optimize(maxiter=200)
        best, best_cost = qaoa.get_optimal_bitstring()
        verification = verify_qaoa(adj, best, encoding="ising")
        if verification.passed:
            break

    n_edges = int(adj.sum() / 2)
    # cost = sum_{edges} s_i s_j -> 割边数 = (E - cost) / 2
    cut_value = int(round((n_edges - best_cost) / 2))
    bits = format(best, "04b")
    chain = [
        {"round": 1, "action": "MaxCut -> Ising 代价哈密顿量", "outcome": "选择 QAOA"},
        {"round": 2, "action": f"p=2 变分优化 (尝试 {attempts} 轮)",
         "outcome": f"bitstring {bits}, 割 {cut_value}/{n_edges}"},
        {"round": 3, "action": "与全枚举最优基线对比",
         "outcome": "通过" if verification.passed else "失败"},
    ]
    return _finalize(
        task="求 4 节点环图 (C4) 的最大割",
        algorithm="qaoa",
        reason="MaxCut 是 QAOA 的经典基准组合优化问题",
        circuit_text=f"邻接矩阵 (Ising 代价):\n{adj}\np=2 层 QAOA",
        result_summary=(f"分割 {bits} (x0 为最低位), 割边数 "
                        f"{cut_value}/{n_edges} (理论最大 {n_edges}), "
                        f"QAOA 期望能量 {energy:.4f}"),
        verification=verification,
        result_payload={"bitstring": bits, "cut": cut_value, "cost": best_cost},
        decision_chain=chain,
        verbose=verbose,
    )


# ----------------------------------------------------------------
# 5. 3 城市 TSP (QAOA)
# ----------------------------------------------------------------

def demo_tsp(verbose: bool = True) -> dict:
    """3 城市 TSP 最短回路 (QAOA/QUBO + 枚举最优校验)"""
    from ..algorithms.qaoa import QAOA

    np.random.seed(5)
    # 非对称距离矩阵: 两条候选回路成本不同
    dist = np.array([
        [0, 2, 9],
        [1, 0, 3],
        [4, 1, 0],
    ], dtype=float)
    tours = {
        0: ("0->1->2->0", dist[0, 1] + dist[1, 2] + dist[2, 0]),
        1: ("0->2->1->0", dist[0, 2] + dist[2, 1] + dist[1, 0]),
    }
    # QUBO: x0 选回路 (0=A, 1=B), x1 恒 0 (惩罚约束)
    # E(0,0)=0, E(1,0)=2, E(0,1)=E(1,1)=10
    Q = np.array([[0.0, -1.0], [-1.0, 10.0]])
    Q[0, 0] = float(tours[1][1] - tours[0][1])  # 2.0

    verification = None
    best, best_cost = 0, 0.0
    for _ in range(4):
        qaoa = QAOA(2, Q, p=2, encoding="qubo")
        qaoa.optimize(maxiter=150)
        best, best_cost = qaoa.get_optimal_bitstring()
        verification = verify_qaoa(Q, best, encoding="qubo")
        if verification.passed:
            break

    optimal = min(tours, key=lambda t: tours[t][1])
    chosen = best & 1
    chain = [
        {"round": 1, "action": "TSP -> 2 比特 QUBO 编码",
         "outcome": "选择 QAOA"},
        {"round": 2, "action": "QAOA 求解 + 枚举两条回路验证最优",
         "outcome": f"选择 {tours[chosen][0]} (成本 {tours[chosen][1]:.0f})"},
        {"round": 3, "action": "QUBO 能量与全枚举基线对比",
         "outcome": "通过" if verification.passed else "失败"},
    ]
    return _finalize(
        task="求 3 城市 TSP 的最短回路 (非对称距离)",
        algorithm="qaoa",
        reason="TSP 是经典 NP 难路径优化问题, 可编码为 QUBO 用 QAOA 求解",
        circuit_text=f"距离矩阵:\n{dist.astype(int)}\nQUBO 矩阵:\n{Q}",
        result_summary=(f"最优回路 {tours[chosen][0]} 成本 "
                        f"{tours[chosen][1]:.0f}; 枚举确认最优为 "
                        f"{tours[optimal][0]} 成本 {tours[optimal][1]:.0f}"),
        verification=verification,
        result_payload={"tour": tours[chosen][0], "cost": float(tours[chosen][1])},
        decision_chain=chain,
        verbose=verbose,
    )


# ----------------------------------------------------------------
# 6. 三角形图 3 着色 (SAT 编码 + Grover)
# ----------------------------------------------------------------

def demo_graph_coloring(verbose: bool = True) -> dict:
    """三角形图 3 着色 (SAT/CNF 编码 + Grover 搜索)"""
    from ..core.grover import GroverSearch

    random.seed(13)
    np.random.seed(13)
    # 3 个节点各 2 比特编码颜色 (0,1,2 合法, 3 非法)
    # 变量: 1,2=节点A; 3,4=节点B; 5,6=节点C (1 基文字)
    clauses = [
        (-1, -2), (-3, -4), (-5, -6),  # 颜色 != 3
        (1, 3, 2, 4), (1, 3, -2, -4), (-1, -3, 2, 4), (-1, -3, -2, -4),   # A!=B
        (1, 5, 2, 6), (1, 5, -2, -6), (-1, -5, 2, 6), (-1, -5, -2, -6),   # A!=C
        (3, 5, 4, 6), (3, 5, -4, -6), (-3, -5, 4, 6), (-3, -5, -4, -6),   # B!=C
    ]

    def satisfied(x: int) -> bool:
        return all(
            any(((x >> (abs(l) - 1)) & 1) == (1 if l > 0 else 0) for l in cl)
            for cl in clauses
        )

    search = GroverSearch(6)
    result = search.search(satisfied, shots=30)
    solutions = result.get_solutions()
    sol = solutions[0] if solutions else 0
    verification = verify_sat(clauses, sol)

    colors = [((sol >> 0) & 3), ((sol >> 2) & 3), ((sol >> 4) & 3)]
    color_names = ["红", "绿", "蓝"]
    coloring = {node: color_names[c] for node, c in zip("ABC", colors)}
    chain = [
        {"round": 1, "action": "3 着色 -> CNF (6 比特, 15 子句)",
         "outcome": "选择 Grover"},
        {"round": 2, "action": "Grover 振幅放大 (6 个合法着色)",
         "outcome": f"测得解 {sol}"},
        {"round": 3, "action": "全部 15 个子句代回校验",
         "outcome": "通过" if verification.passed else "失败"},
    ]
    return _finalize(
        task="给三角形图 (3 个节点两两相连) 做 3 着色",
        algorithm="grover",
        reason="图着色是 NP 完全约束问题, 编码为 SAT 后用 Grover 加速",
        circuit_text=f"CNF 子句 ({len(clauses)} 条):\n" +
                     "\n".join(f"  {cl}" for cl in clauses),
        result_summary=f"合法着色: {coloring}",
        verification=verification,
        result_payload={"coloring": coloring, "solution_int": sol},
        decision_chain=chain,
        verbose=verbose,
    )


# ----------------------------------------------------------------
# 7. 4 比特数据库 Grover 搜索
# ----------------------------------------------------------------

def demo_grover(verbose: bool = True) -> dict:
    """4 比特数据库搜索 (Grover 振幅放大 + top-k 命中率校验)"""
    from ..core.grover import GroverSearch

    random.seed(17)
    np.random.seed(17)
    marked = {6, 11}
    search = GroverSearch(4)
    result = search.search_with_oracle_set(marked, shots=100)
    counts = result.get_measurement_counts()
    verification = verify_grover(sorted(marked), counts, n_qubits=4)

    chain = [
        {"round": 1, "action": "数据库搜索 -> 相位 Oracle 标记 2 个目标",
         "outcome": "选择 Grover"},
        {"round": 2, "action": f"{result.iterations} 次迭代, 100 次测量",
         "outcome": f"成功率 {result.empirical_success_rate:.0%}"},
        {"round": 3, "action": "top-2 命中 + 成功率 > 经典随机 2/16",
         "outcome": "通过" if verification.passed else "失败"},
    ]
    dist = ", ".join(
        f"|{format(s, '04b')}>:{c}" for s, c in sorted(counts.items()))
    return _finalize(
        task="在 16 项数据库中找到 2 个标记项 (|0110> 与 |1011>)",
        algorithm="grover",
        reason="无序数据库搜索是 Grover 算法的原生应用, 平方级加速",
        circuit_text="Grover n=4, 标记态 {6, 11}, 最优迭代 t=2",
        result_summary=f"测量分布: {dist}",
        verification=verification,
        result_payload={"counts": counts,
                        "success_rate": result.empirical_success_rate},
        decision_chain=chain,
        verbose=verbose,
    )


# ----------------------------------------------------------------
# 8. GHZ 态制备
# ----------------------------------------------------------------

def demo_ghz(verbose: bool = True) -> dict:
    """GHZ 纠缠态制备 (测量只出现 000/111)"""
    from ..circuit.library import ghz_state

    qc = ghz_state(3)
    res = qc.execute(shots=400, seed=2024)
    counts = res.counts
    allowed = {0b000, 0b111}
    only_ghz = set(counts) <= allowed and len(set(counts)) == 2
    verification = VerificationResult(
        passed=only_ghz,
        message=(f"测量结果仅出现 |000> 与 |111> "
                 f"(计数 {counts})" if only_ghz
                 else f"出现非 GHZ 分量: {counts}"),
        details={"counts": {format(k, "03b"): v for k, v in counts.items()}},
    )
    chain = [
        {"round": 1, "action": "制备 (|000>+|111>)/√2", "outcome": "H+CNOT 链"},
        {"round": 2, "action": "400 次测量统计", "outcome": f"{counts}"},
        {"round": 3, "action": "校验只出现 000/111",
         "outcome": "通过" if verification.passed else "失败"},
    ]
    return _finalize(
        task="制备 3 比特 GHZ 纠缠态并验证其关联性",
        algorithm="circuit",
        reason="GHZ 态是最基本的多体纠缠态, 是量子通信/纠错的原材料",
        circuit_text=qc.draw(),
        result_summary=f"测量计数: {counts} (理论上 000/111 各占约一半)",
        verification=verification,
        result_payload={"counts": counts},
        decision_chain=chain,
        verbose=verbose,
    )


# ----------------------------------------------------------------
# 9. 量子随机数生成器
# ----------------------------------------------------------------

def demo_qrng(verbose: bool = True) -> dict:
    """8 比特量子随机数生成器 (均匀性卡方粗检)"""
    from ..circuit.circuit import QuantumCircuit

    qc = QuantumCircuit(8, name="QRNG-8")
    for q in range(8):
        qc.h(q)
    shots = 2048
    res = qc.execute(shots=shots, seed=99)
    counts = res.counts

    expected = shots / 256.0
    chi2 = sum((counts.get(x, 0) - expected) ** 2 / expected
               for x in range(256))
    df = 255
    # 粗检阈值: 均值 + 4 倍标准差 (df=255 -> ~345)
    threshold = df + 4 * (2 * df) ** 0.5
    uniform_ok = chi2 < threshold
    verification = VerificationResult(
        passed=uniform_ok,
        message=(f"卡方统计量 {chi2:.1f} < 阈值 {threshold:.1f} "
                 f"(df={df}), 分布与均匀一致" if uniform_ok
                 else f"卡方统计量 {chi2:.1f} >= 阈值 {threshold:.1f}, 分布异常"),
        details={"chi2": chi2, "df": df, "threshold": threshold,
                 "distinct_outcomes": len(counts), "shots": shots},
    )
    sample = [format(x, "08b") for x in list(counts)[:8]]
    chain = [
        {"round": 1, "action": "H^⊗8 制备均匀叠加", "outcome": "256 个等概态"},
        {"round": 2, "action": f"{shots} 次采样", "outcome": f"{len(counts)} 种结果"},
        {"round": 3, "action": "卡方均匀性粗检",
         "outcome": "通过" if verification.passed else "失败"},
    ]
    return _finalize(
        task="生成密码学级量子随机数 (8 比特)",
        algorithm="circuit",
        reason="叠加态测量的内在随机性是物理真随机源",
        circuit_text=qc.draw(),
        result_summary=(f"{shots} 次采样覆盖 {len(counts)}/256 种结果, "
                        f"样例: {sample}"),
        verification=verification,
        result_payload={"distinct": len(counts), "chi2": chi2},
        decision_chain=chain,
        verbose=verbose,
    )


# ----------------------------------------------------------------
# 10. BB84 量子密钥分发
# ----------------------------------------------------------------

def _bb84_run(n_qubits: int, with_eve: bool, rng: random.Random) -> Tuple[float, int]:
    """用 QuantumState 逐比特模拟 BB84 协议, 返回 (筛选后密钥一致率, 密钥长度)。"""
    from ..core.state import QuantumState

    alice_bits, alice_bases, bob_bases, bob_bits = [], [], [], []
    for _ in range(n_qubits):
        bit = rng.randint(0, 1)
        a_basis = rng.randint(0, 1)  # 0=Z, 1=X
        b_basis = rng.randint(0, 1)

        st = QuantumState(1)
        if bit:
            st.x(0)
        if a_basis:
            st.h(0)

        if with_eve:
            # Eve 随机选基测量并重发
            e_basis = rng.randint(0, 1)
            if e_basis:
                st.h(0)
            m, _ = st.measure(collapse=True)
            st = QuantumState(1)
            if m:
                st.x(0)
            if e_basis:
                st.h(0)

        if b_basis:
            st.h(0)
        m, _ = st.measure()

        alice_bits.append(bit)
        alice_bases.append(a_basis)
        bob_bases.append(b_basis)
        bob_bits.append(m)

    sifted = [(a, b) for a, ab, b, bb
              in zip(alice_bits, alice_bases, bob_bits, bob_bases) if ab == bb]
    if not sifted:
        return 0.0, 0
    agree = sum(1 for a, b in sifted if a == b) / len(sifted)
    return agree, len(sifted)


def demo_bb84(verbose: bool = True) -> dict:
    """BB84 量子密钥分发 (无窃听 100% 一致, 有窃听 ~75% 被检出)"""
    random.seed(2024)  # QuantumState.measure 使用全局 random
    rng = random.Random(2024)
    agree_clean, len_clean = _bb84_run(600, with_eve=False, rng=rng)
    agree_eve, len_eve = _bb84_run(600, with_eve=True, rng=rng)

    qber_eve = 1.0 - agree_eve
    detected = qber_eve > 0.10
    passed = (agree_clean == 1.0) and detected
    verification = VerificationResult(
        passed=passed,
        message=(f"无窃听一致率 {agree_clean:.0%}; 有窃听一致率 "
                 f"{agree_eve:.1%} (QBER {qber_eve:.1%} > 10%, 窃听被检出)"
                 if passed else
                 f"无窃听一致率 {agree_clean:.1%}, 有窃听一致率 "
                 f"{agree_eve:.1%}, 结果异常"),
        details={"agree_no_eve": agree_clean, "agree_with_eve": agree_eve,
                 "qber_with_eve": qber_eve, "sifted_no_eve": len_clean,
                 "sifted_with_eve": len_eve},
    )
    chain = [
        {"round": 1, "action": "Alice 随机比特+随机基编码", "outcome": "600 比特"},
        {"round": 2, "action": "无窃听信道", "outcome": f"一致率 {agree_clean:.0%}"},
        {"round": 3, "action": "Eve 中间人窃听", "outcome": f"一致率降至 {agree_eve:.1%}"},
        {"round": 4, "action": "QBER > 10% 判据检出窃听",
         "outcome": "通过" if verification.passed else "失败"},
    ]
    return _finalize(
        task="模拟 BB84 量子密钥分发并检测窃听",
        algorithm="circuit",
        reason="测量破坏叠加态是量子密码安全性的物理基础",
        circuit_text=("逐比特传输: Alice (X? H?) -> 信道 -> Bob (H?) 测量\n"
                      "Eve: 随机基测量(collapse) + 重发 -> 引入 25% QBER"),
        result_summary=(f"无窃听: 筛选密钥 {len_clean} 比特, 一致率 "
                        f"{agree_clean:.0%}; 有窃听: {len_eve} 比特, "
                        f"一致率 {agree_eve:.1%} (理论 75%)"),
        verification=verification,
        result_payload={"agree_no_eve": agree_clean,
                        "agree_with_eve": agree_eve},
        decision_chain=chain,
        verbose=verbose,
    )


# ----------------------------------------------------------------
# 注册表
# ----------------------------------------------------------------

DEMOS: Dict[str, Callable[..., dict]] = {
    "factor": demo_factor,
    "sat": demo_sat,
    "sudoku": demo_sudoku,
    "maxcut": demo_maxcut,
    "tsp": demo_tsp,
    "graph_coloring": demo_graph_coloring,
    "grover": demo_grover,
    "ghz": demo_ghz,
    "qrng": demo_qrng,
    "bb84": demo_bb84,
}


def list_demos() -> List[Tuple[str, str]]:
    """返回 [(演示名称, 中文一句话简介), ...]。"""
    out = []
    for name, fn in DEMOS.items():
        doc = (fn.__doc__ or "").strip().splitlines()
        out.append((name, doc[0] if doc else ""))
    return out


def run_demo(name: str, verbose: bool = False) -> dict:
    """按名称运行演示, 返回统一字典。"""
    if name not in DEMOS:
        raise KeyError(
            f"未知演示 {name!r}, 可选: {sorted(DEMOS)}")
    return DEMOS[name](verbose=verbose)
