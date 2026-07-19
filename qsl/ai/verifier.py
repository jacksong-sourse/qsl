"""
自动验证器 — 对量子算法结果做经典可检验的独立校验。

每个 verify_* 函数返回 VerificationResult(passed, message, details)，
不修改任何算法内部状态，只做"结果 => 经典重算/代回"的交叉验证：

- verify_shor:   因子回乘 + 素性试除
- verify_sat:    把赋值代回子句逐一求值
- verify_qaoa:   与经典基线（全枚举 / 随机采样）对比
- verify_grover: top-k 命中标记集 + 成功率超过经典随机
- verify_vqe:    精确对角化基态能量下界校验 (n <= 12)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


@dataclass
class VerificationResult:
    """一次自动验证的结果。"""
    passed: bool
    message: str
    details: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"VerificationResult({status}: {self.message})"


# ----------------------------------------------------------------
# Shor 因子分解校验
# ----------------------------------------------------------------

def _is_prime_trial(n: int) -> bool:
    """试除法素性检验 (到 sqrt(n))。"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    for i in range(3, r + 1, 2):
        if n % i == 0:
            return False
    return True


def verify_shor(N: int, factors: Sequence[int]) -> VerificationResult:
    """
    校验 Shor 分解结果: 因子均为 >1 的整数、回乘等于 N、且每个因子都是素数。

    参数:
        N: 被分解的整数
        factors: 算法返回的因子列表
    """
    details: Dict[str, Any] = {"N": int(N), "factors": list(factors)}

    if not factors:
        return VerificationResult(False, "因子列表为空", details)

    ints = []
    for f in factors:
        if isinstance(f, (bool, np.bool_)) or not isinstance(f, (int, np.integer)):
            return VerificationResult(
                False, f"因子 {f!r} 不是整数", details)
        if int(f) <= 1:
            return VerificationResult(
                False, f"因子 {int(f)} <= 1，不是有效因子", details)
        ints.append(int(f))
    details["factors"] = ints

    product = 1
    for f in ints:
        product *= f
    details["product"] = product
    if product != N:
        return VerificationResult(
            False, f"回乘失败: {'*'.join(map(str, ints))} = {product} != {N}",
            details)

    composite = [f for f in ints if not _is_prime_trial(f)]
    if composite:
        details["composite_factors"] = composite
        return VerificationResult(
            False, f"因子 {composite} 不是素数 (分解不彻底或错误)", details)

    if len(ints) == 1 and ints[0] == N and not _is_prime_trial(N):
        return VerificationResult(
            False, f"未能分解合数 N={N} (返回了 N 本身)", details)

    return VerificationResult(
        True, f"分解正确: {N} = {' * '.join(map(str, ints))}", details)


# ----------------------------------------------------------------
# SAT 子句校验
# ----------------------------------------------------------------

def _assignment_to_int(assignment: Any, literal_base: Optional[int]) -> int:
    """
    把多种形式的赋值统一为位整数 (bit i = 变量 x_i 的值)。

    支持:
        - int: 直接作为位整数
        - str: 二进制位串 (与 format(x, '0nb') 一致, 左侧为高位)
        - dict: {变量: bool}; 键为 "xN" / int (0 基) /
                对整数文字子句可为 1 基变量编号
    """
    if isinstance(assignment, (int, np.integer)):
        return int(assignment)
    if isinstance(assignment, str):
        s = assignment.strip().replace(" ", "")
        if not s or any(c not in "01" for c in s):
            raise ValueError(f"非法 bitstring 赋值: {assignment!r}")
        return int(s, 2)
    if isinstance(assignment, dict):
        if not assignment:
            return 0
        keys = list(assignment.keys())
        # "xN" 形式的键: N 即 0 基变量索引
        if all(isinstance(k, str) and k.lower().startswith("x") for k in keys):
            value = 0
            for k, v in assignment.items():
                if v:
                    value |= 1 << int(k[1:])
            return value
        int_keys = [int(k) for k in keys]
        # 整数文字子句常用 1 基编号 {1..n}: 键恰好覆盖 1..max 时按 1 基处理
        if (literal_base == 1 and 0 not in int_keys
                and max(int_keys) == len(int_keys)):
            value = 0
            for k, v in assignment.items():
                if v:
                    value |= 1 << (int(k) - 1)
            return value
        value = 0
        for k, v in assignment.items():
            if v:
                value |= 1 << int(k)
        return value
    raise TypeError(f"不支持的赋值类型: {type(assignment).__name__}")


def _eval_clause_literals(clause: Sequence[int], value: int) -> bool:
    """对整数文字子句 (如 (1, -2)) 求值: 文字 L 引用变量 x_{|L|-1}。"""
    for lit in clause:
        lit = int(lit)
        if lit == 0:
            raise ValueError("文字不能为 0")
        bit = (value >> (abs(lit) - 1)) & 1
        if (lit > 0 and bit) or (lit < 0 and not bit):
            return True
    return False


def verify_sat(clauses: Sequence[Any], assignment: Any) -> VerificationResult:
    """
    把赋值代回所有子句逐一求值 (子句之间为逻辑与)。

    参数:
        clauses: 字符串子句列表 (QSLProgram premises 语法,
                 如 ["x0 & x1", "~x1 | x2"]) 或整数文字子句
                 (如 [(1, -2), (2,)], 即 (x1 | ~x2) & (x2))
        assignment: {变量: bool} 字典 / bitstring / 位整数
    """
    if not clauses:
        return VerificationResult(False, "子句列表为空", {"clauses": []})

    is_string_form = isinstance(clauses[0], str)
    value = _assignment_to_int(assignment, literal_base=None if is_string_form else 1)
    details: Dict[str, Any] = {
        "assignment_int": value,
        "n_clauses": len(clauses),
        "clause_results": [],
    }

    if is_string_form:
        from ..core.parser import parse_bool
        parsed = []
        for c in clauses:
            try:
                parsed.append(parse_bool(c))
            except Exception as e:
                return VerificationResult(
                    False, f"子句 {c!r} 解析失败: {e}", details)
        evaluators = [lambda v, e=e: bool(e.evaluate(v)) for e in parsed]
        clause_strs = [e.to_string() for e in parsed]
    else:
        evaluators = [
            lambda v, cl=tuple(cl): _eval_clause_literals(cl, v)
            for cl in clauses
        ]
        clause_strs = [str(tuple(cl)) for cl in clauses]

    failed = []
    for idx, (ev, cs) in enumerate(zip(evaluators, clause_strs)):
        ok = ev(value)
        details["clause_results"].append({"clause": cs, "satisfied": ok})
        if not ok:
            failed.append(cs)

    if failed:
        return VerificationResult(
            False, f"赋值不满足 {len(failed)} 个子句: {failed}", details)
    return VerificationResult(
        True, f"赋值满足全部 {len(clauses)} 个子句", details)


# ----------------------------------------------------------------
# QAOA 校验 (对比经典基线)
# ----------------------------------------------------------------

def _ising_cost(cost_matrix: np.ndarray, bitstring: int) -> float:
    """Ising 能量: E = sum_i Q[i,i]*s_i + sum_{i<j} Q[i,j]*s_i*s_j, s=±1。"""
    Q = np.asarray(cost_matrix, dtype=float)
    n = Q.shape[0]
    cost = 0.0
    spins = [-1.0 if (bitstring >> i) & 1 else 1.0 for i in range(n)]
    for i in range(n):
        cost += Q[i, i] * spins[i]
        for j in range(i + 1, n):
            cost += Q[i, j] * spins[i] * spins[j]
    return cost


def _qubo_cost(cost_matrix: np.ndarray, bitstring: int) -> float:
    """QUBO 能量: E = sum_i Q[i,i]*x_i + sum_{i<j} Q[i,j]*x_i*x_j, x∈{0,1}。"""
    Q = np.asarray(cost_matrix, dtype=float)
    n = Q.shape[0]
    cost = 0.0
    xs = [(bitstring >> i) & 1 for i in range(n)]
    for i in range(n):
        cost += Q[i, i] * xs[i]
        for j in range(i + 1, n):
            cost += Q[i, j] * xs[i] * xs[j]
    return cost


def verify_qaoa(cost_matrix: np.ndarray, bitstring: Union[int, str],
                baseline: str = "random",
                encoding: str = "ising",
                n_random: int = 1000) -> VerificationResult:
    """
    与经典基线对比校验 QAOA 解的质量。

    规则: 量子解 cost <= 经典最好 cost 即通过。
    基线:
        - n <= 16: 全枚举精确最优
        - n > 16:  随机采样 n_random 次的最好值
        - baseline="exact" 强制全枚举 (忽略 n 大小)

    参数:
        cost_matrix: 问题代价矩阵
        bitstring: QAOA 返回的解 (int 或 01 字符串)
        encoding: "ising" (默认) 或 "qubo" (与 QAOA 类约定一致)
    """
    Q = np.asarray(cost_matrix, dtype=float)
    n = Q.shape[0]
    if isinstance(bitstring, str):
        bitstring = int(bitstring.replace(" ", ""), 2)
    bitstring = int(bitstring)

    cost_fn = _qubo_cost if encoding == "qubo" else _ising_cost
    quantum_cost = cost_fn(Q, bitstring)

    if baseline not in ("random", "exact"):
        raise ValueError(f"未知 baseline: {baseline!r} (可选 'random'/'exact')")

    details: Dict[str, Any] = {
        "n_qubits": n,
        "bitstring": bitstring,
        "quantum_cost": quantum_cost,
        "encoding": encoding,
    }

    if baseline == "exact" or (baseline == "random" and n <= 16):
        best = min(cost_fn(Q, x) for x in range(1 << n))
        details["baseline"] = "enumerate"
    else:
        rng = random.Random(2024)
        best = min(cost_fn(Q, rng.randrange(1 << n)) for _ in range(n_random))
        details["baseline"] = f"random{n_random}"
    details["classical_best_cost"] = best

    if quantum_cost <= best + 1e-9:
        return VerificationResult(
            True,
            f"量子解 cost={quantum_cost:.6g} 达到经典基线最优 "
            f"({details['baseline']}: {best:.6g})",
            details)
    return VerificationResult(
        False,
        f"量子解 cost={quantum_cost:.6g} 劣于经典基线 "
        f"({details['baseline']}: {best:.6g})",
        details)


# ----------------------------------------------------------------
# Grover 搜索校验
# ----------------------------------------------------------------

def verify_grover(marked_states: Sequence[int], measured: Dict[int, int],
                  n_qubits: Optional[int] = None) -> VerificationResult:
    """
    校验 Grover 测量结果。

    规则:
        1. 计数最高的 top-k (k = 标记态数量) 全部落在标记集合内
        2. 落在标记集合上的成功率 > 经典随机命中率 k/N

    参数:
        marked_states: 标记态 (解) 的基态索引列表
        measured: 测量计数 {基态索引: 次数}
        n_qubits: 比特数 (None 时从数据推断)
    """
    marked = {int(s) for s in marked_states}
    counts = {int(k): int(v) for k, v in measured.items()}

    if not marked:
        return VerificationResult(False, "标记态集合为空", {})
    if not counts or sum(counts.values()) <= 0:
        return VerificationResult(False, "测量计数为空", {"marked": sorted(marked)})

    if n_qubits is None:
        maxv = max(marked | set(counts))
        n_qubits = max(1, int(maxv).bit_length())
    N = 1 << n_qubits
    k = len(marked)

    top_k = sorted(counts, key=lambda s: -counts[s])[:k]
    top_hit = [s for s in top_k if s in marked]
    total = sum(counts.values())
    hit_count = sum(c for s, c in counts.items() if s in marked)
    success_rate = hit_count / total
    classical_rate = k / N

    details = {
        "n_qubits": n_qubits,
        "marked": sorted(marked),
        "top_k": top_k,
        "top_k_hits": top_hit,
        "success_rate": success_rate,
        "classical_random_rate": classical_rate,
        "shots": total,
    }

    if len(top_hit) < k:
        missing = [s for s in top_k if s not in marked]
        return VerificationResult(
            False, f"top-{k} 测量结果 {missing} 不在标记集合中", details)
    if success_rate <= classical_rate:
        return VerificationResult(
            False,
            f"成功率 {success_rate:.4f} 未超过经典随机 {classical_rate:.4f}",
            details)
    return VerificationResult(
        True,
        f"top-{k} 全部命中标记集合, 成功率 {success_rate:.4f} "
        f"> 经典随机 {classical_rate:.4f}",
        details)


# ----------------------------------------------------------------
# VQE 能量校验 (精确对角化下界)
# ----------------------------------------------------------------

_PAULI_MATS = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_string_matrix(pauli: str) -> np.ndarray:
    """
    构造 Pauli 串的稠密矩阵。

    约定 (与 qsl 一致): pauli[i] 作用在量子比特 i (LSB) 上,
    因此 Kronecker 乘积从高位比特 (串尾) 向低位比特 (串首) 展开。
    """
    m = _PAULI_MATS[pauli[-1]]
    for c in reversed(pauli[:-1]):
        m = np.kron(m, _PAULI_MATS[c])
    return m


def hamiltonian_matrix(hamiltonian_terms: Sequence) -> np.ndarray:
    """把 [(coeff, pauli_str), ...] 组装成稠密哈密顿量矩阵。"""
    terms = list(hamiltonian_terms)
    n = len(terms[0][1])
    H = np.zeros((1 << n, 1 << n), dtype=complex)
    for coeff, pauli in terms:
        if len(pauli) != n:
            raise ValueError(f"Pauli 串长度不一致: {pauli!r}")
        H += float(coeff) * _pauli_string_matrix(pauli)
    return H


def verify_vqe(energy: float, hamiltonian_terms: Sequence,
               ansatz_state: Optional[np.ndarray] = None,
               hf_energy: Optional[float] = None,
               max_exact_qubits: int = 12) -> VerificationResult:
    """
    校验 VQE 能量。

    规则:
        1. energy >= 精确对角化基态能量 - 1e-6 (n <= max_exact_qubits;
           超出时退化为逐项三角下界 sum(-|coeff|))
        2. 若提供 hf_energy (Hartree-Fock 平凡上界), energy <= hf_energy + 1e-6
        3. 若提供 ansatz_state, 重算 <psi|H|psi> 并与 energy 对照
    """
    terms = list(hamiltonian_terms)
    if not terms:
        return VerificationResult(False, "哈密顿量项为空", {})
    n = len(terms[0][1])
    energy = float(energy)
    details: Dict[str, Any] = {"n_qubits": n, "reported_energy": energy}

    if math.isnan(energy) or math.isinf(energy):
        return VerificationResult(False, f"能量非法: {energy}", details)

    # --- 下界校验 ---
    if n <= max_exact_qubits:
        H = hamiltonian_matrix(terms)
        ground = float(np.linalg.eigvalsh(H)[0].real)
        details["exact_ground_energy"] = ground
        details["energy_gap"] = energy - ground
        lower_ok = energy >= ground - 1e-6
        lower_msg = (f"精确基态能量 E0={ground:.8f}, "
                     f"上报能量 E={energy:.8f} (差 {energy - ground:+.2e})")
    else:
        ground = None
        bound = -sum(abs(float(c)) for c, _ in terms)
        details["loose_lower_bound"] = bound
        lower_ok = energy >= bound - 1e-6
        lower_msg = (f"n={n} 超过精确对角化上限, 使用逐项下界 "
                     f"sum(-|c|)={bound:.6g}")

    if not lower_ok:
        return VerificationResult(
            False, f"能量低于物理下界 (不可能): {lower_msg}", details)

    # --- Hartree-Fock 上界校验 ---
    if hf_energy is not None:
        details["hf_energy"] = float(hf_energy)
        if energy > hf_energy + 1e-6:
            return VerificationResult(
                False,
                f"能量 E={energy:.8f} 高于 Hartree-Fock 上界 "
                f"{hf_energy:.8f} (优化失败或未收敛)",
                details)

    # --- 态向量交叉校验 ---
    if ansatz_state is not None:
        state = np.asarray(ansatz_state, dtype=complex).reshape(-1)
        if state.size != (1 << n):
            return VerificationResult(
                False,
                f"ansatz_state 维度 {state.size} 与 n={n} 不符", details)
        if ground is None:
            H = hamiltonian_matrix(terms)
        recomputed = float(np.vdot(state, H @ state).real)
        details["recomputed_energy"] = recomputed
        if abs(recomputed - energy) > 1e-4:
            return VerificationResult(
                False,
                f"上报能量 {energy:.8f} 与态向量重算 "
                f"{recomputed:.8f} 不一致", details)

    return VerificationResult(
        True, f"能量合理 (不低于基态下界): {lower_msg}", details)


# ----------------------------------------------------------------
# 顶层分发器
# ----------------------------------------------------------------

def verify(algorithm: str, result: dict, context: dict) -> VerificationResult:
    """
    按算法名分发到对应的验证器。

    参数:
        algorithm: "shor" / "sat" / "qaoa" / "grover" / "vqe" (含常见别名)
        result: 算法结果字典
            shor:  {"factors": [...]}
            sat:   {"assignment": ...}
            qaoa:  {"bitstring": int}
            grover: {"measured": {state: count}}
            vqe:   {"energy": float, "state": optional}
        context: 问题上下文
            shor:  {"N": int}
            sat:   {"clauses": [...]}
            qaoa:  {"cost_matrix": ..., "baseline": optional, "encoding": optional}
            grover: {"marked_states": [...], "n_qubits": optional}
            vqe:   {"hamiltonian_terms": ..., "hf_energy": optional}
    """
    alg = algorithm.lower()
    context = context or {}
    result = result or {}

    if alg in ("shor", "factor", "factoring"):
        return verify_shor(context["N"], result.get("factors", []))
    if alg in ("sat", "boolean_sat", "3sat"):
        return verify_sat(context["clauses"], result.get("assignment"))
    if alg in ("qaoa", "maxcut", "tsp", "optimization"):
        return verify_qaoa(
            context["cost_matrix"],
            result.get("bitstring"),
            baseline=context.get("baseline", "random"),
            encoding=context.get("encoding", "ising"),
        )
    if alg in ("grover", "search", "database"):
        return verify_grover(
            context["marked_states"],
            result.get("measured", {}),
            n_qubits=context.get("n_qubits"),
        )
    if alg in ("vqe", "chemistry", "ground_state"):
        return verify_vqe(
            result.get("energy"),
            context["hamiltonian_terms"],
            ansatz_state=result.get("state"),
            hf_energy=context.get("hf_energy"),
        )
    return VerificationResult(False, f"未知算法: {algorithm!r}，无法分发验证",
                              {"algorithm": algorithm})
