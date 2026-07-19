"""
OpenQASM 支持 — QuantumCircuit 与 OpenQASM 2.0 / 3.0 文本的相互转换。

提供:
    - dumps_qasm2(circuit) -> str          导出 OpenQASM 2.0 (qelib1.inc)
    - loads_qasm2(text) -> QuantumCircuit  解析 OpenQASM 2.0 子集
    - dumps_qasm3(circuit) -> str          导出 OpenQASM 3.0 (stdgates.inc, 仅导出)
    - QASMParseError                       解析异常 (携带行号)

导出约定:
    - 数值参数用 format(v, '.17g') (17 位有效数字, 保证 double 精确往返)
    - qelib1/stdgates 没有的门 (sx/sxdg/iswap/dcx/ecr/cs/ct/csdg/ctdg/mcx/mcz/
      cswap/unitary 等) 先经 QuantumCircuit.decompose 展开为基础门再导出;
      sx/sxdg 等单比特门走 ZYZ 分解 (rz/ry/rz), 残余全局相位累积到末尾的
      ``// global phase: <值>`` 注释中
    - mcx/mcz 用无辅助比特的 Barenco (Lemma 7.2) 递归分解为 cx/ccx/rz/ry/u1
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import quantum_gates as G
from .circuit import QuantumCircuit
from .gate import Gate, zyz_decompose
from .parameter import resolve

__all__ = ["dumps_qasm2", "loads_qasm2", "dumps_qasm3", "QASMParseError"]


class QASMParseError(Exception):
    """OpenQASM 解析错误, 携带出错行号 (从 1 开始)。"""

    def __init__(self, message: str, line: Optional[int] = None):
        self.message = message
        self.line = line
        if line is not None:
            super().__init__(f"第 {line} 行: {message}")
        else:
            super().__init__(message)


# ====================================================================
# 门名映射表
# ====================================================================

# qelib1.inc 中存在的门 (qsl 名 -> QASM 名)。cu 单独处理 (gamma 非 0 时用
# qelib1 的 cu(θ,φ,λ,γ), 否则用 cu3)。
_QASM2_SIMPLE = {
    "id": "id", "x": "x", "y": "y", "z": "z", "h": "h",
    "s": "s", "sdg": "sdg", "t": "t", "tdg": "tdg",
    "rx": "rx", "ry": "ry", "rz": "rz",
    "p": "u1", "u": "u3",
    "cx": "cx", "cy": "cy", "cz": "cz", "ch": "ch", "swap": "swap",
    "ccx": "ccx", "crx": "crx", "cry": "cry", "crz": "crz", "cp": "cu1",
    "rxx": "rxx", "ryy": "ryy", "rzz": "rzz",
}

# stdgates.inc 额外包含 sx/sxdg/p/cp/cu 等, 可直接导出
_QASM3_SIMPLE = dict(_QASM2_SIMPLE)
_QASM3_SIMPLE.update({"p": "p", "u": "U", "cp": "cp", "sx": "sx", "sxdg": "sxdg"})

# decompose 的目标基础门集 = 可直接导出的 qsl 门名
_QASM2_BASIS = set(_QASM2_SIMPLE) | {"cu"}
_QASM3_BASIS = set(_QASM3_SIMPLE) | {"cu"}

_MCX_RE = re.compile(r"mcx(\d+)")
_MCZ_RE = re.compile(r"mcz(\d+)")

_X_GATE = Gate("x", G.X, 1)


def _fmt(value) -> str:
    """格式化数值参数: 17 位有效数字, 保证 float 精确往返。"""
    return format(float(value), ".17g")


class _QasmWriter:
    """逐条收集 QASM 指令行, 并累积分解产生的全局相位。"""

    def __init__(self, num_qubits: int, simple_map: dict, basis: set,
                 p_name: str, qname: str = "q"):
        self.num_qubits = num_qubits
        self.simple_map = simple_map
        self.basis = basis
        self.p_name = p_name      # 相位门导出名 (qasm2: u1, qasm3: p)
        self.qname = qname
        self.lines: List[str] = []
        self.phase = 0.0

    def line(self, name: str, params: Sequence[float], qubits: Sequence[int]):
        qs = ",".join(f"{self.qname}[{i}]" for i in qubits)
        if params:
            ps = ",".join(_fmt(p) for p in params)
            self.lines.append(f"{name}({ps}) {qs};")
        else:
            self.lines.append(f"{name} {qs};")


# ====================================================================
# 导出
# ====================================================================

def dumps_qasm2(circuit: QuantumCircuit) -> str:
    """导出 OpenQASM 2.0 文本 (qelib1.inc 门集)。"""
    header = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{circuit.num_qubits}];",
    ]
    return _dumps(circuit, header, _QASM2_SIMPLE, _QASM2_BASIS, "u1")


def dumps_qasm3(circuit: QuantumCircuit) -> str:
    """导出 OpenQASM 3.0 文本 (stdgates.inc 门集, 仅导出)。"""
    header = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"qubit[{circuit.num_qubits}] q;",
    ]
    return _dumps(circuit, header, _QASM3_SIMPLE, _QASM3_BASIS, "p")


def _dumps(circuit: QuantumCircuit, header: List[str], simple_map: dict,
           basis: set, p_name: str) -> str:
    w = _QasmWriter(circuit.num_qubits, simple_map, basis, p_name)
    w.phase = float(circuit.global_phase)
    for inst in circuit.data:
        _emit(inst, w)
    lines = header + w.lines
    if abs(w.phase) > 1e-15:
        lines.append(f"// global phase: {_fmt(w.phase)}")
    return "\n".join(lines) + "\n"


def _emit(inst, w: _QasmWriter):
    """把一条指令展开为可导出的 QASM 行, 追加到 writer。"""
    g = inst.gate
    name = g.name

    if name == "barrier":
        w.line("barrier", [], inst.qubits)
        return

    if g.is_parameterized:
        raise ValueError(
            f"门 '{name}' 含未绑定的符号参数, 无法导出 OpenQASM; "
            f"请先调用 circuit.bind_parameters()。"
        )

    # 直接映射的门
    if name in w.simple_map:
        params = [resolve(p) for p in g.params]
        w.line(w.simple_map[name], params, inst.qubits)
        return

    # cu: qelib1 cu3 只有 3 参数; gamma 非 0 时用 qelib1 的 cu(θ,φ,λ,γ)
    if name == "cu":
        params = [resolve(p) for p in g.params]
        params = (list(params) + [0.0])[:4]
        if w.p_name == "p":                     # qasm3 stdgates 原生 cu 4 参数
            w.line("cu", params, inst.qubits)
        elif abs(params[3]) > 1e-15:
            w.line("cu", params, inst.qubits)   # qelib1 cu(θ,φ,λ,γ)
        else:
            w.line("cu3", params[:3], inst.qubits)
        return

    # 多控制门: Barenco 递归分解
    if _MCX_RE.fullmatch(name):
        _emit_mcx(list(inst.qubits[:-1]), inst.qubits[-1], w)
        return
    if _MCZ_RE.fullmatch(name):
        qs = list(inst.qubits)
        w.line("h", [], (qs[-1],))
        _emit_mcx(qs[:-1], qs[-1], w)
        w.line("h", [], (qs[-1],))
        return

    # 其余门: 借助 QuantumCircuit.decompose 展开到基础门集
    qc = QuantumCircuit(w.num_qubits)
    qc.append(g.copy(), inst.qubits)
    dec = qc.decompose(w.basis)
    if len(dec.data) == 1 and dec.data[0].gate.name == name:
        # decompose 无法展开: 单比特门走 ZYZ, 其余报错
        if g.num_qubits == 1:
            theta, phi, lam, gamma = zyz_decompose(g.to_matrix())
            w.line("rz", [lam], inst.qubits)
            w.line("ry", [theta], inst.qubits)
            w.line("rz", [phi], inst.qubits)
            w.phase += gamma
            return
        raise ValueError(
            f"门 '{name}' ({g.num_qubits} 比特) 无法导出为 OpenQASM; "
            f"请先调用 circuit.decompose() 拆解。"
        )
    w.phase += dec.global_phase
    for sub in dec.data:
        _emit(sub, w)


def _emit_mcx(controls: List[int], target: int, w: _QasmWriter):
    """多控制 X: 1 控制=cx, 2 控制=ccx, 更多走 Barenco 递归。"""
    n = len(controls)
    if n == 0:
        w.line("x", [], (target,))
    elif n == 1:
        w.line("cx", [], (controls[0], target))
    elif n == 2:
        w.line("ccx", [], (controls[0], controls[1], target))
    else:
        _emit_controlled(_X_GATE, controls, target, w)


def _is_x(u_gate: Gate) -> bool:
    return np.allclose(u_gate.to_matrix(), G.X, atol=1e-12)


def _emit_controlled(u_gate: Gate, controls: List[int], target: int,
                     w: _QasmWriter):
    """
    无辅助比特的多受控单比特门分解 (Barenco et al. 1995, Lemma 7.2)。

    对 U = V^2, C^n(U) 展开为:
        C(V)[c_n,t] · C^{n-1}(X)[c_1..c_{n-1},c_n] · C(V†)[c_n,t]
        · C^{n-1}(X)[c_1..c_{n-1},c_n] · C^{n-1}(V)[c_1..c_{n-1},t]
    """
    if len(controls) == 1:
        if _is_x(u_gate):
            w.line("cx", [], (controls[0], target))
        else:
            _emit_controlled_1q(u_gate, controls[0], target, w)
        return
    if len(controls) == 2 and _is_x(u_gate):
        w.line("ccx", [], (controls[0], controls[1], target))
        return
    v = u_gate.power(0.5)
    vdg = u_gate.power(-0.5)
    c_last, rest = controls[-1], controls[:-1]
    _emit_controlled(v, [c_last], target, w)
    _emit_controlled(_X_GATE, rest, c_last, w)
    _emit_controlled(vdg, [c_last], target, w)
    _emit_controlled(_X_GATE, rest, c_last, w)
    _emit_controlled(v, rest, target, w)


def _emit_controlled_1q(u_gate: Gate, c: int, t: int, w: _QasmWriter):
    """
    单受控单比特门的 2-CNOT 分解 (Barenco Lemma 5.2, 与 circuit.py 一致)。

    U = e^{iγ} Rz(φ) Ry(θ) Rz(λ)  =>  C-U = P(γ)⊗A·CX·B·CX·C
    """
    theta, phi, lam, gamma = zyz_decompose(u_gate.to_matrix())
    w.line("rz", [(lam - phi) / 2], (t,))
    w.line("cx", [], (c, t))
    w.line("rz", [-(lam + phi) / 2], (t,))
    w.line("ry", [-theta / 2], (t,))
    w.line("cx", [], (c, t))
    w.line("ry", [theta / 2], (t,))
    w.line("rz", [phi], (t,))
    if abs(gamma) > 1e-12:
        w.line(w.p_name, [gamma], (c,))


# ====================================================================
# 解析
# ====================================================================

_OPENQASM_RE = re.compile(r"^OPENQASM\s+(\d+(?:\.\d+)?)\s*$")
_INCLUDE_RE = re.compile(r'^include\s+"[^"]+"\s*$')
_QREG_RE = re.compile(r"^qreg\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(\d+)\s*\]$")
_CREG_RE = re.compile(r"^creg\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(\d+)\s*\]$")
_QREF_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(\d+)\s*\]$")
_MEASURE_RE = re.compile(r"^measure\s+.+\s*->\s*.+$")
_GATE_STMT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?:\((.*)\))?\s+(.+)$")
_PHASE_COMMENT_RE = re.compile(r"//\s*global\s+phase\s*:\s*(.*)$")

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(expr: str, line: Optional[int]) -> float:
    """用 ast 白名单求值参数表达式 (pi、数值、+ - * / **、括号), 禁止任意代码。"""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise QASMParseError(f"参数表达式语法错误: {expr!r}", line)
    return _eval_node(tree.body, line)


def _eval_node(node, line: Optional[int]) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise QASMParseError(f"不支持的常量: {node.value!r}", line)
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return math.pi
        raise QASMParseError(f"未知标识符: {node.id!r} (仅支持 pi)", line)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](
            _eval_node(node.left, line), _eval_node(node.right, line))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand, line))
    raise QASMParseError(f"不支持的表达式: {ast.dump(node)}", line)


def _split_args(s: str) -> List[str]:
    """按顶层逗号切分参数列表 (括号内的逗号不切)。"""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
            cur += ch
        elif ch == ")":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return [a.strip() for a in out if a.strip()]


def _resolve_qarg(token: str, qregs: Dict[str, Tuple[int, int]],
                  line: int) -> List[int]:
    """把 q[i] 或整个寄存器名解析为全局比特下标列表。"""
    m = _QREF_RE.match(token)
    if m:
        reg, idx = m.group(1), int(m.group(2))
        if reg not in qregs:
            raise QASMParseError(f"未声明的寄存器: {reg!r}", line)
        off, size = qregs[reg]
        if not (0 <= idx < size):
            raise QASMParseError(
                f"比特下标越界: {reg}[{idx}] (寄存器大小 {size})", line)
        return [off + idx]
    if token in qregs:
        off, size = qregs[token]
        return list(range(off, off + size))
    raise QASMParseError(f"无法解析的比特参数: {token!r}", line)


def loads_qasm2(text: str) -> QuantumCircuit:
    """
    解析 OpenQASM 2.0 子集为 QuantumCircuit。

    支持: OPENQASM 头、include、qreg (可多个)、creg (忽略)、measure (忽略)、
    barrier、标准 qelib1 门指令、// 注释、``// global phase: <值>`` 注释。
    未识别的门名抛 QASMParseError (携带行号)。
    """
    # 第一遍: 收集全局相位注释, 并剥离 // 注释
    phase = 0.0
    body_lines: List[str] = []
    for i, raw in enumerate(text.splitlines(), 1):
        m = _PHASE_COMMENT_RE.search(raw)
        if m:
            phase += _safe_eval(m.group(1).strip(), i)
        body_lines.append(raw.split("//", 1)[0])

    # 第二遍: 按分号切语句, 记录起始行号
    stmts: List[Tuple[int, str]] = []
    buf, start = "", None
    for i, line in enumerate(body_lines, 1):
        s = line.strip()
        if not s:
            continue
        if start is None:
            start = i
        buf += s + " "
        while ";" in buf:
            stmt, buf = buf.split(";", 1)
            stmt = stmt.strip()
            if stmt:
                stmts.append((start, stmt))
            start = i
        if not buf.strip():
            start = None
    if buf.strip():
        raise QASMParseError("语句缺少结尾分号 ';'", len(body_lines) or 1)

    # 第三遍: 逐语句解析
    qregs: Dict[str, Tuple[int, int]] = {}
    total = 0
    ops: List[Tuple[int, str, List[float], List[int]]] = []
    saw_header = False
    for lineno, stmt in stmts:
        m = _OPENQASM_RE.match(stmt)
        if m:
            if saw_header:
                raise QASMParseError("重复的 OPENQASM 版本头", lineno)
            saw_header = True
            if not m.group(1).startswith("2"):
                raise QASMParseError(
                    f"仅支持 OpenQASM 2.x, 得到版本 {m.group(1)}", lineno)
            continue
        if _INCLUDE_RE.match(stmt):
            continue
        m = _QREG_RE.match(stmt)
        if m:
            reg, size = m.group(1), int(m.group(2))
            if reg in qregs:
                raise QASMParseError(f"重复定义的寄存器: {reg!r}", lineno)
            if size < 1:
                raise QASMParseError(f"寄存器 {reg!r} 大小必须 >= 1", lineno)
            qregs[reg] = (total, size)
            total += size
            continue
        if _CREG_RE.match(stmt):
            continue
        if _MEASURE_RE.match(stmt):
            continue

        m = _GATE_STMT_RE.match(stmt)
        if not m:
            raise QASMParseError(f"无法解析的语句: {stmt!r}", lineno)
        name, pstr, qstr = m.group(1), m.group(2), m.group(3)
        params = ([] if pstr is None
                  else [_safe_eval(a, lineno) for a in _split_args(pstr)])
        qargs = [_resolve_qarg(tok, qregs, lineno) for tok in _split_args(qstr)]
        if name == "barrier":
            flat = [q for sub in qargs for q in sub]
            ops.append((lineno, "barrier", [], flat))
            continue
        for sub in qargs:
            if len(sub) != 1:
                raise QASMParseError(
                    f"门 '{name}' 不支持整个寄存器作为参数", lineno)
        ops.append((lineno, name, params, [sub[0] for sub in qargs]))

    if not saw_header:
        raise QASMParseError("缺少 OPENQASM 版本头", 1)
    if total == 0:
        raise QASMParseError("缺少 qreg 声明", 1)

    qc = QuantumCircuit(total)
    qc.global_phase = float(phase)
    for lineno, name, params, qubits in ops:
        _apply_gate(qc, name, params, qubits, lineno)
    return qc


def _check_arity(name: str, params: Sequence[float], qs: Sequence[int],
                 n_params: int, n_qubits: int, line: int):
    if len(params) != n_params or len(qs) != n_qubits:
        raise QASMParseError(
            f"门 '{name}' 需要 {n_params} 个参数和 {n_qubits} 个比特, "
            f"得到 {len(params)} 个参数和 {len(qs)} 个比特", line)


_IMP_1Q_0P = ("id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg")
_IMP_2Q_0P = ("cx", "cy", "cz", "ch", "swap", "iswap", "dcx", "ecr")


def _apply_gate(qc: QuantumCircuit, name: str, params: List[float],
                qs: List[int], line: int):
    if name == "barrier":
        qc.barrier(qs)
        return
    if name in _IMP_1Q_0P:
        _check_arity(name, params, qs, 0, 1, line)
        getattr(qc, name)(qs[0])
        return
    if name in _IMP_2Q_0P:
        _check_arity(name, params, qs, 0, 2, line)
        getattr(qc, name)(qs[0], qs[1])
        return
    if name in ("rx", "ry", "rz"):
        _check_arity(name, params, qs, 1, 1, line)
        getattr(qc, name)(qs[0], params[0])
        return
    if name in ("u1", "p"):
        _check_arity(name, params, qs, 1, 1, line)
        qc.p(qs[0], params[0])
        return
    if name == "u2":
        _check_arity(name, params, qs, 2, 1, line)
        qc.u(qs[0], math.pi / 2, params[0], params[1])
        return
    if name in ("u3", "u", "U"):
        _check_arity(name, params, qs, 3, 1, line)
        qc.u(qs[0], params[0], params[1], params[2])
        return
    if name == "u0":
        _check_arity(name, params, qs, 1, 1, line)
        qc.id(qs[0])
        return
    if name == "ccx":
        _check_arity(name, params, qs, 0, 3, line)
        qc.ccx(qs[0], qs[1], qs[2])
        return
    if name == "cswap":
        _check_arity(name, params, qs, 0, 3, line)
        qc.cswap(qs[0], qs[1], qs[2])
        return
    if name in ("crx", "cry", "crz"):
        _check_arity(name, params, qs, 1, 2, line)
        getattr(qc, name)(params[0], qs[0], qs[1])
        return
    if name in ("cu1", "cp"):
        _check_arity(name, params, qs, 1, 2, line)
        qc.cp(params[0], qs[0], qs[1])
        return
    if name == "cu3":
        _check_arity(name, params, qs, 3, 2, line)
        qc.cu(params[0], params[1], params[2], qs[0], qs[1], 0.0)
        return
    if name == "cu":
        _check_arity(name, params, qs, 4, 2, line)
        qc.cu(params[0], params[1], params[2], qs[0], qs[1], params[3])
        return
    if name in ("rxx", "ryy", "rzz"):
        _check_arity(name, params, qs, 1, 2, line)
        getattr(qc, name)(params[0], qs[0], qs[1])
        return
    if name == "csx":
        _check_arity(name, params, qs, 0, 2, line)
        qc.controlled(Gate("sx", G.SX, 1, label="√X"), [qs[0]], [qs[1]])
        return
    raise QASMParseError(f"未知的门名: {name!r}", line)
