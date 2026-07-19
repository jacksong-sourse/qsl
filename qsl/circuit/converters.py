"""
Qiskit / Cirq 双向转换器。

提供:
    - to_qiskit(circuit)   : qsl QuantumCircuit -> qiskit.QuantumCircuit
    - from_qiskit(qc)      : qiskit.QuantumCircuit -> qsl QuantumCircuit
    - to_cirq(circuit)     : qsl QuantumCircuit -> cirq.Circuit

qiskit / cirq 均为可选依赖, 仅在调用对应函数时延迟 import;
未安装时抛出 ImportError 并提示 pip install。

参数映射约定:
    - qsl Parameter <-> qiskit.circuit.Parameter (同名)
    - qsl Parameter <-> sympy.Symbol (同名, cirq)
    - 同一参数名在整次转换中共享同一对象 (按名缓存)。
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from .circuit import QuantumCircuit
from .gate import Gate
from .parameter import Parameter, ParameterExpression

__all__ = ["to_qiskit", "from_qiskit", "to_cirq"]


# ====================================================================
# 参数表达式转换
# ====================================================================

def _qsl_expr_to_qiskit(expr, cache: Dict[str, object]):
    """把 qsl 参数表达式递归翻译为 qiskit 表达式 (同名 Parameter 缓存)。"""
    import qiskit  # 调用方保证已安装

    if isinstance(expr, Parameter):
        if expr.name not in cache:
            cache[expr.name] = qiskit.circuit.Parameter(expr.name)
        return cache[expr.name]
    if isinstance(expr, ParameterExpression):
        args = [_qsl_expr_to_qiskit(a, cache) for a in expr._args]
        op = expr._op
        if op == "add":
            return args[0] + args[1]
        if op == "sub":
            return args[0] - args[1]
        if op == "mul":
            return args[0] * args[1]
        if op == "div":
            return args[0] / args[1]
        if op == "pow":
            return args[0] ** args[1]
        if op == "neg":
            return -args[0]
        raise ValueError(f"无法转换为 qiskit 表达式的 qsl 表达式: {expr!r}")
    return expr


def _qiskit_expr_to_qsl(expr, cache: Dict[str, Parameter]):
    """把 qiskit 参数表达式递归翻译为 qsl 表达式 (同名 Parameter 缓存)。"""
    from qiskit.circuit import Parameter as _QiskitParameter
    from qiskit.circuit import ParameterExpression as _QiskitExpr

    if isinstance(expr, _QiskitParameter):
        if expr.name not in cache:
            cache[expr.name] = Parameter(expr.name)
        return cache[expr.name]
    if isinstance(expr, _QiskitExpr):
        symbols = sorted(expr.parameters, key=lambda s: s.name)
        if not symbols:
            return float(expr)
        if len(symbols) == 1:
            s = symbols[0]
            try:
                # 常见线性形式 a*x + b
                b = float(expr.assign(s, 0.0))
                a = float(expr.assign(s, 1.0)) - b
            except TypeError as exc:
                raise ValueError(
                    f"不支持的 qiskit 参数表达式 {expr!r}: "
                    "仅支持常数或单符号线性表达式"
                ) from exc
            qsl_p = _qiskit_expr_to_qsl(s, cache)
            if a == 1.0 and b == 0.0:
                return qsl_p
            return a * qsl_p + b
        raise ValueError(f"不支持的 qiskit 参数表达式 {expr!r}: 含多个符号")
    return expr


# ====================================================================
# to_qiskit
# ====================================================================

def to_qiskit(circuit: QuantumCircuit):
    """
    把 qsl QuantumCircuit 转换为 qiskit.QuantumCircuit。

    未安装 qiskit 时抛出 ImportError (提示 pip install qiskit)。
    """
    try:
        import qiskit  # noqa: F401
        from qiskit import QuantumCircuit as _QiskitQC
        from qiskit.circuit.library import (
            HGate, XGate, YGate, ZGate, SGate, SdgGate, TGate, TdgGate,
            SXGate, SXdgGate, IGate,
            RXGate, RYGate, RZGate, PhaseGate, UGate,
            CXGate, CYGate, CZGate, CHGate, CSGate, CSdgGate,
            CPhaseGate, CRXGate, CRYGate, CRZGate, CUGate,
            SwapGate, iSwapGate, DCXGate, ECRGate,
            RXXGate, RYYGate, RZZGate,
            CCXGate, CSwapGate, UnitaryGate,
        )
    except ImportError as exc:
        raise ImportError(
            "to_qiskit 需要 qiskit, 请先安装: pip install qiskit"
        ) from exc

    fixed = {
        "id": IGate, "x": XGate, "y": YGate, "z": ZGate,
        "h": HGate, "s": SGate, "sdg": SdgGate,
        "t": TGate, "tdg": TdgGate, "sx": SXGate, "sxdg": SXdgGate,
        "cx": CXGate, "cy": CYGate, "cz": CZGate, "ch": CHGate,
        "cs": CSGate, "csdg": CSdgGate,
        "swap": SwapGate, "iswap": iSwapGate, "dcx": DCXGate,
        "ecr": ECRGate, "ccx": CCXGate, "cswap": CSwapGate,
    }
    parametric = {
        "rx": RXGate, "ry": RYGate, "rz": RZGate, "p": PhaseGate,
        "u": UGate, "cp": CPhaseGate, "crx": CRXGate, "cry": CRYGate,
        "crz": CRZGate, "cu": CUGate,
        "rxx": RXXGate, "ryy": RYYGate, "rzz": RZZGate,
    }

    qc = _QiskitQC(circuit.num_qubits, name=circuit.name or None)
    cache: Dict[str, object] = {}

    for inst in circuit.data:
        gate = inst.gate
        name = gate.name
        qs = list(inst.qubits)
        params = [_qsl_expr_to_qiskit(p, cache) for p in gate.params]

        if name in fixed:
            qc.append(fixed[name](), qs)
            continue
        if name in parametric:
            qc.append(parametric[name](*params), qs)
            continue
        # ct / ctdg -> cp(±pi/4)
        if name == "ct":
            qc.append(CPhaseGate(math.pi / 4), qs)
            continue
        if name == "ctdg":
            qc.append(CPhaseGate(-math.pi / 4), qs)
            continue
        # mcx (qsl 命名 "mcx<n>", n 个控制位)
        if name.startswith("mcx"):
            qc.mcx(qs[:-1], qs[-1])
            continue
        # mcz -> ZGate().control(n-1)
        if name.startswith("mcz"):
            qc.append(ZGate().control(len(qs) - 1), qs)
            continue
        if name == "unitary":
            # qsl 矩阵基序 t0=最高位, qiskit qargs[0]=最低位, 需反转比特序
            qc.append(UnitaryGate(gate.to_matrix(), label=gate.label),
                      qs[::-1])
            continue
        if name == "barrier":
            qc.barrier(qs)
            continue
        raise ValueError(f"无法转换为 qiskit 的门: {name!r}")

    qc.global_phase = circuit.global_phase
    return qc


# ====================================================================
# from_qiskit
# ====================================================================

def from_qiskit(qc) -> QuantumCircuit:
    """
    把 qiskit.QuantumCircuit 转换为 qsl QuantumCircuit。

    不认识的门: 有 .definition 则递归展开 (最多 10 层),
    否则用其酉矩阵包成 qsl unitary 门。
    未安装 qiskit 时抛出 ImportError (提示 pip install qiskit)。
    """
    try:
        import qiskit  # noqa: F401
        from qiskit.circuit import Barrier as _Barrier
    except ImportError as exc:
        raise ImportError(
            "from_qiskit 需要 qiskit, 请先安装: pip install qiskit"
        ) from exc

    try:
        n = qc.num_qubits
    except AttributeError as exc:
        raise TypeError("from_qiskit 需要 qiskit.QuantumCircuit 对象") from exc

    from .. import quantum_gates as G

    out = QuantumCircuit(n, name=getattr(qc, "name", "") or "",
                         global_phase=float(qc.global_phase))
    cache: Dict[str, Parameter] = {}

    def make(name: str, params: List) -> Gate:
        """按 qiskit 门名重建 qsl 门; 不认识的门抛 ValueError。"""
        if name == "cx":
            mat = np.eye(4, dtype=complex)
            mat[2, 2] = 0; mat[3, 3] = 0
            mat[2, 3] = 1; mat[3, 2] = 1
            return Gate("cx", mat, 2, label="X")
        if name == "cy":
            return Gate("cy", G.controlled_gate(G.Y, 1), 2, label="Y")
        if name == "cz":
            return Gate("cz", G.controlled_gate(G.Z, 1), 2, label="Z")

        fixed = {
            "id": (G.I, 1, "I"), "x": (G.X, 1, "X"), "y": (G.Y, 1, "Y"),
            "z": (G.Z, 1, "Z"), "h": (G.H, 1, "H"), "s": (G.S, 1, "S"),
            "sdg": (G.Sdg, 1, "S†"), "t": (G.T, 1, "T"),
            "tdg": (G.Tdg, 1, "T†"), "sx": (G.SX, 1, "√X"),
            "sxdg": (G.SXdg, 1, "√X†"),
            "ch": (G.ch(), 2, "H"), "cs": (G.cs(), 2, "S"),
            "csdg": (G.csdg(), 2, "S†"),
            "swap": (G.swap(), 2, "×"), "iswap": (G.iswap(), 2, "i×"),
            "dcx": (G.dcx(), 2, "DCX"), "ecr": (G.ecr(), 2, "ECR"),
            "ccx": (G.mcx(2), 3, "X"), "cswap": (G.cswap(), 3, "×"),
        }
        if name in fixed:
            mat, nq, label = fixed[name]
            return Gate(name, mat, nq, label=label)

        fn_map = {
            "rx": (G.rx, 1, "RX"), "ry": (G.ry, 1, "RY"),
            "rz": (G.rz, 1, "RZ"), "p": (G.p, 1, "P"),
            "u": (G.u3, 1, "U"),
            "cp": (G.cp, 2, "P"), "crx": (G.crx, 2, "RX"),
            "cry": (G.cry, 2, "RY"), "crz": (G.crz, 2, "RZ"),
            "cu": (G.cu, 2, "U"),
            "rxx": (G.rxx, 2, "RXX"), "ryy": (G.ryy, 2, "RYY"),
            "rzz": (G.rzz, 2, "RZZ"),
        }
        if name in fn_map:
            fn, nq, label = fn_map[name]
            return Gate(name, None, nq, params=list(params),
                        matrix_fn=fn, label=label)
        raise ValueError(name)

    def convert_operation(op, qubits: List[int], depth: int):
        """把单条 qiskit 指令转换/展开后追加到 out。"""
        name = op.name

        if isinstance(op, _Barrier) or name == "barrier":
            out.barrier(qubits)
            return
        if name in ("measure", "delay", "reset"):
            raise ValueError(f"不支持的 qiskit 指令: {name!r}")

        # qiskit 原生 unitary 门: 直接包回 qsl unitary (避免 definition
        # 展开引入全局相位漂移; qiskit 矩阵 qargs[0]=最低位, 需反转)
        if name == "unitary":
            mat = np.asarray(op.to_matrix(), dtype=complex)
            out.unitary(mat, qubits[::-1],
                        label=getattr(op, "label", None) or name)
            return

        params = [_qiskit_expr_to_qsl(p, cache)
                  for p in getattr(op, "params", [])]
        try:
            out.append(make(name, params), tuple(qubits))
            return
        except ValueError:
            pass

        # 多控制 X / Z
        base = getattr(op, "base_gate", None)
        num_ctrl = getattr(op, "num_ctrl_qubits", 0)
        if name == "mcx" or (num_ctrl >= 2 and base is not None
                             and base.name == "x"):
            out.mcx(qubits[:-1], qubits[-1])
            return
        if name == "mcz" or (num_ctrl >= 1 and base is not None
                             and base.name == "z"):
            out.mcz(qubits)
            return

        # 有定义则递归展开 (最多 10 层); 定义自带的全局相位累积进来
        definition = getattr(op, "definition", None)
        if definition is not None and depth < 10:
            out.global_phase += float(getattr(definition, "global_phase", 0.0))
            for dinst in definition.data:
                mapped = [qubits[definition.find_bit(q).index]
                          for q in dinst.qubits]
                convert_operation(dinst.operation, mapped, depth + 1)
            return

        # 兜底: 酉矩阵包成 unitary 门
        # (qiskit 矩阵 qargs[0]=最低位, qsl t0=最高位, 需反转比特序)
        try:
            mat = np.asarray(op.to_matrix(), dtype=complex)
        except Exception as exc:
            raise ValueError(
                f"无法转换 qiskit 门 {name!r}: 无定义且无矩阵"
            ) from exc
        out.unitary(mat, qubits[::-1], label=name)

    for qinst in qc.data:
        qubits = [qc.find_bit(q).index for q in qinst.qubits]
        convert_operation(qinst.operation, qubits, 0)

    return out


# ====================================================================
# to_cirq
# ====================================================================

def to_cirq(circuit: QuantumCircuit):
    """
    把 qsl QuantumCircuit 转换为 cirq.Circuit。

    qsl 比特索引 q -> cirq.LineQubit(q);
    qsl Parameter -> sympy.Symbol (同名);
    不认识的门用 cirq.MatrixGate 包装; barrier 被丢弃 (cirq 无此概念);
    global_phase 用 cirq.global_phase_operation 记录。
    未安装 cirq 时抛出 ImportError (提示 pip install cirq)。
    """
    try:
        import cirq
        import sympy
    except ImportError as exc:
        raise ImportError(
            "to_cirq 需要 cirq, 请先安装: pip install cirq"
        ) from exc

    cache: Dict[str, object] = {}

    def to_sym(expr):
        if isinstance(expr, Parameter):
            if expr.name not in cache:
                cache[expr.name] = sympy.Symbol(expr.name)
            return cache[expr.name]
        if isinstance(expr, ParameterExpression):
            args = [to_sym(a) for a in expr._args]
            op = expr._op
            if op == "add":
                return args[0] + args[1]
            if op == "sub":
                return args[0] - args[1]
            if op == "mul":
                return args[0] * args[1]
            if op == "div":
                return args[0] / args[1]
            if op == "pow":
                return args[0] ** args[1]
            if op == "neg":
                return -args[0]
            if op in ("sin", "cos", "tan", "exp", "log", "sqrt"):
                return getattr(sympy, op)(args[0])
            raise ValueError(f"无法转换为 sympy 的 qsl 表达式: {expr!r}")
        return expr

    qubits = [cirq.LineQubit(i) for i in range(circuit.num_qubits)]
    ops: List = []

    simple = {
        "h": cirq.H, "x": cirq.X, "y": cirq.Y, "z": cirq.Z,
        "s": cirq.S, "t": cirq.T,
        "cx": cirq.CNOT, "cz": cirq.CZ, "swap": cirq.SWAP,
        "ccx": cirq.CCNOT, "cswap": cirq.CSWAP,
        "iswap": cirq.ISWAP,
    }

    for inst in circuit.data:
        gate = inst.gate
        name = gate.name
        qs = [qubits[q] for q in inst.qubits]
        params = [to_sym(p) for p in gate.params]

        if name in simple:
            ops.append(simple[name].on(*qs))
        elif name == "sdg":
            ops.append((cirq.S ** -1).on(*qs))
        elif name == "tdg":
            ops.append((cirq.T ** -1).on(*qs))
        elif name == "sx":
            ops.append((cirq.X ** 0.5).on(*qs))
        elif name == "sxdg":
            ops.append((cirq.X ** -0.5).on(*qs))
        elif name == "id":
            ops.append(cirq.I.on(*qs))
        elif name == "rx":
            ops.append(cirq.rx(params[0]).on(*qs))
        elif name == "ry":
            ops.append(cirq.ry(params[0]).on(*qs))
        elif name == "rz":
            ops.append(cirq.rz(params[0]).on(*qs))
        elif name == "p":
            ops.append(cirq.ZPowGate(exponent=params[0] / math.pi).on(*qs))
        elif name == "cp":
            ops.append(cirq.CZPowGate(exponent=params[0] / math.pi).on(*qs))
        elif name == "rxx":
            ops.append(cirq.XXPowGate(exponent=params[0] / math.pi,
                                      global_shift=-0.5).on(*qs))
        elif name == "ryy":
            ops.append(cirq.YYPowGate(exponent=params[0] / math.pi,
                                      global_shift=-0.5).on(*qs))
        elif name == "rzz":
            ops.append(cirq.ZZPowGate(exponent=params[0] / math.pi,
                                      global_shift=-0.5).on(*qs))
        elif name == "barrier":
            continue  # cirq 无 barrier 概念, 直接丢弃
        else:
            # 其余门 (u/cu/crx/cry/crz/cy/ch/cs/.../mcx/mcz/unitary):
            # 统一用 MatrixGate 保证数值正确
            ops.append(cirq.MatrixGate(gate.to_matrix()).on(*qs))

    result = cirq.Circuit(ops)
    if circuit.global_phase:
        result = cirq.Circuit(
            cirq.global_phase_operation(
                complex(np.exp(1j * circuit.global_phase)))) + result
    return result
