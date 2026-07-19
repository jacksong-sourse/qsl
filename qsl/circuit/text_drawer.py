"""ASCII 文本电路绘图器。"""

from __future__ import annotations

from typing import List

from .circuit import QuantumCircuit, Instruction

_CONTROLLED_PREFIXES = ("cx", "cy", "cz", "ch", "cs", "csdg", "ct", "ctdg",
                        "crx", "cry", "crz", "cp", "cu")
_SINGLE_QUBIT_GATES = {"id", "x", "y", "z", "h", "s", "sdg", "t", "tdg",
                       "sx", "sxdg", "rx", "ry", "rz", "p", "u"}


def draw_text(circuit: QuantumCircuit, fold: int = -1) -> str:
    """
    把电路绘制为 ASCII 图。

    参数:
        fold: 每行最多列数, -1 表示不换行
    返回:
        str 电路图
    """
    n = circuit.num_qubits
    insts: List[Instruction] = list(circuit.data)

    # 为每条指令计算显示标签与控制/目标结构
    cells = []  # (per_qubit_render_fn, span_qubits, label_info)
    columns: List[dict] = []
    for inst in insts:
        col = _render_instruction(inst, n)
        columns.append(col)

    # 计算每列宽度
    if not columns:
        body = [f"q{i}: ───" for i in range(n)]
        return "\n".join(body)

    widths = []
    for col in columns:
        w = max(len(c) for c in col["cells"])
        widths.append(max(w, 3))

    lines = [[] for _ in range(n)]
    for col, w in zip(columns, widths):
        for qi in range(n):
            cell = col["cells"][qi]
            lines[qi].append(cell.center(w, "─") if cell.startswith("─")
                             else cell.ljust(w, "─"))

    out_lines = []
    for qi in range(n):
        out_lines.append(f"q{qi}: " + "".join(lines[qi]))
    return "\n".join(out_lines)


def _render_instruction(inst: Instruction, n: int) -> dict:
    g = inst.gate
    name = g.name
    qs = inst.qubits
    cells = ["───"] * n

    if name == "barrier":
        for qi in range(n):
            cells[qi] = "░"
        return {"cells": cells}

    label = g.label
    if g.params:
        try:
            ps = ", ".join(
                f"{float(p):.3g}" if not hasattr(p, "parameters") or not p.parameters
                else str(p)
                for p in g.params
            )
            label = f"{label}({ps})"
        except Exception:
            label = f"{label}(?)"

    if len(qs) == 1:
        cells[qs[0]] = f"[{label}]"
        return {"cells": cells}

    # 多比特门: 判断控制/目标结构
    if name in _CONTROLLED_PREFIXES or name in ("ccx", "cswap") or name.startswith("mcx"):
        if name == "cswap":
            controls = [qs[0]]
            targets = list(qs[1:])
            for c in controls:
                cells[c] = "■"
            for t in targets:
                cells[t] = "×"
        elif name.startswith("mcx") or name == "ccx":
            controls = list(qs[:-1])
            target = qs[-1]
            for c in controls:
                cells[c] = "■"
            cells[target] = f"[{label}]"
        else:
            control, target = qs[0], qs[1]
            cells[control] = "■"
            cells[target] = f"[{label}]"
    elif name in ("swap", "iswap"):
        cells[qs[0]] = "×"
        cells[qs[1]] = "×" if name == "swap" else "[i×]"
    elif name.startswith("mcz"):
        for q in qs[:-1]:
            cells[q] = "■"
        cells[qs[-1]] = f"[{label}]"
    else:
        lo, hi = min(qs), max(qs)
        for q in qs:
            cells[q] = f"[{label}]" if q == lo else "║"

    # 控制-目标连线
    involved = sorted(qs)
    for qi in range(involved[0] + 1, involved[-1]):
        if cells[qi] == "───":
            cells[qi] = "│"
    return {"cells": cells}


__all__ = ["draw_text"]
