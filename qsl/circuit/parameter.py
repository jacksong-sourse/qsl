"""
符号参数 (Parameter) 模块 — 参数化量子电路的地基。

支持:
    - Parameter: 命名字符串符号，可参与算术表达式
    - ParameterExpression: 符号表达式树 (加/减/乘/除/幂/取负)
    - bind(): 将符号绑定到具体数值

示例:
    >>> from qsl.circuit import Parameter
    >>> theta = Parameter("θ")
    >>> expr = 2 * theta + 1
    >>> expr.bind({theta: 0.5})
    2.0
"""

from __future__ import annotations

import math
from typing import Dict, Union, Set

Number = Union[int, float, complex]


class ParameterExpression:
    """符号参数表达式。可嵌套组合，延迟到 bind() 时求值。"""

    __slots__ = ("_op", "_args")

    _OPS = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b,
        "pow": lambda a, b: a ** b,
        "neg": lambda a: -a,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "exp": math.exp,
        "log": math.log,
        "sqrt": math.sqrt,
    }

    _SYMBOLS = {
        "add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**",
    }

    def __init__(self, op: str, args: tuple):
        self._op = op
        self._args = args

    # ------------------------------------------------------------
    # 求值
    # ------------------------------------------------------------
    def bind(self, mapping: Dict["Parameter", Number]) -> Number:
        """递归绑定符号到数值，返回计算结果。"""
        values = []
        for arg in self._args:
            if isinstance(arg, Parameter):
                if arg not in mapping:
                    raise ValueError(
                        f"参数 '{arg.name}' 未绑定值。"
                        f"请在 mapping 中提供 {{{arg.name!r}: value}}。"
                    )
                values.append(mapping[arg])
            elif isinstance(arg, ParameterExpression):
                values.append(arg.bind(mapping))
            else:
                values.append(arg)
        return self._OPS[self._op](*values)

    @property
    def parameters(self) -> Set["Parameter"]:
        """表达式中涉及的所有自由符号。"""
        out: Set[Parameter] = set()
        for arg in self._args:
            if isinstance(arg, Parameter):
                out.add(arg)
            elif isinstance(arg, ParameterExpression):
                out |= arg.parameters
        return out

    @property
    def is_bound(self) -> bool:
        """是否为纯数值表达式（不含自由符号）。"""
        return len(self.parameters) == 0

    # ------------------------------------------------------------
    # 运算符重载
    # ------------------------------------------------------------
    def _wrap(self, other):
        return other

    def __add__(self, other):
        return ParameterExpression("add", (self, self._wrap(other)))

    def __radd__(self, other):
        return ParameterExpression("add", (self._wrap(other), self))

    def __sub__(self, other):
        return ParameterExpression("sub", (self, self._wrap(other)))

    def __rsub__(self, other):
        return ParameterExpression("sub", (self._wrap(other), self))

    def __mul__(self, other):
        return ParameterExpression("mul", (self, self._wrap(other)))

    def __rmul__(self, other):
        return ParameterExpression("mul", (self._wrap(other), self))

    def __truediv__(self, other):
        return ParameterExpression("div", (self, self._wrap(other)))

    def __rtruediv__(self, other):
        return ParameterExpression("div", (self._wrap(other), self))

    def __pow__(self, other):
        return ParameterExpression("pow", (self, self._wrap(other)))

    def __rpow__(self, other):
        return ParameterExpression("pow", (self._wrap(other), self))

    def __neg__(self):
        return ParameterExpression("neg", (self,))

    # ------------------------------------------------------------
    # 显示
    # ------------------------------------------------------------
    def __repr__(self) -> str:
        if self._op in self._SYMBOLS:
            sym = self._SYMBOLS[self._op]
            return f"({self._args[0]!r} {sym} {self._args[1]!r})"
        if self._op == "neg":
            return f"(-{self._args[0]!r})"
        return f"{self._op}({', '.join(repr(a) for a in self._args)})"

    def __str__(self) -> str:
        def fmt(a):
            if isinstance(a, Parameter):
                return a.name
            if isinstance(a, ParameterExpression):
                return str(a)
            if isinstance(a, float) and a == int(a):
                return str(int(a))
            return str(a)

        if self._op in self._SYMBOLS:
            return f"({fmt(self._args[0])}{self._SYMBOLS[self._op]}{fmt(self._args[1])})"
        if self._op == "neg":
            return f"(-{fmt(self._args[0])})"
        return f"{self._op}({', '.join(fmt(a) for a in self._args)})"


class Parameter(ParameterExpression):
    """
    命名符号参数，如 VQE/QAOA 中的变分参数。

    参数:
        name: 参数名（字符串），如 "θ", "beta0"

    示例:
        >>> theta = Parameter("θ")
        >>> circuit.rx(0, theta)
        >>> bound = circuit.bind_parameters({theta: 3.14159})
    """

    __slots__ = ("_name",)

    def __init__(self, name: str):
        if not isinstance(name, str) or not name:
            raise ValueError("Parameter name 必须是非空字符串")
        self._name = name
        # Parameter 本身也是叶子表达式（bind 时直接查表）
        super().__init__("leaf", ())

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> Set["Parameter"]:
        return {self}

    def bind(self, mapping: Dict["Parameter", Number]) -> Number:
        if self not in mapping:
            raise ValueError(f"参数 '{self._name}' 未绑定值。")
        return mapping[self]

    def __hash__(self) -> int:
        return hash(("Parameter", self._name))

    def __eq__(self, other) -> bool:
        return isinstance(other, Parameter) and other._name == self._name

    def __repr__(self) -> str:
        return f"Parameter({self._name})"

    def __str__(self) -> str:
        return self._name


def sin(x):
    """符号正弦（若输入为表达式则延迟求值）。"""
    if isinstance(x, ParameterExpression):
        return ParameterExpression("sin", (x,))
    return math.sin(x)


def cos(x):
    """符号余弦。"""
    if isinstance(x, ParameterExpression):
        return ParameterExpression("cos", (x,))
    return math.cos(x)


def exp(x):
    """符号指数。"""
    if isinstance(x, ParameterExpression):
        return ParameterExpression("exp", (x,))
    return math.exp(x)


def sqrt(x):
    """符号平方根。"""
    if isinstance(x, ParameterExpression):
        return ParameterExpression("sqrt", (x,))
    return math.sqrt(x)


def resolve(value) -> float:
    """将 数值/已绑定表达式 解析为 float。含自由符号时报错。"""
    if isinstance(value, ParameterExpression):
        params = value.parameters
        if params:
            names = ", ".join(sorted(p.name for p in params))
            raise ValueError(
                f"表达式仍含未绑定参数: {names}。"
                f"请先调用 circuit.bind_parameters()。"
            )
        return float(value.bind({}))
    return float(value)


__all__ = ["Parameter", "ParameterExpression", "sin", "cos", "exp", "sqrt", "resolve"]
