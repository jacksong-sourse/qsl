"""
布尔表达式解析器。

将中缀布尔表达式 (如 "x0 & x1 | ~x2") 解析为抽象语法树 (AST)，
每个 AST 节点支持 evaluate(assignment) 方法在给定位赋值下求值。

语法 (BNF, 按优先级递增):
    expr     := xor_expr ('|' xor_expr)*          # OR, 最低优先级
    xor_expr := and_expr ('^' and_expr)*           # XOR
    and_expr := not_expr ('&' not_expr)*           # AND, 最高优先级
    not_expr := '~' not_expr | atom                # NOT (右结合)
    atom     := IDENTIFIER | '(' expr ')'          # 原子
    IDENTIFIER := [a-zA-Z_][a-zA-Z0-9_]*

特殊形式:
    - "x" 后跟数字 (如 x0, x1, x12) 被解析为变量 VarExpr(index=N)
    - 其他标识符 (如 constraint_a) 保留为命名字段 (暂按变量处理)

失败模式分析:
    1. 空表达式: 无有效语法树
    2. 不匹配的括号: 解析位置异常
    3. 未知运算符: 如使用了 & | ^ ~ 之外的符号
    4. 变量索引过大: 在 evaluate 时 assignment 可能只覆盖部分位
    5. 深度嵌套: 递归下降可能栈溢出 (由 Python 递归限制保护)
    6. 运算符优先级混淆: 如 "x0 & x1 | x2" 的正确性是设计选择
"""

from typing import Callable
from ..utils.exceptions import BooleanParseError


# ----------------------------------------------------------------
# AST 节点类
# ----------------------------------------------------------------

class BooleanExpr:
    """布尔表达式 AST 的抽象基类。"""

    def evaluate(self, assignment: int) -> bool:
        """
        在给定整数值赋下求值。

        参数:
            assignment: 整数，第 i 位对应变量 x_i 的值

        返回:
            布尔表达式的结果
        """
        raise NotImplementedError

    def to_string(self) -> str:
        """返回人类可读的表达式字符串。"""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.to_string()}>"


class VarExpr(BooleanExpr):
    """变量节点: x[k] 表示第 k 位。"""

    def __init__(self, index: int):
        self.index = index

    def evaluate(self, assignment: int) -> bool:
        return bool((assignment >> self.index) & 1)

    def to_string(self) -> str:
        return f"x{self.index}"


class NotExpr(BooleanExpr):
    """逻辑非: ~expr。"""

    def __init__(self, expr: BooleanExpr):
        self.expr = expr

    def evaluate(self, assignment: int) -> bool:
        return not self.expr.evaluate(assignment)

    def to_string(self) -> str:
        inner = self.expr.to_string()
        if isinstance(self.expr, VarExpr):
            return f"~{inner}"
        return f"~({inner})"


class AndExpr(BooleanExpr):
    """逻辑与: left & right。"""

    def __init__(self, left: BooleanExpr, right: BooleanExpr):
        self.left = left
        self.right = right

    def evaluate(self, assignment: int) -> bool:
        return self.left.evaluate(assignment) and self.right.evaluate(assignment)

    def to_string(self) -> str:
        return f"({self.left.to_string()} & {self.right.to_string()})"


class OrExpr(BooleanExpr):
    """逻辑或: left | right。"""

    def __init__(self, left: BooleanExpr, right: BooleanExpr):
        self.left = left
        self.right = right

    def evaluate(self, assignment: int) -> bool:
        return self.left.evaluate(assignment) or self.right.evaluate(assignment)

    def to_string(self) -> str:
        return f"({self.left.to_string()} | {self.right.to_string()})"


class XorExpr(BooleanExpr):
    """逻辑异或: left ^ right。"""

    def __init__(self, left: BooleanExpr, right: BooleanExpr):
        self.left = left
        self.right = right

    def evaluate(self, assignment: int) -> bool:
        return self.left.evaluate(assignment) != self.right.evaluate(assignment)

    def to_string(self) -> str:
        return f"({self.left.to_string()} ^ {self.right.to_string()})"


# ----------------------------------------------------------------
# 递归下降解析器
# ----------------------------------------------------------------

class BooleanParser:
    """
    布尔表达式递归下降解析器。

    使用经典的递归下降法，每个非终结符对应一个 _parse_* 方法。
    通过 lookahead (预读) 一个字符来决定使用哪条产生式。

    属性:
        source: 原始表达式字符串
        pos: 当前解析位置
    """

    def __init__(self, source: str):
        self.source = source
        self.pos = 0

    def parse(self) -> BooleanExpr:
        """
        解析完整表达式。

        返回:
            AST 根节点

        失败模式:
            - 空输入: 抛出 BooleanParseError
            - 语法错误: 抛出 BooleanParseError (带位置信息)
            - 多余的尾随字符: 抛出 BooleanParseError
        """
        self._skip_whitespace()
        if self.pos >= len(self.source):
            raise BooleanParseError(
                "表达式不能为空", self.source, self.pos
            )
        result = self._parse_expr()
        self._skip_whitespace()
        if self.pos < len(self.source):
            raise BooleanParseError(
                f"表达式末尾有未预期的字符 '{self.source[self.pos]}'",
                self.source, self.pos
            )
        return result

    # --- 词法层 ---

    def _skip_whitespace(self):
        """跳过空白字符。"""
        while self.pos < len(self.source) and self.source[self.pos] in ' \t\n\r':
            self.pos += 1

    def _peek(self) -> str:
        """预读当前字符 (不消耗)。"""
        self._skip_whitespace()
        if self.pos < len(self.source):
            return self.source[self.pos]
        return ''

    def _consume(self) -> str:
        """消耗并返回当前字符。"""
        self._skip_whitespace()
        ch = self.source[self.pos]
        self.pos += 1
        return ch

    # --- 语法层 (按优先级从低到高) ---

    def _parse_expr(self) -> BooleanExpr:
        """
        expr := xor_expr ('|' xor_expr)*
        OR 是最外层运算符，优先级最低。
        """
        left = self._parse_xor_expr()
        while self._peek() == '|':
            self._consume()
            right = self._parse_xor_expr()
            left = OrExpr(left, right)
        return left

    def _parse_xor_expr(self) -> BooleanExpr:
        """
        xor_expr := and_expr ('^' and_expr)*
        """
        left = self._parse_and_expr()
        while self._peek() == '^':
            self._consume()
            right = self._parse_and_expr()
            left = XorExpr(left, right)
        return left

    def _parse_and_expr(self) -> BooleanExpr:
        """
        and_expr := not_expr ('&' not_expr)*
        AND 优先级最高 (除 NOT 外)。
        """
        left = self._parse_not_expr()
        while self._peek() == '&':
            self._consume()
            right = self._parse_not_expr()
            left = AndExpr(left, right)
        return left

    def _parse_not_expr(self) -> BooleanExpr:
        """
        not_expr := '~' not_expr | atom

        NOT 是前缀运算符，右结合。
        ～～x0 等价于 x0。
        """
        if self._peek() == '~':
            self._consume()
            expr = self._parse_not_expr()
            return NotExpr(expr)
        return self._parse_atom()

    def _parse_atom(self) -> BooleanExpr:
        """
        atom := IDENTIFIER | '(' expr ')'

        失败模式:
            - 不匹配的右括号: 抛出 BooleanParseError
            - 无效的标识符: 抛出 BooleanParseError
        """
        ch = self._peek()
        if ch == '(':
            self._consume()  # 吃掉 '('
            expr = self._parse_expr()
            if self._peek() != ')':
                raise BooleanParseError(
                    "缺少右括号 ')'", self.source, self.pos
                )
            self._consume()  # 吃掉 ')'
            return expr
        elif ch == '':
            raise BooleanParseError(
                "意外的表达式结尾", self.source, self.pos
            )
        else:
            ident = self._parse_identifier()
            # Special handling for "x" followed by digits: VarExpr(index=N)
            import re
            x_match = re.match(r'^x(\d+)$', ident)
            if x_match:
                return VarExpr(int(x_match.group(1)))
            # Reject non-xN identifiers with clear error
            raise BooleanParseError(
                f"不支持的变量标识符: '{ident}'。"
                f"支持的格式: x0, x1, x2, ... (x后跟数字)。"
                f"当前变量: '{ident}'",
                self.source, self.pos - len(ident)
            )

    def _parse_identifier(self) -> str:
        """
        解析标识符: [a-zA-Z_][a-zA-Z0-9_]*

        失败模式:
            - 以数字开头: 抛出 BooleanParseError
            - 包含非法字符: 被截断 (非严格模式)

        返回:
            标识符字符串
        """
        self._skip_whitespace()
        start = self.pos
        ch = self.source[start] if start < len(self.source) else ''
        if not (ch.isalpha() or ch == '_'):
            raise BooleanParseError(
                f"期望标识符，但遇到 '{ch}'", self.source, start
            )
        self.pos += 1
        while self.pos < len(self.source) and (
            self.source[self.pos].isalnum() or self.source[self.pos] == '_'
        ):
            self.pos += 1
        return self.source[start:self.pos]


# ----------------------------------------------------------------
# 便捷函数
# ----------------------------------------------------------------

def parse_bool(expression: str) -> BooleanExpr:
    """
    解析布尔表达式字符串为 AST。

    参数:
        expression: 如 "x0 & x1 | ~x2"

    返回:
        BooleanExpr AST 节点

    失败模式:
        - 语法错误: 抛出 BooleanParseError (带源码位置)
    """
    return BooleanParser(expression).parse()


def build_oracle_function(expressions: list) -> Callable[[int], bool]:
    """
    从多个布尔表达式构建组合 Oracle 函数。

    所有表达式必须同时满足 (逻辑与):
        f(x) = AND_{expr in expressions} expr.evaluate(x)

    参数:
        expressions: BooleanExpr 列表

    返回:
        可调用的 oracle 函数 f(x) -> bool

    失败模式:
        - expressions 为空: 返回恒真函数 (search all)
    """
    if not expressions:
        return lambda x: True
    return lambda x: all(e.evaluate(x) for e in expressions)
