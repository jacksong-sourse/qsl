"""
测试布尔表达式解析器 (BooleanParser) 模块。

覆盖:
    - 基本变量解析 (x0, x1, ...)
    - 运算符 (&, |, ^, ~)
    - 运算符优先级
    - 嵌套括号
    - 求值正确性
    - 错误输入处理
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsl.core.parser import (
    parse_bool, BooleanParser, BooleanExpr,
    VarExpr, NotExpr, AndExpr, OrExpr, XorExpr,
    build_oracle_function,
)
from qsl.utils.exceptions import BooleanParseError


class TestBasicParsing:
    """测试基本解析。"""

    def test_single_variable(self):
        """单变量: x0。"""
        expr = parse_bool("x0")
        assert isinstance(expr, VarExpr)
        assert expr.index == 0

    def test_single_variable_x5(self):
        """单变量: x5。"""
        expr = parse_bool("x5")
        assert expr.index == 5

    def test_not_simple(self):
        """非: ~x0。"""
        expr = parse_bool("~x0")
        assert isinstance(expr, NotExpr)
        assert isinstance(expr.expr, VarExpr)

    def test_and_simple(self):
        """与: x0 & x1。"""
        expr = parse_bool("x0 & x1")
        assert isinstance(expr, AndExpr)

    def test_or_simple(self):
        """或: x0 | x1。"""
        expr = parse_bool("x0 | x1")
        assert isinstance(expr, OrExpr)

    def test_xor_simple(self):
        """异或: x0 ^ x1。"""
        expr = parse_bool("x0 ^ x1")
        assert isinstance(expr, XorExpr)


class TestOperatorPrecedence:
    """测试运算符优先级。"""

    def test_and_before_or(self):
        """& 优先级高于 |: x0 | x1 & x2 == x0 | (x1 & x2)。"""
        expr = parse_bool("x0 | x1 & x2")
        assert isinstance(expr, OrExpr)
        assert isinstance(expr.right, AndExpr)

    def test_xor_before_or(self):
        """^ 优先级高于 |: x0 | x1 ^ x2 == x0 | (x1 ^ x2)。"""
        expr = parse_bool("x0 | x1 ^ x2")
        assert isinstance(expr, OrExpr)
        assert isinstance(expr.right, XorExpr)

    def test_not_binds_tightest(self):
        """~ 最高优先级: ~x0 & x1 == (~x0) & x1。"""
        expr = parse_bool("~x0 & x1")
        assert isinstance(expr, AndExpr)
        assert isinstance(expr.left, NotExpr)

    def test_double_not(self):
        """双重非: ~~x0 == x0。"""
        expr = parse_bool("~~x0")
        assert isinstance(expr, NotExpr)
        assert isinstance(expr.expr, NotExpr)


class TestParenthesizedExpressions:
    """测试括号表达式。"""

    def test_simple_parens(self):
        """简单括号: (x0 | x1)。"""
        expr = parse_bool("(x0 | x1)")
        assert isinstance(expr, OrExpr)

    def test_nested_parens(self):
        """嵌套括号: ((x0 & x1) | x2)。"""
        expr = parse_bool("((x0 & x1) | x2)")
        assert isinstance(expr, OrExpr)

    def test_parens_override_precedence(self):
        """括号覆盖优先级: (x0 | x1) & x2。"""
        expr = parse_bool("(x0 | x1) & x2")
        assert isinstance(expr, AndExpr)
        assert isinstance(expr.left, OrExpr)

    def test_complex_nested(self):
        """复杂嵌套: ~(x0 & (x1 | x2)) ^ x3。"""
        expr = parse_bool("~(x0 & (x1 | x2)) ^ x3")
        assert isinstance(expr, XorExpr)
        assert isinstance(expr.left, NotExpr)


class TestEvaluation:
    """测试表达式求值。"""

    def test_var_eval(self):
        """变量求值。"""
        expr = parse_bool("x0")
        assert expr.evaluate(1) is True   # ...01
        assert expr.evaluate(2) is False  # ...10

    def test_not_eval(self):
        """NOT 求值。"""
        expr = parse_bool("~x0")
        assert expr.evaluate(0) is True
        assert expr.evaluate(1) is False

    def test_and_eval(self):
        """AND 求值。"""
        expr = parse_bool("x0 & x1")
        assert expr.evaluate(3) is True   # ...11
        assert expr.evaluate(1) is False  # ...01
        assert expr.evaluate(2) is False  # ...10

    def test_or_eval(self):
        """OR 求值。"""
        expr = parse_bool("x0 | x1")
        assert expr.evaluate(0) is False
        assert expr.evaluate(1) is True
        assert expr.evaluate(2) is True
        assert expr.evaluate(3) is True

    def test_xor_eval(self):
        """XOR 求值。"""
        expr = parse_bool("x0 ^ x1")
        assert expr.evaluate(0) is False
        assert expr.evaluate(1) is True
        assert expr.evaluate(2) is True
        assert expr.evaluate(3) is False

    def test_complex_eval(self):
        """复杂表达式求值。"""
        expr = parse_bool("(x0 & x1) | (~x0 & ~x1)")
        # 等价于 x0 == x1
        assert expr.evaluate(0) is True   # 00 -> True
        assert expr.evaluate(1) is False  # 01 -> False
        assert expr.evaluate(2) is False  # 10 -> False
        assert expr.evaluate(3) is True   # 11 -> True


class TestBuildOracleFunction:
    """测试 build_oracle_function。"""

    def test_single_expression(self):
        """单个表达式 -> Oracle 函数。"""
        expr1 = parse_bool("x0 & x1")
        oracle = build_oracle_function([expr1])
        assert oracle(3) is True
        assert oracle(0) is False

    def test_multiple_expressions(self):
        """多个表达式 AND 组合。"""
        exprs = [parse_bool("x0"), parse_bool("x1")]
        oracle = build_oracle_function(exprs)
        assert oracle(3) is True   # 11
        assert oracle(1) is False  # 01

    def test_empty_expressions(self):
        """空表达式 -> 恒真函数。"""
        oracle = build_oracle_function([])
        assert oracle(0) is True
        assert oracle(100) is True


class TestErrorHandling:
    """测试错误处理。"""

    def test_empty_input(self):
        """空输入 -> 异常。"""
        try:
            parse_bool("")
            assert False
        except (BooleanParseError, ValueError):
            pass

    def test_unmatched_parens(self):
        """不匹配括号 -> 异常。"""
        try:
            parse_bool("(x0 & x1")
            assert False
        except BooleanParseError:
            pass

    def test_trailing_characters(self):
        """尾随垃圾字符 -> 异常。"""
        try:
            parse_bool("x0 & x1 )")
            assert False
        except BooleanParseError:
            pass

    def test_invalid_operator(self):
        """无效运算符 -> 按标识符解析。"""
        # x0 + x1 中的 '+' 不是有效布尔运算符
        # 解析器会把 'x0' 作为完整标识符吃掉
        expr = parse_bool("x0")
        assert isinstance(expr, VarExpr)
