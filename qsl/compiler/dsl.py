"""
QSL DSL 文本语法解析器。

将 QSL 领域特定语言的文本源码解析为 QSLProgram 数据结构。

语法:
    program "名称" {
        qubits: <整数>

        premise {
            <布尔表达式>
            ...
        }

        tools {
            oracle: <策略>
        }

        question {
            find: <目标>
            where: <条件>
        }

        main {
            algorithm: <算法名>
            shots: <整数>
            backend: <后端名>
        }
    }

注释: // 或 # 开头的行

失败模式分析:
    1. 无 program 声明: 解析器返回 None -> 报错
    2. qubits 缺失: n_qubits 保持 0 -> 后续验证会报错
    3. premise 段缺失: premises 保持 [] -> 无解搜索
    4. 嵌套括号不匹配: 段无法正确关闭
    5. 重复字段: 后者覆盖前者
    6. 无效数值: int() 转换失败
    7. 特殊字符: 表达式直接原样保留，由 BooleanParser 验证
"""

import re
from typing import Optional

from .program import QSLProgram
from ..utils.exceptions import DSLParseError


# 有效的算法列表
SUPPORTED_ALGORITHMS = {"grover", "shor", "qaoa", "vqe"}

# 有效的后端列表 (允许扩展)
SUPPORTED_BACKENDS = {"simulator", "ibm"}


def parse_qsl(source: str) -> QSLProgram:
    """
    解析 QSL DSL 源码字符串。

    参数:
        source: QSL 源码文本

    返回:
        QSLProgram 实例

    失败模式:
        - 源码为空: 抛出 DSLParseError
        - 缺少 program 声明: 抛出 DSLParseError
        - 语法错误: 抛出 DSLParseError (带行号)
    """
    if not source or not source.strip():
        raise DSLParseError("QSL 源码不能为空")

    program: Optional[QSLProgram] = None
    program_name: str = ""
    current_section: Optional[str] = None
    section_depth: int = 0

    lines = source.split('\n')

    for line_no, raw_line in enumerate(lines, 1):
        # 去除注释 (支持 // 和 #)
        line = _strip_comment(raw_line)
        line = line.strip()

        if not line:
            continue

        # --- 程序声明 ---
        if line.startswith('program '):
            match = re.match(r'program\s+"([^"]*)"\s*\{?', line)
            if not match:
                raise DSLParseError(
                    f"第 {line_no} 行: program 声明格式错误, "
                    f"应为: program \"名称\" {{",
                    source, _line_offset(source, line_no)
                )
            program_name = match.group(1)
            program = QSLProgram(name=program_name, n_qubits=0)
            current_section = None
            section_depth = 1 if '{' in line else 0
            continue

        # --- qubits 声明 ---
        if 'qubits:' in line.lower():
            match = re.search(r'qubits\s*:\s*(\d+)', line, re.IGNORECASE)
            if not match:
                raise DSLParseError(
                    f"第 {line_no} 行: qubits 格式错误, 应为: qubits: <整数>",
                    source, _line_offset(source, line_no)
                )
            qubits = int(match.group(1))
            if program is None:
                raise DSLParseError(
                    f"第 {line_no} 行: qubits 必须在 program 声明之后",
                    source, _line_offset(source, line_no)
                )
            program.n_qubits = qubits
            continue

        # --- 段开始 ---
        section_start = _detect_section_start(line)
        if section_start:
            current_section = section_start
            # Count brace outside of quoted strings
            if _count_braces_outside_strings(line) > 0:
                section_depth += 1
            continue

        # --- 段结束 ---
        if line == '}':
            section_depth -= 1
            if section_depth <= 0:
                current_section = None
            continue

        # --- 段内容 ---
        if program is None:
            raise DSLParseError(
                f"第 {line_no} 行: 内容必须在 program 声明之后",
                source, _line_offset(source, line_no)
            )

        if current_section == 'premise':
            # 跳过可能的关键词前缀
            content = line
            for prefix in ['constraint:', 'premise:']:
                if content.lower().startswith(prefix):
                    content = content[content.index(':') + 1:].strip()
                    break
            if content:
                program.premises.append(content)

        elif current_section == 'tools':
            if 'oracle:' in line:
                program.tools.append(line.strip())
            elif ':' in line:
                # 忽略不认识的工具行
                pass

        elif current_section == 'question':
            # 问题描述段目前仅记录不解析
            pass

        elif current_section == 'main':
            if 'algorithm:' in line:
                alg_match = re.search(r'algorithm\s*:\s*(\w+)', line, re.IGNORECASE)
                if alg_match:
                    program.main_algorithm = alg_match.group(1).lower()
            if 'shots:' in line:
                shots_match = re.search(r'shots\s*:\s*(\d+)', line, re.IGNORECASE)
                if shots_match:
                    program.shots = int(shots_match.group(1))
            if 'backend:' in line:
                backend_match = re.search(
                    r'backend\s*:\s*(\S+)', line, re.IGNORECASE
                )
                if backend_match:
                    program.backend = backend_match.group(1).lower()

    # 后处理验证
    if program is None:
        raise DSLParseError(
            "解析失败: 未找到 'program \"名称\"' 声明",
            source
        )

    if program.n_qubits <= 0:
        raise DSLParseError(
            f"程序 '{program.name}' 缺少 qubits 声明",
            source
        )

    # 验证程序
    try:
        program.validate()
    except Exception as e:
        raise DSLParseError(
            f"程序 '{program.name}' 验证失败: {e}",
            source
        )

    return program


def _strip_comment(line: str) -> str:
    """移除行内注释 (// 或 # 之后的部分)。"""
    for marker in ('//', '#'):
        idx = line.find(marker)
        if idx >= 0:
            return line[:idx]
    return line


def _detect_section_start(line: str) -> Optional[str]:
    """检测行是否是段开始，返回段名。"""
    clean = line.strip().lower()
    # 移除开括号
    name = clean.split('{')[0].strip() if '{' in clean else clean
    valid_sections = {'premise', 'tools', 'question', 'main'}
    if name in valid_sections:
        return name
    return None


def _count_braces_outside_strings(line: str) -> int:
    """Count '{' characters that are not inside double-quoted strings."""
    count = 0
    in_string = False
    for i, ch in enumerate(line):
        if ch == '"' and (i == 0 or line[i - 1] != '\\'):
            in_string = not in_string
        elif ch == '{' and not in_string:
            count += 1
    return count


def _line_offset(source: str, line_no: int) -> int:
    """计算给定行号在源码字符串中的字符偏移量。"""
    lines = source.split('\n')
    offset = 0
    for i in range(min(line_no - 1, len(lines))):
        offset += len(lines[i]) + 1  # +1 for newline
    return offset
