[← 返回文档中心](../index.md)

# QSL AI 量子科学家：自然语言驱动的量子算法设计

欢迎使用 QSL 的 AI 量子科学家模块！本教程将带你体验如何用自然语言描述问题，让 AI 自动设计、实现并验证量子算法——无需手动编写量子电路。

---

## 🤖 什么是 QuantumAgent？

`QuantumAgent` 是 QSL 的核心 AI 组件，它能够：
- **理解自然语言任务描述**：用中文或英文描述你想解决的问题
- **自动设计量子算法**：选择合适的算法（Grover、Shor、QAOA、VQE 等）
- **实现并执行量子电路**：自动编写代码、运行模拟
- **验证结果正确性**：内置多种问题的自动验证器
- **输出结构化报告**：通过 `.to_markdown()` 生成人类可读的结果文档

### 无 Key 也能用！

QSL 的 AI 模块有一个独特优势：**即使没有配置任何 LLM API Key，也能使用规则引擎**进行中文参数抽取和算法匹配。配置 LLM 后，你将获得更强大的自然语言理解和代码生成能力。

---

## 📦 前置要求

AI 模块需要额外的依赖：

```bash
pip install "qsl-quantum[ai]"
```

核心功能（规则引擎 + 内置演示）仅需 numpy 即可运行。如果需要 LLM 增强功能，请配置对应的 API Key（见下文）。

---

## 🚀 快速开始：你的第一个 AI 量子实验

让我们从最简单的例子开始——用自然语言让 AI 帮我们分解整数。

```python
from qsl import QuantumAgent

print("=" * 60)
print("  QSL AI 量子科学家演示：整数分解")
print("=" * 60)

# 1. 创建 AI 量子科学家代理
agent = QuantumAgent(
    task_description=None,  # 稍后指定任务
    max_iterations=3,       # 最多尝试迭代次数
    verbose=True            # 显示详细思考过程
)

# 2. 用自然语言描述任务并运行
result = agent.run(task="请将 15 分解为两个素数的乘积")

# 3. 查看结构化结果（Markdown 格式）
print("\n" + "=" * 60)
print("  AI 执行结果报告")
print("=" * 60)
print(result.to_markdown())
```

### 预期输出（无 API Key 时使用规则引擎）

```
============================================================
  QSL AI 量子科学家演示：整数分解
============================================================
[QuantumAgent] 未检测到 LLM API Key，使用规则引擎模式
[QuantumAgent] 正在分析任务...
[QuantumAgent] 检测到任务类型：整数分解 (Shor 算法)
[QuantumAgent] 提取参数：N=15
[QuantumAgent] 选择算法：Shor 整数分解算法
[QuantumAgent] 正在构建量子电路...
[QuantumAgent] 执行模拟...
[QuantumAgent] 验证结果...

============================================================
  AI 执行结果报告
============================================================
## 任务：整数分解

- **目标数**: 15
- **分解结果**: 3 × 5 = 15
- **使用算法**: Shor 量子分解算法
- **验证状态**: ✅ 通过
- **量子比特数**: 8
- **执行 shots**: 1024
```

---

## 🔧 配置 LLM（大语言模型）

如果想要更强大的自然语言理解能力，可以配置以下任意一个 LLM 提供商的 API Key：

### 支持的 LLM 提供商

| 提供商 | 环境变量 | 获取地址 |
|--------|----------|----------|
| DeepSeek | `DEEPSEEK_API_KEY` | https://platform.deepseek.com |
| Moonshot (Kimi) | `MOONSHOT_API_KEY` | https://platform.moonshot.cn |
| OpenAI | `OPENAI_API_KEY` | https://platform.openai.com |
| DashScope (通义千问) | `DASHSCOPE_API_KEY` | https://dashscope.aliyuncs.com |

### 配置方法

**方法一：设置环境变量（推荐）**

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY = "your-api-key-here"

# Windows CMD
set DEEPSEEK_API_KEY=your-api-key-here

# Linux/macOS
export DEEPSEEK_API_KEY="your-api-key-here"
```

**方法二：在 Python 代码中配置**

```python
from qsl import QuantumAgent, create_provider, set_default_provider

# 创建指定提供商
provider = create_provider("deepseek", api_key="your-api-key-here")
set_default_provider(provider)

# 然后正常创建 agent
agent = QuantumAgent(verbose=True)
```

### LLM 增强模式示例

配置 LLM 后，你可以用更自然的语言描述复杂任务：

```python
from qsl import QuantumAgent

agent = QuantumAgent(max_iterations=5, verbose=True)

# 更模糊、更自然的任务描述
result = agent.run("""
    我有一个图，有 4 个顶点，边是 (0,1), (1,2), (2,3), (3,0), (0,2)。
    请帮我找到一个顶点划分，使得跨边的数量最多——这就是 MaxCut 问题。
    用 QAOA 算法来解决它。
""")

print(result.to_markdown())
```

---

## 🎬 内置演示：10 个经典量子算法

QSL 内置了 **10 个精心设计的中文演示**，无需任何 API Key 即可运行，是学习量子算法的绝佳素材。

### 列出所有演示

```python
from qsl import list_demos

demos = list_demos()
print("=" * 70)
print(f"  共有 {len(demos)} 个内置演示")
print("=" * 70)
for demo in demos:
    print(f"\n[{demo['id']}] {demo['name']}")
    print(f"    关键字: {demo['key']}")
    print(f"    描述: {demo['desc']}")
```

### 演示列表

| ID | 名称 | key | 算法 |
|----|------|-----|------|
| 1 | 整数分解 | `factor` | Shor 算法 |
| 2 | 3-SAT 问题 | `sat` | Grover 搜索 |
| 3 | 数独求解 | `sudoku` | Grover 搜索 |
| 4 | MaxCut 图分割 | `maxcut` | QAOA |
| 5 | 旅行商问题 (TSP) | `tsp` | QAOA |
| 6 | 图着色 | `coloring` | Grover/QAOA |
| 7 | Grover 搜索 | `grover` | Grover 算法 |
| 8 | GHZ 纠缠态 | `ghz` | 量子纠缠 |
| 9 | 量子随机数生成 (QRNG) | `qrng` | 量子测量 |
| 10 | BB84 量子密钥分发 | `bb84` | 量子密码 |

### 运行演示

支持**数字索引**和**名称 key**两种方式调用：

```python
from qsl import run_demo

print("=" * 60)
print("  方式 1：使用数字索引运行演示")
print("=" * 60)
result1 = run_demo(1)  # 运行第 1 个演示（整数分解）
print(result1.to_markdown())

print("\n" + "=" * 60)
print("  方式 2：使用 key 名称运行演示")
print("=" * 60)
result2 = run_demo("grover")  # 运行 Grover 搜索
print(result2.to_markdown())

print("\n" + "=" * 60)
print("  演示结果可当作 dict 使用")
print("=" * 60)
print(f"算法名称: {result1['algorithm']}")
print(f"执行成功: {result1['success']}")
print(f"量子比特: {result1['n_qubits']}")
```

### 逐个运行所有演示

```python
from qsl import list_demos, run_demo

demos = list_demos()
for demo in demos:
    print(f"\n{'='*60}")
    print(f"  正在运行演示 {demo['id']}: {demo['name']}")
    print('='*60)
    result = run_demo(demo['id'])
    print(result.to_markdown())
```

---

## ✅ 自动验证器

QSL AI 内置了专用的结果验证器，确保量子算法输出的正确性：

| 验证器函数 | 适用问题 | 功能 |
|-----------|----------|------|
| `verify_shor` | 整数分解 | 验证因子是否为素数、乘积是否等于 N |
| `verify_sat` | 布尔 SAT | 验证解是否满足所有子句 |
| `verify_grover` | Grover 搜索 | 验证搜索结果是否为目标项 |
| `verify_qaoa` | QAOA 优化 | 验证解是否达到最优/近似最优 |

### 使用示例

```python
from qsl import run_demo
from qsl.ai.verifiers import verify_shor, verify_sat

# 运行 Shor 分解演示
result = run_demo("factor")

# 手动验证
factors = result.get("factors", [])
N = result.get("N", 15)
is_valid = verify_shor(N, factors)
print(f"验证结果: {'✅ 正确' if is_valid else '❌ 错误'}")

# SAT 问题验证
sat_result = run_demo("sat")
solution = sat_result.get("solution")
clauses = sat_result.get("clauses")
sat_valid = verify_sat(clauses, solution)
print(f"SAT 验证: {'✅ 满足' if sat_valid else '❌ 不满足'}")
```

---

## 📝 完整示例：3-SAT 问题求解

让我们用 AI 解决一个经典的 3-SAT 问题，体验完整流程：

```python
from qsl import QuantumAgent
import matplotlib.pyplot as plt

print("=" * 60)
print("  AI 求解 3-SAT 问题完整示例")
print("=" * 60)

# 问题：3 个变量，4 个子句
# (x0 ∨ x1 ∨ x2) ∧ (~x0 ∨ ~x1) ∧ (x1 ∨ ~x2) ∧ (~x0 ∨ x2)
task = """
求解一个 3-SAT 问题，有 3 个布尔变量 x0, x1, x2：
子句 1: x0 或 x1 或 x2
子句 2: 非 x0 或 非 x1
子句 3: x1 或 非 x2
子句 4: 非 x0 或 x2
请用量子算法找到满足所有子句的解。
"""

# 创建 agent 并运行
agent = QuantumAgent(max_iterations=3, verbose=True)
result = agent.run(task=task)

# 输出 Markdown 报告
print("\n" + result.to_markdown())

# 如果安装了可视化，可以查看电路和结果
try:
    from qsl import plot_histogram
    qc = result['circuit']
    counts = result['counts']
    
    # 绘制电路
    fig1, ax1 = qc.draw(output="mpl", style="iqp")
    fig1.suptitle("Grover SAT 求解电路", fontsize=14)
    
    # 绘制结果直方图，高亮正确解
    solution = result.get('solution_binary', '')
    fig2, ax2 = plt.subplots()
    plot_histogram(counts, title="测量结果分布", highlight=solution, ax=ax2)
    
    plt.show()
except ImportError:
    print("\n[提示] 安装 viz 依赖可查看可视化: pip install qsl-quantum[viz]")
```

---

## 💡 高级技巧

### 1. 调整迭代次数

`max_iterations` 控制 AI 自我修正的次数。如果问题复杂，可以适当调大：

```python
agent = QuantumAgent(max_iterations=5, verbose=True)
```

### 2. AgentResult 对象结构

`agent.run()` 返回的 `AgentResult` 包含丰富信息：

```python
result = agent.run("分解 21")

# 常用属性
print(result.success)        # 是否成功
print(result.algorithm)      # 使用的算法名称
print(result.circuit)        # 构建的量子电路
print(result.counts)         # 测量统计
print(result.solution)       # 找到的解
print(result.n_qubits)       # 使用的量子比特数
print(result.iterations)     # 实际迭代次数
print(result.error_message)  # 错误信息（如果失败）

# 转换为 Markdown
md = result.to_markdown()

# 当作 dict 使用
for key, value in result.items():
    print(f"{key}: {value}")
```

### 3. 保存结果报告

```python
result = run_demo(1)
md_content = result.to_markdown()

with open("ai_result.md", "w", encoding="utf-8") as f:
    f.write("# AI 量子算法执行报告\n\n")
    f.write(md_content)

print("报告已保存到 ai_result.md")
```

---

## ❓ 常见问题解答

### Q1: 没有 API Key 真的可以用吗？

**A**：完全可以！QSL 内置了中文规则引擎，可以识别任务类型、提取参数，并运行对应的算法。配置 LLM 后，AI 的泛化能力会更强，可以理解更自由的描述方式。

### Q2: 支持哪些问题类型？

**A**：目前内置支持：
- 整数分解（Shor）
- 布尔 SAT / k-SAT（Grover）
- 数独求解（Grover）
- MaxCut、TSP、图着色（QAOA）
- Grover 无序搜索
- GHZ 态制备、QRNG、BB84 等量子协议演示

配置 LLM 后，AI 可以尝试更多问题类型。

### Q3: 为什么有时候 AI 会在多次迭代后才找到正确解？

**A**：这是正常的！和人类科学家一样，AI 也可能：
1. 第一次选错了算法
2. 参数提取有误
3. 电路构造有 bug
4. 迭代次数不够

`max_iterations` 就是给 AI 自我修正的机会。通常 3-5 次迭代足够解决大部分问题。

### Q4: 可以自定义新的验证器吗？

**A**：当然可以！验证器只是一个返回布尔值的函数：

```python
def my_verifier(problem, solution):
    """自定义验证逻辑"""
    # 你的验证代码
    return is_correct

# 在 agent 中使用
agent = QuantumAgent(custom_verifiers={"my_problem": my_verifier})
```

---

## 🎯 下一步

恭喜你掌握了 AI 量子科学家的使用方法！接下来你可以：

1. **运行所有内置演示**：`python -m qsl --list-demos` 然后逐个体验
2. **配置 LLM**：尝试用更自然的语言描述问题
3. **结合可视化**：用 `plot_histogram` 高亮 AI 找到的解
4. **查看 CLI 教程**：`09_cli_guide.md` 学习如何用命令行直接调用 AI 演示
5. **阅读源码**：查看 `qsl/ai/` 目录了解 AI 模块的实现

---

**让 AI 帮你做量子科研吧！** 🧪⚛️

无需记忆复杂的量子门和算法细节——用自然语言描述你的问题，剩下的交给 `QuantumAgent`。
