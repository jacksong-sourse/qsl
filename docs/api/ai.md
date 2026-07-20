[← 返回文档中心](../index.md)

# AI 模块 API 参考

qsl.ai 模块提供 LLM 驱动的量子计算自动化功能，包括自主量子代理、问题翻译、结果解释、假设检验等。无需 API key 也可使用内置的双语规则路由。

---

## QuantumAgent

**自主量子代理**，解析自然语言任务描述，自动选择算法、编译电路、执行计算并验证结果。

### 构造函数

```python
QuantumAgent(
    task_description: Optional[str] = None,
    max_iterations: int = 3,
    verbose: bool = True
)
```

**参数：**
- `task_description` (Optional[str])：自然语言任务描述（可选，也可在 run() 时传入）
- `max_iterations` (int)：最大决策轮数（验证失败时自动重规划），默认 3
- `verbose` (bool)：是否打印进度信息，默认 True

**支持的任务类型（双语关键词路由）：**
- **Shor 算法**：分解、因数、质因数、因子、解密、factor、factorize、rsa、crack、shor
- **QAOA**：优化、最大割、旅行商、组合、调度、着色、覆盖、maxcut、tsp、portfolio、optimize、schedule、coloring、qaoa
- **VQE**：基态、能量、分子、化学、哈密顿、本征值、ground state、energy、molecule、chemistry、hamiltonian、vqe
- **Grover**：搜索、查找、数据库、布尔、数独、可满足、sudoku、sat、3-sat、search、find、database、grover

### 方法

#### run()

执行代理的决策循环：解析任务 → 选择算法 → 提取参数 → 执行 → 验证 → 解释。

```python
run(task: Optional[str] = None) -> AgentResult
```

**参数：**
- `task` (Optional[str])：任务描述；若提供则覆盖构造函数中的 task_description

**返回值：**
- `AgentResult`：包含执行结果、验证状态、决策链等

**异常：**
- `ValueError`：未提供任务描述时抛出

**示例：**
```python
from qsl.ai import QuantumAgent

# 方式1：构造时传入任务
agent = QuantumAgent("分解整数 15", verbose=True)
result = agent.run()

# 方式2：run() 时传入任务
agent2 = QuantumAgent(verbose=False)
result2 = agent2.run("4 节点环图的最大割")

print(result.success)
print(result.algorithm_used)
print(result.result_summary)
```

---

#### suggest_clarification()

当任务缺少必要参数时，返回中文追问问题。

```python
suggest_clarification() -> Optional[str]
```

**返回值：**
- `Optional[str]`：中文追问字符串，参数齐全时返回 None

**示例：**
```python
agent = QuantumAgent("分解一个数")
hint = agent.suggest_clarification()
print(hint)  # "请问要分解的整数是多少？（例如：分解 15）"
```

---

## AgentResult

代理执行结果数据类。

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `task` | str | 原始任务描述 |
| `success` | bool | 是否成功完成 |
| `algorithm_used` | str | 使用的算法名（shor/qaoa/vqe/grover） |
| `backend_used` | str | 使用的后端（simulator 等） |
| `iterations` | int | 实际执行轮数 |
| `result_summary` | str | 结果摘要（LLM 解释或默认文本） |
| `data` | dict | 原始结果数据（factors/bitstring/energy 等） |
| `error` | Optional[str] | 错误信息（失败时） |
| `verification` | Optional[VerificationResult] | 自动验证结果 |
| `decision_chain` | list | 决策链记录（每轮的 action 和 detail） |
| `verified` | bool | 属性，结果是否通过自动验证 |

### 方法

#### to_markdown()

直接返回 Markdown 格式的中文报告。

```python
to_markdown(algorithm_reason: str = "") -> str
```

**参数：**
- `algorithm_reason` (str)：算法选择理由说明

**返回值：**
- `str`：Markdown 格式报告

#### to_report()

生成结构化的 AgentReport 对象。

```python
to_report(algorithm_reason: str = "") -> AgentReport
```

**示例：**
```python
result = agent.run("分解 15")
if result.success:
    print(result.to_markdown())
    print(f"因子: {result.data['factors']}")
    print(f"验证通过: {result.verified}")
```

---

## Demos（内置演示）

### list_demos()

返回所有内置演示的列表（id 从 1 开始）。

```python
list_demos() -> List[dict]
```

**返回值：**
- `List[dict]`：每个元素包含 `id`、`key`、`name`、`desc` 字段

### run_demo()

按编号或名称运行演示，返回 DemoResult。

```python
run_demo(name, verbose: bool = False) -> DemoResult
```

**参数：**
- `name`：演示编号（1-based int/str）或字符串 key
- `verbose` (bool)：是否打印详细信息，默认 False

**返回值：**
- `DemoResult`：字典子类，支持 `.to_markdown()` 方法

**内置演示列表：**

| id | key | 名称 | 说明 |
|----|-----|------|------|
| 1 | factor | 整数分解 | 分解整数 15 (Shor 算法 + 回乘/素性校验) |
| 2 | sat | 3-SAT 求解 | 3-SAT 布尔可满足性求解 (Grover/BBHT + 子句代回校验) |
| 3 | sudoku | 迷你数独 | 2x2 迷你数独 (SAT 编码 + Grover 搜索) |
| 4 | maxcut | 最大割 | 4 节点环图最大割 (QAOA + 全枚举基线校验) |
| 5 | tsp | 旅行商问题 | 3 城市 TSP 最短回路 (QAOA/QUBO + 枚举最优校验) |
| 6 | graph_coloring | 图着色 | 三角形图 3 着色 (SAT/CNF 编码 + Grover 搜索) |
| 7 | grover | 数据库搜索 | 4 比特数据库搜索 (Grover 振幅放大 + top-k 命中率校验) |
| 8 | ghz | GHZ 纠缠态 | GHZ 纠缠态制备 (测量只出现 000/111) |
| 9 | qrng | 量子随机数 | 8 比特量子随机数生成器 (均匀性卡方粗检) |
| 10 | bb84 | BB84 量子密钥分发 | BB84 量子密钥分发 (无窃听 100% 一致, 有窃听 ~75% 被检出) |

**示例：**
```python
from qsl.ai.demos import list_demos, run_demo

# 列出所有演示
for demo in list_demos():
    print(f"{demo['id']}. {demo['name']}: {demo['desc']}")

# 按编号运行
result = run_demo(1, verbose=True)
print(result.report_markdown)

# 按名称运行
result2 = run_demo("bb84")
print(result2.to_markdown())
```

---

## LLM Provider（LLM 提供商）

### create_provider()

自动检测并创建 LLM 提供商实例（根据环境变量）。

```python
create_provider(model=None, api_key=None) -> Optional[LLMProvider]
```

**支持的提供商（按优先级自动检测）：**
- **DeepSeek**：`DEEPSEEK_API_KEY` 环境变量，默认模型 `deepseek-chat`
- **Kimi (Moonshot)**：`MOONSHOT_API_KEY`
- **Qwen (DashScope)**：`DASHSCOPE_API_KEY`
- **OpenAI**：`OPENAI_API_KEY`，默认模型 `gpt-4`
- **Ollama**：本地运行，无需 API key

**参数：**
- `model`：指定模型名称（可选）
- `api_key`：指定 API key（可选，优先使用环境变量）

**返回值：**
- `LLMProvider` 实例，无可用 API key 时返回 None

### set_default_provider()

设置全局默认 LLM 提供商。

```python
set_default_provider(provider: LLMProvider)
```

### get_default_provider()

获取全局默认 LLM 提供商（若未设置则自动调用 create_provider）。

```python
get_default_provider() -> Optional[LLMProvider]
```

### LLMConfig

LLM 配置数据类。

```python
LLMConfig(
    provider: str = "auto",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.1
)
```

---

## ProblemTranslator

自然语言到量子问题的翻译器（需要 LLM）。

```python
ProblemTranslator(llm_provider: Optional[LLMProvider] = None)
```

### 方法

#### translate()

将自然语言描述翻译为结构化的量子问题规范。

```python
translate(nl_description: str) -> dict
```

---

## ResultExplainer

量子结果解释器（需要 LLM）。

```python
ResultExplainer(llm_provider: Optional[LLMProvider] = None)
```

### 方法

#### explain()

用自然语言解释量子计算结果。

```python
explain(result: dict, algorithm: str, context: str = "") -> str
```

---

## HypothesisTester

量子假设检验器（需要 LLM）。

```python
HypothesisTester(llm_provider: Optional[LLMProvider] = None)
```

用于自动生成和验证关于量子系统的假设。

---

## DiscoveryPipeline

自动量子发现管道（需要 LLM）。

```python
DiscoveryPipeline(llm_provider: Optional[LLMProvider] = None)
```

端到端的科学发现流程：提出假设 → 设计实验 → 量子模拟 → 分析结果。

---

## 环境变量配置

| 环境变量 | 说明 |
|----------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `MOONSHOT_API_KEY` | Kimi/Moonshot API key |
| `DASHSCOPE_API_KEY` | Qwen/通义千问 API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `QSL_LLM` | 强制指定提供商（deepseek/kimi/qwen/openai/ollama） |

**示例（.env 或命令行）：**
```bash
# 使用 DeepSeek
export DEEPSEEK_API_KEY="your-key-here"

# 使用 OpenAI
export OPENAI_API_KEY="sk-..."

# 强制使用 Ollama（本地）
export QSL_LLM="ollama"
```

---

## 无 LLM 回退模式

当未配置任何 LLM API key 时，QuantumAgent 自动回退到：
1. **双语关键词路由**：通过中文/英文关键词匹配选择算法
2. **正则参数提取**：从任务文本中自动提取 N（Shor）、n_qubits（QAOA/VQE/Grover）等参数
3. **默认参数兜底**：参数缺失时使用合理默认值（如 Shor 默认分解 15）
4. **结果自动验证**：经典交叉验证（回乘校验、全枚举基线等）照常工作

因此即使没有 API key，`python -m qsl --ai-demo` 和核心 AI 功能仍可离线运行。
