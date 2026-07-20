[← 返回文档中心](../index.md)

# QSL 命令行工具（CLI）完全指南

QSL 提供了功能强大的命令行界面，让你无需编写 Python 脚本就能快速体验量子计算、运行 AI 演示、求解问题、执行 DSL 文件。本教程将详细介绍所有 CLI 命令的使用方法。

---

## 🚀 快速开始

### 两种调用方式

安装 QSL 后，你可以通过两种方式调用命令行工具：

**方式一：使用 `python -m qsl`（推荐，无需配置 PATH）**

```bash
python -m qsl [命令] [参数]
```

**方式二：使用 `qsl` 命令（安装后可用）**

```bash
qsl [命令] [参数]
```

> 💡 如果 `qsl` 命令不可用，请确保 Python 的 Scripts 目录已添加到系统 PATH，或者继续使用 `python -m qsl` 方式。

---

## 📋 基础命令

### 1. 查看版本号

使用 `--version` 或 `-V` 参数查看当前安装的 QSL 版本：

```bash
python -m qsl --version
```

或简写形式：

```bash
python -m qsl -V
```

**预期输出**：
```
QSL version 0.6.3
```

### 2. 查看帮助信息

使用 `--help` 或 `-h` 参数查看完整的命令帮助：

```bash
python -m qsl --help
```

或简写形式：

```bash
python -m qsl -h
```

这会列出所有可用命令和参数说明。

### 3. 交互式启动

直接运行 `python -m qsl` 不带任何参数，将进入**交互式模式**：

```bash
python -m qsl
```

在交互式模式中，你可以：
- 输入数字选择演示
- 实时查看执行结果
- 反复运行不同实验

**交互式菜单示例**：
```
============================================================
  QSL 量子计算框架 - 交互式演示
============================================================
  [1] Grover 搜索演示
  [2] AI 量子科学家演示
  [3] 运行 .qsl DSL 文件
  [0] 退出
============================================================
请选择: 
```

---

## 🎬 Grover 搜索演示

`--demo` 参数用于快速运行经典的 Grover 搜索算法演示。

### 交互式选择演示

不带参数运行 `--demo`，会进入演示选择菜单：

```bash
python -m qsl --demo
```

### 直接运行指定演示

传入数字参数 `N`（0-4）可以直接运行对应演示：

```bash
# 运行第 1 个演示：SAT 求解（3 变量）
python -m qsl --demo 1

# 运行第 2 个演示：图着色（3 顶点 2 色）
python -m qsl --demo 2

# 运行第 3 个演示：恰好一个 1（n=3）
python -m qsl --demo 3

# 运行第 4 个演示：大空间搜索（n=6）
python -m qsl --demo 4

# 运行所有演示！
python -m qsl --demo 0
```

### 可用的 Grover 演示列表

| N | 演示名称 | 问题描述 | 量子比特数 |
|---|---------|----------|-----------|
| 1 | SAT 求解 | 3 变量布尔可满足性问题 | 3 |
| 2 | 图着色 | 3 顶点 2 色着色问题 | 4 |
| 3 | 恰好一个 1 | n=3 比特中恰好一个 1 的搜索 | 3 |
| 4 | 大空间搜索 | 6 比特空间中搜索单个目标 | 6 |
| 0 | 运行全部 | 依次运行以上所有演示 | - |

### 运行示例（python -m qsl --demo 1）

```
============================================================
  Grover 演示 1: 3-SAT 问题求解
============================================================

问题描述：
  (x0 ∨ x1) ∧ (~x0 ∨ x2) ∧ (x1 ∨ ~x2)
  共 3 个变量，3 个子句

构建量子电路...
执行模拟（shots=1024）...

测量结果：
  101: 512 次 ████████████████████████████████ 50.0%
  110: 487 次 ███████████████████████████████  47.6%
  其他:  25 次 ██                             2.4%

找到的解：
  x0=1, x1=0, x2=1  →  二进制 101  (5)
  x0=1, x1=1, x2=0  →  二进制 110  (6)

验证：所有子句均满足 ✅
============================================================
```

---

## 🤖 AI 量子科学家演示

QSL 内置了 **10 个中文 AI 演示**，无需配置 API Key 即可运行，涵盖经典量子算法和量子协议。

### 列出所有 AI 演示

使用 `--list-demos` 查看所有可用的中文演示：

```bash
python -m qsl --list-demos
```

**预期输出**：
```
============================================================
  QSL 中文 AI 演示（共 10 个）
============================================================
  [1]  整数分解      - 使用 Shor 算法分解大整数
  [2]  3-SAT 问题    - Grover 算法求解布尔可满足性
  [3]  数独求解      - Grover 算法求解 4×4 数独
  [4]  MaxCut 图分割 - QAOA 求解最大割问题
  [5]  TSP 旅行商    - QAOA 求解旅行商问题
  [6]  图着色        - 量子算法求解图着色问题
  [7]  Grover 搜索   - 无序数据库搜索演示
  [8]  GHZ 纠缠态    - 制备多粒子 GHZ 纠缠态
  [9]  QRNG 随机数   - 量子随机数生成器
  [10] BB84 密钥分发 - BB84 量子密钥分发协议
============================================================
  使用方法: python -m qsl --ai-demo N  (N=1-10)
============================================================
```

### 运行指定 AI 演示

使用 `--ai-demo N` 运行第 N 个演示（N 从 1 到 10）：

```bash
# 运行第 1 个演示：整数分解（Shor 算法）
python -m qsl --ai-demo 1

# 运行第 7 个演示：Grover 搜索
python -m qsl --ai-demo 7

# 运行第 10 个演示：BB84 量子密钥分发
python -m qsl --ai-demo 10
```

### AI 演示详解

让我们逐个了解这 10 个演示：

#### 1. 整数分解（Shor 算法）
```bash
python -m qsl --ai-demo 1
```
演示 Shor 量子算法分解 15 = 3 × 5，展示量子计算在大数分解上的指数加速潜力。

#### 2. 3-SAT 问题
```bash
python -m qsl --ai-demo 2
```
用 Grover 算法求解布尔可满足性问题，展示量子搜索在 NP 问题上的平方加速。

#### 3. 数独求解
```bash
python -m qsl --ai-demo 3
```
用量子算法求解 4×4 数独谜题，展示量子搜索在约束满足问题上的应用。

#### 4. MaxCut 图分割
```bash
python -m qsl --ai-demo 4
```
使用 QAOA（量子近似优化算法）求解最大割问题，展示量子优化在组合问题上的应用。

#### 5. TSP 旅行商问题
```bash
python -m qsl --ai-demo 5
```
用 QAOA 求解小规模旅行商问题，展示量子计算在路径优化中的潜力。

#### 6. 图着色
```bash
python -m qsl --ai-demo 6
```
用量子算法求解图着色问题，相邻顶点颜色不同。

#### 7. Grover 搜索
```bash
python -m qsl --ai-demo 7
```
Grover 算法标准演示，在 N 个元素中搜索目标项，仅需 O(√N) 次查询。

#### 8. GHZ 纠缠态
```bash
python -m qsl --ai-demo 8
```
制备 3 粒子 GHZ 纠缠态 |000⟩ + |111⟩，展示量子纠缠的非经典特性。

#### 9. QRNG 量子随机数
```bash
python -m qsl --ai-demo 9
```
基于量子测量的真随机数生成器，区别于伪随机数。

#### 10. BB84 密钥分发
```bash
python -m qsl --ai-demo 10
```
演示 BB84 量子密钥分发协议，展示量子力学如何保证通信安全。

---

## 🔧 命令行 SAT 求解

使用 `--solve` 参数可以直接在命令行求解布尔 SAT 问题，无需编写 Python 代码。

### 命令格式

```bash
python -m qsl --solve N <子句1> [子句2] [子句3] ...
```

其中：
- `N`：布尔变量的数量
- 后续每个参数是一个子句，用逻辑表达式表示

### 子句语法

子句中可以使用以下符号：

| 符号 | 含义 | 示例 |
|------|------|------|
| `x0`, `x1`, `x2`... | 布尔变量 | `x0` 表示变量 0 |
| `~`, `!`, `¬` | 逻辑非（NOT） | `~x1` 表示 x1 取反 |
| `\|`, `∨` | 逻辑或（OR） | `x0\|~x1` 表示 x0 或非 x1 |
| `&`, `∧` | 逻辑与（AND） | `x0&x1` 表示 x0 与 x1 |

> 💡 注意：在某些 shell 中，`|` 和 `&` 是特殊字符，需要用引号包裹子句。

### 基础示例

让我们求解一个经典的 3-SAT 问题：

```bash
python -m qsl --solve 3 "x0|~x1" "x1|x2" "~x0|~x2"
```

这个问题有 3 个变量，3 个子句：
1. x0 或 非x1
2. x1 或 x2
3. 非x0 或 非x2

**预期输出**：
```
============================================================
  QSL 命令行 SAT 求解器
============================================================

变量数量: 3
子句数量: 3

子句列表：
  1. x0 ∨ ¬x1
  2. x1 ∨ x2
  3. ¬x0 ∨ ¬x2

正在构建 Grover 电路...
量子比特数: 3
执行模拟（shots=1024）...

测量结果：
  001: 498 次 ████████████████████████████████ 48.6%
  011: 506 次 ████████████████████████████████ 49.4%
  其他:  20 次 ██                             2.0%

找到的解：
  解 1: x0=0, x1=0, x2=1  →  二进制 001  ✓
  解 2: x0=0, x1=1, x2=1  →  二进制 011  ✓

验证：所有解均满足全部子句 ✅
============================================================
```

### 更多 SAT 示例

**示例 1：简单的 2-SAT**
```bash
python -m qsl --solve 2 "x0|x1" "~x0|~x1"
```

**示例 2：4 变量问题**
```bash
python -m qsl --solve 4 "x0|x1|x2" "~x1|~x2|x3" "x0|~x3" "~x0|x2"
```

**示例 3：蕴含式（A → B 等价于 ~A | B）**
```bash
# x0 → x1, x1 → x2, x2 → x0, 即三个变量等价
python -m qsl --solve 3 "~x0|x1" "~x1|x2" "~x2|x0"
```

---

## 📄 运行 .qsl DSL 文件

QSL 支持自定义的量子电路 DSL（领域特定语言），你可以将电路定义写在 `.qsl` 文件中，然后用命令行直接执行。

### 命令格式

```bash
python -m qsl --file <文件路径>
```

### .qsl 文件语法

`.qsl` 文件是一种简单的文本格式，每行一条指令：

| 指令 | 含义 | 示例 |
|------|------|------|
| `qubit N` | 声明 N 个量子比特 | `qubit 3` |
| `h q` | Hadamard 门 | `h 0` |
| `x q` | Pauli-X 门 | `x 1` |
| `y q` | Pauli-Y 门 | `y 2` |
| `z q` | Pauli-Z 门 | `z 0` |
| `cx c t` | CNOT 门 | `cx 0 1` |
| `cz c t` | 受控 Z 门 | `cz 1 2` |
| `swap a b` | SWAP 门 | `swap 0 2` |
| `rx theta q` | 绕 X 轴旋转 | `rx 1.57 0` |
| `ry theta q` | 绕 Y 轴旋转 | `ry 3.14 1` |
| `rz theta q` | 绕 Z 轴旋转 | `rz 0.785 2` |
| `measure` | 测量所有比特 | `measure` |
| `// 注释` | 注释行 | `// 这是注释` |
| `# 注释` | 注释行（同上） | `# 这也是注释` |

### 创建你的第一个 .qsl 文件

创建一个名为 `bell_state.qsl` 的文件，内容如下：

```qsl
// Bell 态制备电路
// 创建 EPR 纠缠对：|Φ⁺⟩ = (|00⟩ + |11⟩)/√2

qubit 2

// 在 qubit 0 上施加 Hadamard 门，创建叠加态
h 0

// CNOT 门：控制位 0，目标位 1，产生纠缠
cx 0 1

// 添加测量
measure

// 执行参数：shots=4096
// shots 4096
```

### 运行 .qsl 文件

```bash
python -m qsl --file bell_state.qsl
```

**预期输出**：
```
============================================================
  QSL DSL 文件执行器
============================================================

文件: bell_state.qsl
量子比特: 2

解析电路...
  [1] h 0
  [2] cx 0 1
  [3] measure

电路结构：
     ┌───┐
q_0: ┤ H ├──■──
     └───┘┌─┴─┐
q_1: ─────┤ X ├
          └───┘
       ┌─┐┌─┐
q_0: ┤M├┤M├
     └╥┘└╥┘
q_1: ─╫──╫─
      ║  ║
c: 2/═╩══╩═
      0  1

执行模拟（shots=1024）...

测量结果统计：
  00: 512 次 ████████████████████████████████ 50.0%
  11: 498 次 ████████████████████████████████ 48.6%
  01:  8 次  █                             0.8%
  10:  6 次  █                             0.6%

量子态（Dirac 记号）：
0.7071|00⟩ + 0.7071|11⟩
============================================================
```

### 更多 .qsl 示例

**示例 1：GHZ 纠缠态（ghz.qsl）**
```qsl
// GHZ 态：(|000⟩ + |111⟩)/√2
qubit 3
h 0
cx 0 1
cx 0 2
measure
```

运行：
```bash
python -m qsl --file ghz.qsl
```

**示例 2：量子随机数生成器（qrng.qsl）**
```qsl
// QRNG：生成 3 比特真随机数
qubit 3
h 0
h 1
h 2
measure
// shots 8192
```

运行：
```bash
python -m qsl --file qrng.qsl
```

**示例 3：Grover 搜索（grover101.qsl）**
```qsl
// Grover 搜索：在 3 比特空间中搜索 |101⟩
qubit 3

// 初始化：均匀叠加
h 0
h 1
h 2

// Oracle：标记 |101⟩
x 1
h 2
ccx 0 1 2
h 2
x 1

// Diffusion 算子
h 0
h 1
h 2
x 0
x 1
x 2
h 2
ccx 0 1 2
h 2
x 0
x 1
x 2
h 0
h 1
h 2

measure
// shots 4096
```

运行：
```bash
python -m qsl --file grover101.qsl
```

---

## ⚡ 可执行命令：qsl

通过 pip 安装 QSL 后，系统会自动安装一个 `qsl` 可执行命令。如果你的 Python Scripts 目录在 PATH 中，可以直接使用：

```bash
# 查看版本
qsl --version

# 查看帮助
qsl --help

# 交互式启动
qsl

# Grover 演示
qsl --demo 1

# 列出 AI 演示
qsl --list-demos

# 运行 AI 演示
qsl --ai-demo 8

# SAT 求解
qsl --solve 3 "x0|~x1" "x1|x2" "~x0|~x2"

# 运行 DSL 文件
qsl --file my_circuit.qsl
```

### Windows 注意事项

如果 `qsl` 命令在 Windows 上不可用，可以：

**方法一：继续使用 python -m**
```powershell
python -m qsl --version
```

**方法二：找到 Scripts 目录并添加到 PATH**
```powershell
# 查看 Scripts 位置
python -c "import site; print(site.getuserbase())"
# 通常是 C:\Users\<用户名>\AppData\Roaming\Python\Python3x\Scripts
```

**方法三：使用完整路径**
```powershell
# 先找到 qsl.exe 的位置
python -c "import sysconfig; print(sysconfig.get_path('scripts'))"
# 然后用完整路径
& "C:\Users\...\Scripts\qsl.exe" --version
```

---

## 📖 命令速查表

| 命令 | 简写 | 功能 |
|------|------|------|
| `python -m qsl` | - | 启动交互式模式 |
| `python -m qsl --version` | `-V` | 查看版本号 |
| `python -m qsl --help` | `-h` | 查看帮助信息 |
| `python -m qsl --demo` | - | 交互式选择 Grover 演示 |
| `python -m qsl --demo N` | - | 运行第 N 个 Grover 演示（0-4） |
| `python -m qsl --list-demos` | - | 列出所有 10 个中文 AI 演示 |
| `python -m qsl --ai-demo N` | - | 运行第 N 个 AI 演示（1-10） |
| `python -m qsl --solve N <子句...>` | - | 命令行 SAT 求解 |
| `python -m qsl --file <path>` | - | 运行 .qsl DSL 文件 |

---

## 💡 实用技巧

### 1. 组合使用：管道与重定向

将 CLI 输出保存到文件：

```bash
# 将 AI 演示结果保存到文件
python -m qsl --ai-demo 1 > shor_result.txt

# 追加到已有文件
python -m qsl --ai-demo 7 >> all_demos.txt
```

### 2. 快速验证安装

安装后快速验证一切正常：

```bash
python -m qsl --version && python -m qsl --demo 1
```

### 3. 在脚本中调用

你也可以在 shell 脚本或批处理文件中调用 QSL：

**Windows batch (run_all_demos.bat)**：
```batch
@echo off
echo Running QSL AI Demos...
for /L %%i in (1,1,10) do (
    echo ========================
    echo Running Demo %%i
    echo ========================
    python -m qsl --ai-demo %%i
)
echo All demos completed!
```

**Bash (run_all_demos.sh)**：
```bash
#!/bin/bash
echo "Running QSL AI Demos..."
for i in {1..10}; do
    echo "========================"
    echo "Running Demo $i"
    echo "========================"
    python -m qsl --ai-demo $i
done
echo "All demos completed!"
```

### 4. 环境变量配置

可以通过环境变量配置 QSL CLI 行为：

```bash
# Windows PowerShell
$env:QSL_SILENT_BACKEND_CHECK = "1"        # 静默后端检查警告
$env:DEEPSEEK_API_KEY = "your-key-here"    # 配置 LLM API Key
$env:QSL_DEFAULT_SHOTS = "8192"            # 默认测量次数

# Linux/macOS
export QSL_SILENT_BACKEND_CHECK=1
export DEEPSEEK_API_KEY="your-key-here"
export QSL_DEFAULT_SHOTS=8192
```

---

## ❓ 常见问题解答

### Q1: `python -m qsl` 报错 "No module named qsl"？

**A**：这说明 QSL 没有安装，或者安装在不同的 Python 环境中。请确认：
```bash
# 检查是否安装
pip list | findstr qsl  # Windows
pip list | grep qsl     # Linux/macOS

# 如果未安装，重新安装
pip install qsl-quantum
```

### Q2: `--ai-demo` 报错缺少依赖？

**A**：AI 演示需要 AI 依赖：
```bash
pip install "qsl-quantum[ai]"
```

### Q3: 中文在终端显示乱码怎么办？

**A**：
- Windows PowerShell：执行 `chcp 65001` 切换到 UTF-8
- Windows CMD：执行 `chcp 65001`
- Linux/macOS：确保终端编码为 UTF-8
- 或者设置环境变量 `PYTHONIOENCODING=utf-8`

### Q4: 如何退出交互式模式？

**A**：
- 选择菜单中的 `0`（退出）
- 或者按 `Ctrl+C` / `Ctrl+D`

### Q5: 可以自定义 .qsl 文件的 shots 数量吗？

**A**：可以！在 .qsl 文件中添加：
```qsl
// 设置执行次数
shots 8192
```
或者在命令行中无法直接指定，但你可以在 DSL 文件中配置。

### Q6: --solve 支持多少个变量？

**A**：本地模拟器下建议不超过 20 个变量（但实际上 Grover 需要 2^n 量子态，n=20 需要约 16GB 内存）。推荐用于 3-12 个变量的问题。

---

## 🎯 下一步

恭喜你掌握了 QSL 的命令行工具！接下来你可以：

1. **体验 AI 科学家**：阅读 `07_ai_scientist.md` 学习 Python API 方式调用 AI
2. **可视化结果**：阅读 `08_visualization.md` 学习绘制漂亮的电路图和直方图
3. **编写 DSL 文件**：创建自己的 `.qsl` 文件，描述任意量子电路
4. **探索 Python API**：回到 `01_getting_started.md` 开始用 Python 编写更复杂的量子程序
5. **结合 LLM**：配置 API Key 后，AI 模块将更加强大

---

**命令行是快速体验量子计算的最佳入口！** ⌨️⚛️

无需打开 IDE，无需编写脚本——一行命令，立即运行量子算法。从 Grover 搜索到 Shor 分解，从 SAT 求解到 BB84 协议，QSL CLI 让量子计算触手可及。
