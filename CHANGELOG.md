# Changelog

本文件遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 规范，
项目版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.6.1] - 2026-07-19

### Added

- `QuantumCircuit.measure_all(shots, seed)`：便捷方法直接执行电路并返回 counts。
- `QuantumCircuit.reverse_bits()`：返回比特索引反转后的新电路。
- `QuantumCircuit.assign_parameters()` / `assign()`：`bind_parameters()` 的 Qiskit 风格别名。
- `QuantumCircuit.width()` / `num_nonlocal_gates()` / `num_tensor_factors()` 统计 API；`qubits` / `clbits` 属性。
- `QuantumCircuit.get_instructions(name)` / `remove_final_measurements()` / `has_register()`：Qiskit 兼容工具方法。
- `QuantumState.dirac()` / `pretty_print()`：以 Dirac 记号（如 `0.7071|00⟩ + 0.7071|11⟩`）美观打印量子态。
- CLI 新增 `--version` / `--list-demos` / `--ai-demo <N>` 参数；`--demo N` 支持直接运行指定编号 demo。
- GitHub Actions `build.yml` + `lint.yml`：wheel 构建验证与 ruff 代码检查。

### Fixed

- **参数化受控/两比特门参数顺序对齐 Qiskit**：`crx/cry/crz/cp/cu/rxx/ryy/rzz` 现统一为角度参数在前、比特在后（例：`qc.crx(θ, c, t)`），与 `rx(θ, q)` 保持一致，并与 Qiskit API 完全对齐。
- 修复 Grover BBHT 搜索单轮概率漏检问题：增加外层重启机制（最多 8 轮）。
- 修复 CLI `--demo N` 参数被忽略的问题（原实现始终进入交互选择）。
- 删除与 `pyproject.toml` 冲突的过时 `setup.py`（其版本停留在 0.4.1 且 `classifiers` 字段拼错为 `classifier`）。
- 修复密度矩阵 `execute_density()` 对初始态向量的归一化处理。
- QASM 2.0/3.0 导入导出同步新参数顺序。
- QFT 标准电路 `library.qft()` 受控相位门位序修正。

## [0.6.0] - 2026-07-19

### Added

- **QuantumCircuit 电路层**（`qsl.circuit`）：完整的量子电路对象模型，支持参数化门（`Parameter`）、`decompose()` 分解（任意受控门递归分解为 1q/2q 基础门）、`transpile()` 多级优化、`control()` / `power()` / `inverse()` 电路变换。
- **40+ 标准量子门**（`qsl.quantum_gates` 与 `qsl.circuit.library`）：涵盖 Pauli、Hadamard、相位、旋转（rx/ry/rz/rxx/ryy/rzz）、交换（swap/iswap/cswap）、sqrt-X 族（sx/sxdg）、echoed CR（ecr/dcx）、通用 U 门与多受控门（ccx/cu/cp/crx/cry/crz）等。
- **OpenQASM 互通**：OpenQASM 2.0 导入与导出（`from_qasm` / `to_qasm`），以及 OpenQASM 3.0 导出。
- **跨框架转换器**：`to_qiskit()` / `from_qiskit()` / `to_cirq()`，与 Qiskit、Cirq 生态双向互操作。
- **matplotlib 可视化**（`qsl.viz`）：电路图绘制、Bloch 球、态密度城市图（city plot）、Q 球（Q-sphere）与测量结果直方图。
- **密度矩阵噪声模拟**：`NoiseModel`（比特翻转 / 退相位 / 振幅阻尼 / 退极化等通道）与 `execute_density()`，支持含噪声电路的密度矩阵演化。
- **GPU 加速**：cupy 后端开关，状态向量模拟可无缝切换至 GPU。
- **免采样期望值**：`QuantumCircuit.expectation()` 直接解析计算 Pauli 串 / 矩阵可观测量期望值，无需 shots 采样。
- **LLMProvider 多模型抽象**（`qsl.ai.llm_provider`）：统一封装 OpenAI / DeepSeek / Kimi（Moonshot）/ 通义千问 / Ollama 本地模型，凭证走环境变量，支持 `QSL_LLM` / `QSL_LLM_MODEL` 全局切换。
- **规则路由表 + 中文参数抽取 + 追问**：自然语言翻译器在无 LLM 时回退到规则路由，支持中文意图识别、参数抽取与缺参主动追问。
- **自动验证器 + 重规划 + AgentReport**：智能体执行结果自动验证，失败时自动重规划，并生成结构化 `AgentReport` 报告。
- **10 个中文演示**（`qsl.ai.demos`）：覆盖 Grover、Shor、VQE、QAOA、QML 等场景的端到端中文示例。

### Fixed

- 修复 ZYZ 分解（`zyz_decompose`）返回角度错误，受控旋转门分解结果数值不正确的问题。
- 修复受控门分解误用整块矩阵而非正确 Barenco 分解路径的错误。
- 修复 `cswap`（Fredkin）门比特序错误。
- 修复 `_CX_MATRIX` 未定义导致的 NameError。
- 修复 `Dict` 类型标注导入缺失。
- 修复 torch 与 numpy 2.x 的兼容性问题。
- 修复 Grover 迭代次数计算偏差。
- 修复 Shor 算法在部分情形下不当退回经典回退路径的问题。

## [0.5.0] - 2025-XX

### Added

- 初始公开版本：声明式量子搜索 DSL、Grover / Shor / QAOA / VQE 算法、量子机器学习（QNN/QSVM/QGAN/QuantumLayer）、本地模拟器与 IBM/AWS 真机后端。
