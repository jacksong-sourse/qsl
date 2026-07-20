# QSL 文档中心

<div align="center">

**QSL (Quantum Search Language) — 全栈量子计算框架 · 中文友好 · AI 驱动**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/qsl-quantum?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/qsl-quantum/)
[![License](https://img.shields.io/badge/License-MIT-4CC61E)](../LICENSE)

</div>

---

## 📚 文档导航

### 🚀 入门教程

| 教程 | 难度 | 简介 |
|------|------|------|
| [01 - 快速开始](tutorial/01_getting_started.md) | ⭐ | 安装、第一个量子电路、Hello Quantum World |
| [02 - 电路基础](tutorial/02_circuit_basics.md) | ⭐⭐ | 量子门、电路构建、参数化电路、测量 |
| [03 - Grover 搜索算法](tutorial/03_grover_search.md) | ⭐⭐ | 无序数据库搜索、SAT 问题求解 |
| [04 - Shor 大数分解](tutorial/04_shor_factorization.md) | ⭐⭐⭐ | RSA 破解原理、量子周期查找 |
| [05 - QAOA 组合优化](tutorial/05_qaoa_optimization.md) | ⭐⭐⭐ | 最大割、TSP 等组合优化问题 |
| [06 - VQE 量子化学](tutorial/06_vqe_chemistry.md) | ⭐⭐⭐ | 变分量子本征求解器、基态能量计算 |
| [07 - AI 量子科学家](tutorial/07_ai_scientist.md) | ⭐⭐ | 自然语言驱动量子计算、自动验证 |
| [08 - 可视化指南](tutorial/08_visualization.md) | ⭐ | 电路图、Bloch 球、态可视化 |
| [09 - 命令行工具](tutorial/09_cli_guide.md) | ⭐ | CLI 使用、演示程序、批处理 |

### 📖 API 参考

- [核心 API 速查](api/core.md)
- [电路 API](api/circuit.md)
- [算法 API](api/algorithms.md)
- [AI 模块 API](api/ai.md)
- [可视化 API](api/visualization.md)

### 📁 其他资源

- [项目 README](../README.md)
- [变更日志](../CHANGELOG.md)
- [GitHub 仓库](https://github.com/jacksong-sourse/qsl)
- [Gitee 镜像](https://gitee.com/song-jack/qsl)
- [PyPI 包](https://pypi.org/project/qsl-quantum/)

---

## 🎯 设计理念

QSL 遵循以下设计原则：

1. **零依赖可运行**：核心仅依赖 numpy，`pip install qsl-quantum` 装完即可运行模拟器、Grover、Shor 基础功能
2. **中文友好**：完整中文文档、中文错误提示、中文 AI 演示
3. **Qiskit 兼容**：API 风格与 Qiskit 高度相似，门参数顺序（角度在前、比特在后）完全对齐，降低迁移成本
4. **AI 原生**：内置中文 AI 量子科学家，自然语言驱动算法选择、电路编译、结果验证
5. **工程化**：700+ 单元测试、CI 持续集成、类型注解完善

---

## 📦 安装方式

```bash
# 最小安装（仅 numpy）
pip install qsl-quantum

# 完整安装（含可视化、算法、AI）
pip install "qsl-quantum[full]"

# 验证安装
python -c "import qsl; print(qsl.__version__)"
python -m qsl --version
```

---

## 💡 5 分钟快速体验

```python
from qsl import QuantumCircuit

# 创建 Bell 态电路
qc = QuantumCircuit(2)
qc.h(0)       # Hadamard 门创建叠加
qc.cx(0, 1)   # CNOT 门创建纠缠

# 执行模拟
result = qc.execute(shots=1024)
print("测量结果:", result.counts)
print("态向量:", result.statevector())
result.state.pretty_print()
```

运行 CLI 查看内置演示：

```bash
python -m qsl --list-demos    # 列出 10 个中文 AI 演示
python -m qsl --ai-demo 1     # 运行整数分解演示
```

---

## 🆘 获取帮助

- **问题反馈**：[GitHub Issues](https://github.com/jacksong-sourse/qsl/issues)
- **示例代码**：参考 `tutorial/` 目录下的可运行示例
- **内置演示**：`python -m qsl --list-demos` 查看 10 个端到端演示

---

## 📄 许可证

MIT License © 2026 Song Ziming
