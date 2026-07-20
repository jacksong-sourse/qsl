[← 返回文档中心](../index.md)

# 04 - Shor 大数分解算法

> **难度**：⭐⭐⭐ | **依赖**：`pip install qsl-quantum[algorithms]`（需要 scipy）

Shor 算法是 Peter Shor 于 1994 年提出的量子算法，能够在多项式时间内完成大整数分解，对经典 RSA 加密体系构成潜在威胁。其核心思想是将因子分解问题转化为**周期查找问题**，利用量子傅里叶变换（QFT）的指数加速能力高效求解。

---

## 📋 目录

1. [算法原理](#算法原理)
2. [安装依赖](#安装依赖)
3. [快速开始：分解 15](#快速开始分解-15)
4. [更多示例：分解 21、35](#更多示例分解-2135)
5. [结果验证](#结果验证)
6. [进阶参数说明](#进阶参数说明)
7. [注意事项与限制](#注意事项与限制)

---

## 算法原理

Shor 算法包含两大部分：**经典约化**和**量子周期查找**。

### 经典约化步骤

我们需要找到合数 $N$ 的一个非平凡因子 $p$（$1 < p < N$）：

1. 若 $N$ 是偶数，直接返回 $2$；
2. 若 $N = a^b$（$a>1, b\ge2$），返回 $a$；
3. 随机选取 $a \in [2, N-1]$，计算 $d = \gcd(a, N)$。若 $d > 1$，则 $d$ 就是因子（运气好！）；
4. 否则，利用量子算法求函数 $f(x) = a^x \bmod N$ 的**周期** $r$；
5. 若 $r$ 是偶数且 $a^{r/2} \not\equiv -1 \pmod{N}$，则因子为：
   $$p = \gcd(a^{r/2} + 1, N), \quad q = \gcd(a^{r/2} - 1, N)$$

### 量子周期查找

量子部分使用**量子相位估计（QPE）**电路：

- **控制寄存器**：$2n$ 个量子比特（$n = \lceil\log_2 N\rceil$），用于相位编码；
- **目标寄存器**：$n$ 个量子比特，存储模幂运算结果；
- **受控模幂**：$U_a^{2^k} |x\rangle|y\rangle = |x\rangle|a^{2^k} y \bmod N\rangle$；
- **逆 QFT**：从控制寄存器提取相位 $\phi \approx s/r$；
- **连分数展开**：从测量得到的相位 $\phi$ 恢复周期 $r$。

---

## 安装依赖

Shor 算法的量子模拟本身只依赖 numpy，但优化部分需要 scipy：

```bash
pip install "qsl-quantum[algorithms]"
# 或单独安装 scipy
pip install scipy
```

---

## 快速开始：分解 15

这是最经典的 Shor 算法示例——分解 $15 = 3 \times 5$：

```python
from qsl.algorithms import ShorSolver

# 创建 Shor 求解器实例
solver = ShorSolver(N=15)

# 执行分解
factors = solver.factor()

print(f"N = 15 的因子: {factors}")
print(f"验证: 15 = {' × '.join(map(str, factors))}")
print(f"所有因子都是质数: {all(solver._is_prime(f) for f in factors)}")
```

**预期输出**：
```
N = 15 的因子: [3, 5]
验证: 15 = 3 × 5
所有因子都是质数: True
```

---

## 更多示例：分解 21、35

### 示例 1：分解 21 = 3 × 7

```python
from qsl.algorithms import ShorSolver

solver = ShorSolver(N=21)
factors = solver.factor()

print(f"N = 21 的因子: {factors}")
product = 1
for f in factors:
    product *= f
print(f"乘积验证: {' × '.join(map(str, factors))} = {product}")
assert product == 21, "分解结果错误！"
print("✅ 验证通过！")
```

### 示例 2：分解 35 = 5 × 7

```python
from qsl.algorithms import ShorSolver
import numpy as np

# 设置随机种子以获得可复现结果
np.random.seed(42)

solver = ShorSolver(N=35, max_control_qubits=12)
factors = solver.factor()

print(f"N = 35 的因子: {factors}")
print(f"排序后: {sorted(factors)}")
```

### 示例 3：分解 9（质数幂 $3^2$）

ShorSolver 能自动检测质数幂并正确分解：

```python
from qsl.algorithms import ShorSolver

solver = ShorSolver(N=9)
factors = solver.factor()

print(f"N = 9 的因子: {factors}")  # 输出 [3, 3]
print(f"3^2 = 9: {3**2 == 9}")
```

---

## 结果验证

编写一个验证函数确保分解正确：

```python
from qsl.algorithms import ShorSolver
import math

def verify_shor(N: int, factors: list[int]) -> bool:
    """
    验证 Shor 分解结果的正确性。
    
    验证内容：
    1. 所有因子都是质数
    2. 因子乘积等于 N
    3. 每个因子都满足 1 < f < N 或 N 本身是质数
    """
    # 检查乘积
    product = 1
    for f in factors:
        product *= f
    if product != N:
        print(f"❌ 乘积错误: {product} != {N}")
        return False
    
    # 检查是否为质数
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    for f in factors:
        if not is_prime(f):
            print(f"❌ {f} 不是质数")
            return False
    
    print(f"✅ 验证通过: {N} = {' × '.join(map(str, factors))}")
    return True


# 测试多个数值
test_numbers = [15, 21, 33, 35, 51]

for N in test_numbers:
    solver = ShorSolver(N)
    factors = solver.factor()
    print(f"\n分解 N = {N}:")
    verify_shor(N, factors)
```

---

## 进阶参数说明

### ShorSolver 构造参数

```python
ShorSolver(
    N: int,                          # 待分解的合数
    max_control_qubits: int = 18,    # 控制寄存器最大量子比特数
    allow_classical_fallback: bool = False  # 是否允许经典回退
)
```

| 参数 | 说明 |
|------|------|
| `N` | 要分解的合数，必须 $\ge 2$ |
| `max_control_qubits` | 量子模拟的控制寄存器上限。控制寄存器大小为 $2n$，$n$ 是 $N$ 的比特长度。默认 18 个量子比特对应状态空间 $2^{18} = 262144$ 维 |
| `allow_classical_fallback` | 当量子模拟需要超出 `max_control_qubits` 时，是否回退到经典周期查找算法（Floyd 判圈算法）。默认 `False`，即严格使用量子算法 |

### factor() 方法参数

```python
solver.factor(max_attempts: int = 10)
```

- `max_attempts`：尝试不同随机底数 $a$ 的最大次数。量子算法有概率失败（约 50%），多次尝试可提高成功率。

### 属性

- `solver.factors`：获取已计算的因子列表（未计算时返回 `None`）。

---

## 注意事项与限制

### ⚠️ 经典模拟的物理限制

Shor 算法的量子相位估计需要 $2n$ 个控制量子比特，对应的状态向量维度为 $2^{2n} \times 2^n = 2^{3n}$（精确稀疏模拟为 $2^{2n}$）。这意味着：

| N（十进制） | n（比特数） | 控制量子比特 | 状态空间大小 |
|------------|------------|-------------|-------------|
| 15 | 4 | 8 | 256 |
| 21 | 5 | 10 | 1,024 |
| 51 | 6 | 12 | 4,096 |
| 200 | 8 | 16 | 65,536 |
| 1000 | 10 | 20 | 1,048,576 |

- 默认 `max_control_qubits=18` 可快速分解约 1000 以内的数字；
- 对于更大的数，请增加 `max_control_qubits`（但内存和时间会指数增长）；
- 如果只是想体验算法流程或测试大数，可设置 `allow_classical_fallback=True`。

### 示例：使用经典回退分解较大数

```python
from qsl.algorithms import ShorSolver

# 分解 1001 = 7 × 11 × 13（量子模拟需要较多资源，使用经典回退）
solver = ShorSolver(
    N=1001,
    allow_classical_fallback=True
)
factors = solver.factor()
print(f"1001 = {' × '.join(map(str, sorted(factors)))}")
# 输出: 1001 = 7 × 11 × 13
```

### 依赖提醒

如果未安装 scipy，部分优化功能可能受限，但 Shor 算法的核心量子模拟仅依赖 numpy，仍可正常运行：

```bash
# 最小安装也可运行 Shor（量子模拟部分）
pip install qsl-quantum

# 推荐完整安装
pip install "qsl-quantum[algorithms]"
```

---

## 完整可运行脚本

```python
"""
Shor 大数分解算法完整示例
"""
import numpy as np
from qsl.algorithms import ShorSolver


def is_prime(n: int) -> bool:
    """简单质数检测"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(np.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def verify_shor(N: int, factors: list[int]) -> bool:
    """验证分解结果"""
    product = 1
    for f in factors:
        product *= f
    
    all_prime = all(is_prime(f) for f in factors)
    product_ok = product == N
    
    if product_ok and all_prime:
        print(f"✅ {N} = {' × '.join(map(str, sorted(factors)))}")
        return True
    else:
        print(f"❌ {N} 分解失败: factors={factors}, product={product}")
        return False


def main():
    print("=" * 60)
    print("Shor 大数分解算法演示")
    print("=" * 60)
    
    # 可在经典模拟范围内快速求解的数字
    test_cases = [15, 21, 33, 35, 51, 65]
    
    np.random.seed(123)
    
    for N in test_cases:
        solver = ShorSolver(N, max_control_qubits=14)
        factors = solver.factor(max_attempts=15)
        verify_shor(N, factors)
    
    print("\n" + "=" * 60)
    print("使用经典回退演示分解更大的数:")
    print("=" * 60)
    
    large_N = 1001  # 7 × 11 × 13
    solver_large = ShorSolver(large_N, allow_classical_fallback=True)
    factors_large = solver_large.factor()
    verify_shor(large_N, factors_large)


if __name__ == "__main__":
    main()
```

---

## 🔗 相关阅读

- [03 - Grover 搜索算法](03_grover_search.md)
- [05 - QAOA 组合优化](05_qaoa_optimization.md)
- [算法 API 参考](../api/algorithms.md)
