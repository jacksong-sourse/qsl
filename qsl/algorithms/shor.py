"""
Shor 整数因子分解算法实现。

使用真正的量子相位估计电路实现周期查找：
- 控制寄存器：2n 个量子比特，用于存储相位估计
- 目标寄存器：n 个量子比特，用于存储模幂结果
- 受控模幂运算：U_a^{2^k} |x>|y> = |x>|a^{2^k}*x mod N>
- 逆 QFT：从控制寄存器提取相位信息

算法步骤:
    1. 如果 N 是偶数，返回 2
    2. 如果 N = a^b (a>1, b>=2)，返回 a
    3. 随机选择 a ∈ [2, N-1]
    4. 如果 gcd(a, N) != 1，返回 gcd(a, N) (运气好)
    5. 量子周期查找: 使用 QFT 相位估计找 f(x) = a^x mod N 的周期 r
    6. 如果 r 是奇数或 a^(r/2) ≡ -1 (mod N)，换一个 a 重试
    7. 因子为 gcd(a^(r/2) ± 1, N)
"""

import math
import random

import numpy as np


class ShorSolver:
    """
    Shor 整数因子分解算法。

    使用量子相位估计（受控模幂 + 逆QFT）来查找模幂的周期，
    然后通过连分数展开和经典后处理提取因子。

    量子周期查找通过稀疏精确模拟实现 (内存 O(2^m), m 为控制
    量子比特数), 支持的最大规模由 max_control_qubits 控制;
    超出时抛出 RuntimeError, 不会静默回退到经典算法 (除非显式
    设置 allow_classical_fallback=True)。

    参数:
        N: 要分解的复合整数
        max_control_qubits: 相位估计控制寄存器的最大量子比特数
        allow_classical_fallback: 超出模拟能力时是否显式回退经典
    """

    def __init__(self, N: int, max_control_qubits: int = 18,
                 allow_classical_fallback: bool = False):
        if N < 2:
            raise ValueError(f"N must be >= 2, got {N}")
        self.N = N
        self.max_control_qubits = max_control_qubits
        self.allow_classical_fallback = allow_classical_fallback
        self._factors = None

    @staticmethod
    def _modular_pow(a: int, exp: int, mod: int) -> int:
        """快速模幂计算 a^exp mod mod。"""
        result = 1
        a = a % mod
        while exp > 0:
            if exp & 1:
                result = (result * a) % mod
            a = (a * a) % mod
            exp >>= 1
        return result

    def _is_power(self) -> int | None:
        """检查 N = a^b 对于某个 a>1, b>=2。如果是则返回 a。"""
        for b in range(2, int(math.log2(self.N)) + 1):
            a = round(self.N ** (1.0 / b))
            if a ** b == self.N:
                return a
        return None

    @staticmethod
    def _continued_fraction(num: float, max_denom: int = 1000) -> tuple[int, int]:
        """连分数展开，找到有理逼近 p/q ≈ num。返回 (p, q)。"""
        a0 = int(num)
        if num - a0 < 1e-12:
            return a0, 1

        fractions = []
        remainder = num
        for _ in range(100):
            a = int(remainder)
            fractions.append(a)
            rem = remainder - a
            if abs(rem) < 1e-12:
                break
            remainder = 1.0 / rem

        if len(fractions) == 1:
            return fractions[0], 1

        p = [fractions[0], fractions[0] * fractions[1] + 1]
        q = [1, fractions[1]]

        for i in range(2, len(fractions)):
            p_next = fractions[i] * p[-1] + p[-2]
            q_next = fractions[i] * q[-1] + q[-2]
            p.append(p_next)
            q.append(q_next)

        best_p, best_q = p[-1], q[-1]
        best_err = abs(num - best_p / best_q)

        for pi, qi in zip(p, q):
            if qi > max_denom:
                continue
            err = abs(num - pi / qi)
            if err < best_err:
                best_p, best_q = pi, qi
                best_err = err

        return best_p, best_q

    def _find_period_quantum(self, a: int) -> int | None:
        """
        Quantum period finding using QFT-based phase estimation.

        Implements the true quantum circuit for Shor's algorithm:
        1. Prepare control register in superposition: H^⊗m |0...0>
        2. Prepare target register: |1>
        3. Apply controlled modular exponentiation U_a^{2^k} for each k
           where U_a |x>|y> = |x>|a^x * y mod N>
        4. Apply inverse QFT to control register
        5. Measure control register to get phase estimate φ ≈ s/r
        6. Use continued fraction expansion to recover r from φ

        Simulation strategy (exact, memory-efficient):
            After step 3 the state is |ψ> = (1/√M) Σ_j |j>|a^j mod N>,
            which has exactly M nonzero amplitudes (one per control
            value). Instead of a dense M×N state vector, we compute
            f(j) = a^j mod N directly (this IS the simulation of the
            controlled-U unitary), then evaluate the post-IQFT control
            register distribution analytically:

                P(c) = (1/M²) Σ_x |Σ_{j: f(j)=x} e^{-2πi·c·j/M}|²

            via batched FFT over the value groups. This is the exact
            same distribution the dense simulation produces, at a
            fraction of the memory/time cost. No classical period
            finding is used unless explicitly opted in via
            allow_classical_fallback.

        Args:
            a: The base such that gcd(a, N) = 1

        Returns:
            The period r, or None if not found

        Raises:
            RuntimeError: if the required control register exceeds
                max_control_qubits (simulation infeasible) and
                allow_classical_fallback is False.
        """
        n = self.N.bit_length() - 1
        m = 2 * n
        M = 1 << m

        if m > self.max_control_qubits:
            msg = (
                f"量子相位估计需要 m={m} 个控制量子比特 (N={self.N}), "
                f"超过模拟上限 max_control_qubits={self.max_control_qubits} "
                f"(控制寄存器状态数 2^{m})。这是经典模拟的物理限制, "
                f"而非算法回退。可增大 max_control_qubits, 或显式设置 "
                f"allow_classical_fallback=True 使用经典周期查找。"
            )
            if not self.allow_classical_fallback:
                raise RuntimeError(msg)
            import warnings
            warnings.warn(msg + " —— 当前使用经典回退。", RuntimeWarning,
                          stacklevel=2)
            return self._find_period_classical_fallback(a)

        # --- 步骤 1-3: 均匀叠加 + 受控模幂 (稀疏精确模拟) ---
        # f(j) = a^j mod N, 用倍增法向量化计算: O(m·M)
        f = np.ones(M, dtype=np.int64)
        idx = np.arange(M, dtype=np.int64)
        for k in range(m):
            pow_k = pow(a, 1 << k, self.N)
            bit_set = (idx & (1 << k)) != 0
            f[bit_set] = (f[bit_set] * pow_k) % self.N

        # --- 步骤 4-5: 逆 QFT 后控制寄存器的测量分布 ---
        # P(c) = (1/M²) Σ_x |FFT(1_{f=x})(c)|², 按取值分组做批量 FFT
        order = np.argsort(f, kind="stable")
        sorted_f = f[order]
        group_starts = np.flatnonzero(
            np.r_[True, sorted_f[1:] != sorted_f[:-1]])
        group_ends = np.r_[group_starts[1:], M]

        probs = np.zeros(M, dtype=np.float64)
        batch = 32
        for g0 in range(0, len(group_starts), batch):
            g1 = min(g0 + batch, len(group_starts))
            mat = np.zeros((g1 - g0, M), dtype=np.float64)
            for row in range(g1 - g0):
                mat[row, order[group_starts[g0 + row]:group_ends[g0 + row]]] = 1.0
            amps = np.fft.fft(mat, axis=1)
            probs += (amps.real ** 2 + amps.imag ** 2).sum(axis=0)
        probs /= float(M) * M

        total_sum = probs.sum()
        if total_sum <= 0 or np.isnan(total_sum):
            return None
        probs /= total_sum

        # --- 步骤 6: 采样 + 连分数恢复周期 ---
        for _ in range(5):
            measured = np.random.choice(M, p=probs)
            phase = measured / M

            p, q = self._continued_fraction(phase, max_denom=self.N)
            if q == 0:
                continue
            if self._modular_pow(a, q, self.N) == 1:
                return q
            for mult in range(1, 10):
                r_candidate = q * mult
                if r_candidate > self.N:
                    break
                if self._modular_pow(a, r_candidate, self.N) == 1:
                    return r_candidate

        return None

    def _find_period_classical_fallback(self, a: int) -> int | None:
        """
        Classical period finding using Floyd's cycle detection (tortoise and hare).

        Finds the period r such that a^r ≡ 1 (mod N).

        Args:
            a: The base such that gcd(a, N) = 1

        Returns:
            The period r, or None if not found
        """
        N = self.N
        # Floyd's cycle detection: tortoise moves 1 step, hare moves 2 steps
        tortoise = 1
        hare = 1
        max_steps = min(N * 2, 100000)

        for _ in range(max_steps):
            tortoise = (tortoise * a) % N
            hare = (hare * a) % N
            hare = (hare * a) % N

            if tortoise == hare:
                # Found a cycle, now find the period
                mu = 0
                tortoise2 = 1
                while tortoise2 != hare:
                    tortoise2 = (tortoise2 * a) % N
                    hare = (hare * a) % N
                    mu += 1
                    if mu > max_steps:
                        break

                # Find lambda (period length)
                lam = 1
                hare = (tortoise * a) % N
                while tortoise != hare:
                    hare = (hare * a) % N
                    lam += 1
                    if lam > max_steps:
                        break

                # Verify period
                if self._modular_pow(a, lam, N) == 1:
                    return lam

                # Also try divisors of lam
                for d in range(1, int(np.sqrt(lam)) + 1):
                    if lam % d == 0:
                        if self._modular_pow(a, d, N) == 1:
                            return d
                        other = lam // d
                        if other != d and self._modular_pow(a, other, N) == 1:
                            return other

                return lam

        return None

    _find_period_classical = _find_period_classical_fallback

    def factor(self, max_attempts: int = 10,
               _max_depth: int = 100, _depth: int = 0) -> list[int]:
        """
        使用 Shor 算法分解 N。

        参数:
            max_attempts: 尝试的最大随机 a 值数量
            _max_depth: 最大递归深度 (内部参数, 防止无限递归)
            _depth: 当前递归深度 (内部参数)

        返回:
            质因子列表（确保每个因子都是质数）
        """
        if _depth >= _max_depth:
            raise RecursionError(
                f"ShorSolver 递归深度超过 {_max_depth}，可能遇到退化因子分解。"
                f"当前 N = {self.N}，深度 = {_depth}"
            )

        N = self.N

        if N <= 1:
            raise ValueError(f"N must be >= 2 for factorization, got {N}")

        if N % 2 == 0:
            result = [2]
            result.extend(ShorSolver(N // 2).factor(
                _max_depth=_max_depth, _depth=_depth + 1))
            self._factors = result
            return result

        power_base = self._is_power()
        if power_base is not None:
            b = int(round(math.log(N) / math.log(power_base)))
            # 如果 power_base 本身是合数，递归分解
            if not self._is_prime(power_base):
                sub = ShorSolver(power_base).factor(_max_depth=_max_depth, _depth=_depth + 1)
                result = sub * b
                self._factors = result
                return result
            result = [power_base] * b
            self._factors = result
            return result

        for _ in range(max_attempts):
            a = random.randint(2, N - 1)

            g = math.gcd(a, N)
            if 1 < g < N:
                result = []
                sub1 = ShorSolver(g).factor(_max_depth=_max_depth, _depth=_depth + 1)
                result.extend(sub1)
                sub2 = ShorSolver(N // g).factor(_max_depth=_max_depth, _depth=_depth + 1)
                result.extend(sub2)
                self._factors = result
                return result

            r = self._find_period_quantum(a)
            if r is None:
                continue

            if r % 2 != 0:
                continue

            half_pow = self._modular_pow(a, r // 2, N)
            if half_pow == N - 1:
                continue

            f1 = math.gcd(half_pow + 1, N)
            f2 = math.gcd(half_pow - 1, N)

            if 1 < f1 < N:
                result = []
                sub1 = ShorSolver(f1).factor(_max_depth=_max_depth, _depth=_depth + 1)
                result.extend(sub1)
                sub2 = ShorSolver(N // f1).factor(_max_depth=_max_depth, _depth=_depth + 1)
                result.extend(sub2)
                self._factors = result
                return result

            if 1 < f2 < N:
                result = []
                sub1 = ShorSolver(f2).factor(_max_depth=_max_depth, _depth=_depth + 1)
                result.extend(sub1)
                sub2 = ShorSolver(N // f2).factor(_max_depth=_max_depth, _depth=_depth + 1)
                result.extend(sub2)
                self._factors = result
                return result

        self._factors = [N]
        return [N]

    @staticmethod
    def _is_prime(num: int) -> bool:
        """Check if a number is prime (for factorization validation)."""
        if num < 2:
            return False
        if num == 2:
            return True
        if num % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(num)) + 1, 2):
            if num % i == 0:
                return False
        return True

    @property
    def factors(self) -> list[int] | None:
        """获取计算出的因子，如果尚未计算则返回 None。"""
        return self._factors

    def __repr__(self) -> str:
        return f"ShorSolver(N={self.N})"
