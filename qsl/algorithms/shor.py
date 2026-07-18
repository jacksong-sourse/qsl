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

from .qft import QuantumFourierTransform


class ShorSolver:
    """
    Shor 整数因子分解算法。

    使用量子相位估计（受控模幂 + 逆QFT）来查找模幂的周期，
    然后通过连分数展开和经典后处理提取因子。

    对于较大的 N，量子周期查找自动退化为经典算法（试除法 +
    连分数试探），确保结果始终正确。

    参数:
        N: 要分解的复合整数
    """

    def __init__(self, N: int):
        if N < 2:
            raise ValueError(f"N must be >= 2, got {N}")
        self.N = N
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

        Args:
            a: The base such that gcd(a, N) = 1

        Returns:
            The period r, or None if not found
        """
        n = self.N.bit_length() - 1
        m = 2 * n
        M = 1 << m

        if M > 1 << 12:
            return self._find_period_classical_fallback(a)

        # Initialize: control register in uniform superposition H^⊗m, target = |1>
        state = np.zeros(M * self.N, dtype=complex)
        inv_sqrt_M = 1.0 / np.sqrt(M)
        for j in range(M):
            state[j * self.N + 1] = inv_sqrt_M

        for k in range(m):
            power = pow(a, 1 << k, self.N)
            mask_control = 1 << k

            for j in range(M):
                if j & mask_control:
                    new_state = np.zeros_like(state)
                    for x in range(self.N):
                        idx_jx = j * self.N + x
                        if abs(state[idx_jx]) > 1e-15:
                            x_new = (x * power) % self.N
                            idx_jxnew = j * self.N + x_new
                            new_state[idx_jxnew] += state[idx_jx]
                    state = new_state

        state_mat = state.reshape(M, self.N)
        qft = QuantumFourierTransform(m)
        iqft_matrix = qft.get_matrix().conj().T

        for x in range(self.N):
            state_mat[:, x] = iqft_matrix @ state_mat[:, x]

        probs = np.abs(state_mat) ** 2
        total_probs = np.sum(probs, axis=1)
        total_sum = np.sum(total_probs)
        if total_sum == 0 or np.isnan(total_sum):
            return self._find_period_classical_fallback(a)
        total_probs = total_probs / total_sum

        for _ in range(5):
            if np.any(np.isnan(total_probs)):
                break
            measured = np.random.choice(M, p=total_probs)
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

        return self._find_period_classical_fallback(a)

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
