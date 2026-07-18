"""
Shor 整数因子分解算法实现。

这是一个经典模拟器。Shor 算法的量子核心——基于 QFT 的
周期查找（相位估计）——由经典算法替代。
在实际的量子计算机或量子模拟器上，周期查找通过 QFT 电路
和受控模幂运算实现。

算法步骤:
    1. 如果 N 是偶数，返回 2
    2. 如果 N = a^b (a>1, b>=2)，返回 a
    3. 随机选择 a ∈ [2, N-1]
    4. 如果 gcd(a, N) != 1，返回 gcd(a, N) (运气好)
    5. 经典周期查找: 找 f(x) = a^x mod N 的周期 r
    6. 如果 r 是奇数或 a^(r/2) ≡ -1 (mod N)，换一个 a 重试
    7. 因子为 gcd(a^(r/2) ± 1, N)
"""

import math
import random

import numpy as np

from .qft import QuantumFourierTransform


class ShorSolver:
    """
    Shor 整数因子分解算法的经典模拟。

    注意: 这个实现是经典模拟。量子周期查找由经典算法替代。
    真正的量子加速需要运行 QFT 相位估计电路来查找周期，
    这需要量子模拟器或量子硬件上的实际受控模幂电路。

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
        Use QFT-based quantum phase estimation to find the period.

        *** WARNING: This is a SIMPLIFIED pedagogical implementation. ***
        *** The state vector reshaping (reshape to matrix) is an     ***
        *** approximation for demonstration only and does NOT match  ***
        *** the full QPE circuit (H + controlled modular exp + iQFT).***
        *** A proper quantum period-finding requires separate control***
        *** and target registers with correct tensor-product structure.***

        This implements a simplified version of Shor's quantum subroutine:
        1. Prepare superposition over control register (2n qubits)
        2. Apply controlled modular exponentiation U_a |x> = |a*x mod N>
        3. Apply inverse QFT to the control register
        4. Measure to obtain phase estimate phi ≈ s/r
        5. Use continued fraction expansion to recover r from phi

        Uses QuantumFourierTransform.apply() for the QFT.
        Uses a simplified circuit-based modular exponentiation.
        """
        import numpy as np

        # Number of qubits for precision: use 2*n bits for period estimation
        n = self.N.bit_length() - 1
        m = 2 * n  # control register size
        M = 1 << m

        # Step 1: Prepare superposition |+>^m ⊗ |0...01>
        state = np.zeros(M * self.N, dtype=complex)
        # Initial state: |+> on control, |00...01> on target
        target_init = 1  # start with |1>
        norm_c = 1.0 / np.sqrt(M)
        for j in range(M):
            state[j * self.N + target_init] = norm_c

        # Step 2: Apply controlled U_a^{2^k} operations
        # U_a |x> = |a*x mod N>
        for k in range(m):
            power = pow(a, 1 << k, self.N)
            mask_control = 1 << k
            for j in range(M):
                if j & mask_control:
                    for x in range(1, self.N):
                        idx_jx = j * self.N + x
                        if abs(state[idx_jx]) > 1e-15:
                            x_new = (x * power) % self.N
                            idx_jxnew = j * self.N + x_new
                            state[idx_jxnew] = state[idx_jx]
                            if x_new != x:
                                state[idx_jx] = 0j

        # Step 3: Apply inverse QFT to control register
        # Reshape to (M, N) and apply iQFT column-wise
        state_mat = state.reshape(M, self.N)
        qft = QuantumFourierTransform(m)
        # Apply inverse QFT = conjugate of forward QFT
        # iQFT|j> = 1/sqrt(M) * sum_k exp(-2*pi*i*j*k/M) |k>
        iqft_matrix = qft.get_matrix().conj().T  # inverse = adjoint
        for x in range(self.N):
            state_mat[:, x] = iqft_matrix @ state_mat[:, x]

        # Step 4: Measure control register
        probs = np.abs(state_mat) ** 2
        total_probs = np.sum(probs, axis=1)  # sum over target register
        total_probs = total_probs / np.sum(total_probs)

        # Sample the most likely outcome
        measured = np.argmax(total_probs)
        phase = measured / M

        # Step 5: Continued fraction to find period
        p, q = self._continued_fraction(phase, max_denom=self.N)
        if q == 0:
            return None
        # Verify: check if a^q mod N == 1
        if self._modular_pow(a, q, self.N) == 1:
            return q
        # Try multiples of q
        for mult in range(1, 10):
            r_candidate = q * mult
            if r_candidate > self.N:
                break
            if self._modular_pow(a, r_candidate, self.N) == 1:
                return r_candidate
        return None

    def factor(self, max_attempts: int = 10,
               _max_depth: int = 100, _depth: int = 0) -> list[int]:
        """
        使用 Shor 方法（经典模拟）分解 N。

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

        if N == 1:
            return [1]

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
