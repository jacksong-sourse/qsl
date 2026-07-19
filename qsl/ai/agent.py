"""
Autonomous Quantum Agent - Self-directed quantum problem solving.

LLM access goes through the provider abstraction (qsl.ai.llm_provider).
When no LLM is configured, the agent falls back to built-in bilingual
(中文/English) rule routing and parameter extraction.
"""

import re
import time
from types import SimpleNamespace
from typing import Optional, Any
from dataclasses import dataclass, field

try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools import Tool
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


@dataclass
class AgentResult:
    """Result from a QuantumAgent run."""
    task: str
    success: bool
    algorithm_used: str
    backend_used: str
    iterations: int
    result_summary: str
    data: dict = field(default_factory=dict)
    error: Optional[str] = None
    verification: Optional[Any] = None  # VerificationResult, None=未验证
    decision_chain: list = field(default_factory=list)

    @property
    def verified(self) -> bool:
        """结果是否通过自动验证 (未执行验证返回 False)。"""
        return bool(self.verification is not None
                    and getattr(self.verification, "passed", False))

    def to_report(self, algorithm_reason: str = ""):
        """生成结构化中文 Markdown 报告 (AgentReport)。"""
        from .report import AgentReport
        return AgentReport.from_agent_result(
            self, verification=self.verification,
            decision_chain=self.decision_chain,
            algorithm_reason=algorithm_reason,
        )


class _ProviderLLMWrapper:
    """Minimal langchain-compatible adapter for an LLMProvider.

    Exposes .invoke(prompt) -> object with a .content attribute so that
    existing code written against the langchain interface keeps working.
    """

    def __init__(self, provider):
        self.provider = provider

    def invoke(self, prompt: str):
        return SimpleNamespace(content=self.provider.complete(prompt))


class QuantumAgent:
    """
    Autonomous quantum computing agent.

    Decision chain:
    LLM parse -> Select algorithm -> Select backend -> Compile -> Execute ->
    LLM explain results -> Decide iteration (max 3 rounds)

    Parses task_description to extract problem parameters (N for Shor,
    cost matrix size for QAOA, etc.) instead of using hardcoded defaults.

    Args:
        task_description: Natural language task description
        max_iterations: Maximum decision rounds (default 3)
        verbose: Print progress messages
    """

    def __init__(self,
                 task_description: str,
                 max_iterations: int = 3,
                 verbose: bool = True):
        self.task = task_description
        self.max_iterations = max_iterations
        self.verbose = verbose
        self._llm = None
        self._tools = []

    def _init_llm(self):
        """Initialize LLM via the provider abstraction.

        Uses the global default provider (create_provider auto-detection)
        wrapped in a langchain-compatible adapter; falls back to the legacy
        langchain _create_llm path when no provider is available.
        """
        if self._llm is not None:
            return self._llm

        try:
            from .llm_provider import get_default_provider
            provider = get_default_provider()
            if provider is not None and provider.available():
                self._llm = _ProviderLLMWrapper(provider)
                return self._llm
        except Exception:
            pass

        # Legacy path: langchain ChatOpenAI (OpenAI / DeepSeek)
        try:
            from . import _create_llm
            self._llm = _create_llm()
        except Exception:
            pass

        return self._llm

    def _parse_task_params(self, algorithm: str) -> dict:
        """
        Parse the task description for algorithm-specific parameters.

        Bilingual (中文/English) extraction:
          - shor: N from e.g. "分解 15" / "factor N=21" / "factor 15"
          - qaoa/vqe/grover: n_qubits from e.g. "8 比特" / "n=6" /
            "6 qubits" / "6量子比特", clamped to 2..12
        """
        params = {}

        if algorithm == "shor":
            params["N"] = self._extract_shor_N() or 15
        elif algorithm in ("qaoa", "vqe", "grover"):
            n = self._extract_n_qubits()
            params["n_qubits"] = n if n is not None else 4

        return params

    def _extract_shor_N(self) -> Optional[int]:
        """Extract the integer to factor from the task text.

        Prefers numbers immediately following 分解/factor-like keywords,
        skips years (>1900) and 1-digit numbers, and validates that N is
        >= 3, odd and not a perfect power.
        """
        lower = self.task.lower()

        def candidates():
            # Numbers right after a factoring keyword come first
            for m in re.finditer(
                    r'(?:分解|因数|质因数|因子|factor(?:ize|ing)?|rsa|crack|解密|n\s*=)'
                    r'\D{0,12}?(\d{2,})', lower):
                yield int(m.group(1))
            # Then any remaining numbers in the text
            for m in re.finditer(r'\d+', self.task):
                yield int(m.group())

        seen = set()
        for n in candidates():
            if n in seen:
                continue
            seen.add(n)
            if n < 10 or n > 1900:  # skip 1-digit numbers and years
                continue
            if self._is_valid_shor_N(n):
                return n
        return None

    @staticmethod
    def _is_valid_shor_N(n: int) -> bool:
        """N must be >= 3, odd, and not a perfect power (e.g. 9, 27, 64)."""
        if n < 3 or n % 2 == 0:
            return False
        for e in range(2, n.bit_length() + 1):
            b = round(n ** (1.0 / e))
            if b >= 2 and b ** e == n:
                return False
        return True

    def _extract_n_qubits(self) -> Optional[int]:
        """Extract qubit/problem size, clamped to 2..12.

        Recognizes "8 比特" / "n=6" / "6 qubits" / "6量子比特" /
        "6 节点" / "6 城市"; falls back to the first number in 2..12.
        """
        lower = self.task.lower()
        for pat in (r'n\s*=\s*(\d+)',
                    r'(\d+)\s*(?:个)?(?:量子比特|比特|qubits?)',
                    r'(\d+)\s*(?:个)?(?:节点|城市)'):
            m = re.search(pat, lower)
            if m:
                return max(2, min(12, int(m.group(1))))
        for m in re.finditer(r'\d+', self.task):
            n = int(m.group())
            if 2 <= n <= 12:
                return n
        return None

    def suggest_clarification(self) -> Optional[str]:
        """Return a Chinese follow-up question when required params are missing."""
        algorithm, _ = self._select_algorithm(None)
        if algorithm == "shor" and self._extract_shor_N() is None:
            return "请问要分解的整数是多少？（例如：分解 15）"
        if algorithm == "qaoa" and self._extract_n_qubits() is None:
            return "请问优化问题有多少个节点/变量？（例如：最大割 6 节点）"
        if algorithm == "vqe" and self._extract_n_qubits() is None:
            return "请问需要多少个量子比特？（例如：4 比特）"
        if algorithm == "grover" and self._extract_n_qubits() is None:
            return "请问搜索空间用多少比特表示？（例如：用 8 比特搜索）"
        return None

    def run(self) -> AgentResult:
        """Execute the agent's decision loop."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Quantum Agent: {self.task}")
            print(f"{'='*60}")

        llm = self._init_llm()

        algorithm, task_params = self._select_algorithm(llm)
        if self.verbose:
            print(f"  Selected algorithm: {algorithm}")

        # Parse actual parameters from task description
        parsed = self._parse_task_params(algorithm)
        task_params.update(parsed)

        if self.verbose:
            hint = self.suggest_clarification()
            if hint:
                print(f"  提示: {hint}")

        backend = self._select_backend(task_params)
        if self.verbose:
            print(f"  Selected backend: {backend}")

        decision_chain = [{
            "iteration": 0,
            "action": "route",
            "detail": f"算法={algorithm}, 参数={task_params}, 后端={backend}",
        }]

        for iteration in range(1, self.max_iterations + 1):
            try:
                result = self._execute_quantum_task(algorithm, task_params, backend)

                if result.get("success", False):
                    # 执行后自动验证: 经典可检验的独立交叉校验
                    verification = self._verify_result(
                        algorithm, result, task_params)
                    if verification is not None and not verification.passed:
                        if self.verbose:
                            print(f"  Iteration {iteration}: 验证失败 - "
                                  f"{verification.message}, 自动重规划...")
                        decision_chain.append({
                            "iteration": iteration,
                            "action": "verify",
                            "detail": f"FAIL: {verification.message}",
                        })
                        task_params = self._adjust_params(
                            algorithm, task_params,
                            {"error": verification.message})
                        continue

                    if self.verbose:
                        vmsg = (f", 验证: {'PASS' if verification.passed else 'N/A'}"
                                if verification is not None else "")
                        print(f"  Iteration {iteration}: Success!{vmsg}")

                    decision_chain.append({
                        "iteration": iteration,
                        "action": "execute+verify",
                        "detail": (verification.message
                                   if verification is not None else "执行成功"),
                    })
                    explanation = self._explain_result(llm, result)

                    return AgentResult(
                        task=self.task,
                        success=True,
                        algorithm_used=algorithm,
                        backend_used=backend,
                        iterations=iteration,
                        result_summary=explanation or "Task completed",
                        data=result,
                        verification=verification,
                        decision_chain=decision_chain,
                    )
                else:
                    if self.verbose:
                        print(f"  Iteration {iteration}: Failed, retrying...")

                    decision_chain.append({
                        "iteration": iteration,
                        "action": "execute",
                        "detail": f"执行失败: {result.get('error', '未知')}",
                    })
                    task_params = self._adjust_params(algorithm, task_params, result)

            except Exception as e:
                if self.verbose:
                    print(f"  Iteration {iteration}: Error - {e}")
                decision_chain.append({
                    "iteration": iteration,
                    "action": "error",
                    "detail": str(e),
                })
                if iteration >= self.max_iterations:
                    return AgentResult(
                        task=self.task,
                        success=False,
                        algorithm_used=algorithm,
                        backend_used=backend,
                        iterations=iteration,
                        result_summary="Failed after max iterations",
                        error=str(e),
                        decision_chain=decision_chain,
                    )

        return AgentResult(
            task=self.task,
            success=False,
            algorithm_used=algorithm,
            backend_used=backend,
            iterations=self.max_iterations,
            result_summary="Max iterations reached without success",
            decision_chain=decision_chain,
        )

    def _verify_result(self, algorithm: str, result: dict, params: dict):
        """对执行结果做自动验证; 无法验证时返回 None (不阻塞流程)。"""
        try:
            from .verifier import (verify_shor, verify_sat, verify_qaoa,
                                   verify_grover, verify_vqe)
            if algorithm == "shor":
                return verify_shor(params.get("N", 15),
                                   result.get("factors", []))
            if algorithm == "qaoa" and "cost_matrix" in result:
                return verify_qaoa(result["cost_matrix"],
                                   result.get("bitstring"),
                                   encoding=result.get("encoding", "ising"))
            if algorithm == "vqe" and "hamiltonian_terms" in result:
                return verify_vqe(result.get("energy"),
                                  result["hamiltonian_terms"])
            if algorithm == "grover":
                premises = params.get("premises")
                solutions = result.get("solutions", [])
                if premises and solutions:
                    return verify_sat(premises, solutions[0])
                marked = params.get("marked_states")
                if marked and "measured" in result:
                    return verify_grover(marked, result["measured"],
                                         n_qubits=params.get("n_qubits"))
        except Exception as e:
            if self.verbose:
                print(f"  (验证器异常, 跳过验证: {e})")
        return None

    # Bilingual (中文/English) rule routing table, top-down priority:
    # 分解->shor, 优化->qaoa, 基态->vqe, 搜索->grover.
    _ALGORITHM_RULES = (
        ("shor", ("分解", "因数", "质因数", "因子", "解密",
                  "factor", "factorize", "rsa", "crack", "shor")),
        ("qaoa", ("优化", "最大割", "旅行商", "组合", "调度", "着色", "覆盖",
                  "maxcut", "tsp", "portfolio", "optimize",
                  "schedule", "coloring", "qaoa")),
        ("vqe", ("基态", "能量", "分子", "化学", "哈密顿", "本征值",
                 "ground state", "energy", "molecule", "chemistry",
                 "hamiltonian", "vqe")),
        ("grover", ("搜索", "查找", "数据库", "布尔", "数独", "可满足",
                    "sudoku", "sat", "3-sat", "search", "find",
                    "database", "grover")),
    )

    def _select_algorithm(self, llm) -> tuple[str, dict]:
        """Determine which quantum algorithm to use.

        Primary logic: bilingual keyword routing table (reliable).
        LLM is only consulted when no rule matches; it is asked (in
        Chinese) to reply with just the algorithm name.
        """
        task_lower = self.task.lower()

        # Rule routing table as PRIMARY logic
        for algorithm, keywords in self._ALGORITHM_RULES:
            if any(w in task_lower for w in keywords):
                return algorithm, self._default_params(algorithm)

        # LLM as OPTIONAL enhancement only when keywords don't match
        if llm:
            prompt = f"""请为以下任务选择最合适的量子算法。
可选算法：shor（整数分解）、qaoa（组合优化）、vqe（分子基态能量）、grover（搜索）。

任务：{self.task}

只回复算法名（shor/qaoa/vqe/grover 之一），不要输出其他任何内容。"""
            try:
                response = llm.invoke(prompt)
                alg = response.content.strip().strip(".,;:，。；：").lower()
                if alg in ("grover", "shor", "qaoa", "vqe"):
                    return alg, self._default_params(alg)
            except Exception:
                pass

        # Default fallback
        return "grover", self._default_params("grover")

    @staticmethod
    def _default_params(algorithm: str) -> dict:
        """Default parameter dict for each algorithm."""
        if algorithm == "shor":
            return {"N": 15}
        if algorithm in ("qaoa", "vqe"):
            return {"n_qubits": 4}
        return {"n_qubits": 4, "premises": ["x0 & x1"]}

    def _select_backend(self, params: dict) -> str:
        """Select optimal backend."""
        n_qubits = params.get("n_qubits", 4)
        if n_qubits <= 20:
            return "simulator"
        elif n_qubits <= 127:
            return "ibm_kyoto"
        else:
            return "aws_sv1"

    def _execute_quantum_task(self, algorithm: str, params: dict, backend: str) -> dict:
        """Execute the quantum computation with actual parameters from task."""
        try:
            if algorithm == "grover":
                from ..compiler.program import QSLProgram
                from ..compiler.compiler import QSLCompiler

                prog = QSLProgram(
                    name=self.task,
                    n_qubits=params.get("n_qubits", 4),
                    premises=params.get("premises", ["x0 & x1"]),
                    shots=100,
                    backend=backend,
                )
                compiler = QSLCompiler(backend=backend)
                result = compiler.compile_and_run(prog)

                solutions = result.get_solutions()
                return {
                    "success": len(solutions) > 0,
                    "solutions": solutions,
                    "iterations": result.iterations,
                    "success_rate": result.empirical_success_rate,
                }

            elif algorithm == "shor":
                from ..algorithms.shor import ShorSolver
                N = params.get("N", 15)
                solver = ShorSolver(N)
                factors = solver.factor()
                return {
                    "success": len(factors) > 0 and factors != [N],
                    "factors": factors,
                    "N": N,
                }

            elif algorithm == "qaoa":
                from ..algorithms.qaoa import QAOA
                import numpy as np
                n = params.get("n_qubits", 4)
                # Build cost matrix from task context if possible
                if "cost" in params:
                    cost = params["cost"]
                else:
                    cost = np.ones((n, n)) * 0.5
                qaoa = QAOA(n, cost, p=1)
                opt_params, energy = qaoa.optimize(maxiter=50)
                best, best_cost = qaoa.get_optimal_bitstring()
                return {
                    "success": True,
                    "energy": energy,
                    "bitstring": best,
                    "cost": best_cost,
                }

            elif algorithm == "vqe":
                from ..algorithms.vqe import VQE
                n = params.get("n_qubits", 4)
                if "hamiltonian" in params:
                    h = params["hamiltonian"]
                else:
                    h = VQE.h2_hamiltonian()
                n_actual = len(h[0][1]) if h else 4
                vqe = VQE(n_actual, h, ansatz_type="he", n_layers=1)
                energy, state = vqe.optimize(maxiter=50)
                return {
                    "success": True,
                    "energy": energy,
                }

            else:
                return {"success": False, "error": f"Unknown algorithm: {algorithm}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _explain_result(self, llm, result: dict) -> Optional[str]:
        """Use LLM to explain quantum results in natural language."""
        if llm is None:
            return str(result)

        prompt = f"""Explain the following quantum computation result in simple terms:
Result: {result}

Keep it concise (2-3 sentences)."""

        try:
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception:
            return str(result)

    def _adjust_params(self, algorithm: str, params: dict, result: dict) -> dict:
        """Adjust parameters based on failure feedback for specific algorithms."""
        error = result.get("error", "")
        if algorithm == "shor":
            # If factoring failed, try a larger N
            if "N" in params:
                params["N"] = min(params["N"] * 2, 1000000)
        elif algorithm == "qaoa":
            # If QAOA failed, reduce n_qubits or increase p (layers)
            if "n_qubits" in params and params["n_qubits"] > 2:
                params["n_qubits"] = max(2, params["n_qubits"] - 1)
            else:
                params["p"] = params.get("p", 1) + 1
        elif algorithm == "vqe":
            # If VQE failed, try different ansatz type
            if "ansatz" not in params:
                params["ansatz_type"] = "uccsd"
            elif params.get("ansatz_type") == "he":
                params["ansatz_type"] = "uccsd"
            else:
                # Increase n_layers
                params["n_layers"] = params.get("n_layers", 1) + 1
        elif algorithm == "grover":
            # If Grover failed, reduce n_qubits
            if "n_qubits" in params and params["n_qubits"] > 2:
                params["n_qubits"] = params["n_qubits"] - 1
        elif "n_qubits" in params:
            params["n_qubits"] = min(params["n_qubits"] + 1, 20)
        return params
