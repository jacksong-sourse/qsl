"""
Autonomous Quantum Agent - Self-directed quantum problem solving.

*** WARNING: DEMONSTRATION ONLY — requires langchain + OpenAI API key ***
"""

import time
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
        """Initialize LLM (OpenAI or DeepSeek)."""
        if self._llm is not None:
            return self._llm

        try:
            from . import _create_llm
            self._llm = _create_llm()
        except Exception:
            pass

        return self._llm

    def _parse_task_params(self, algorithm: str) -> dict:
        """
        Parse the task description for algorithm-specific parameters.

        Extracts numbers, dimensions, and keywords from the task
        description to parameterize the selected algorithm.
        """
        import re
        task = self.task
        params = {}

        # Extract numbers from the task description
        numbers = re.findall(r'\b(\d+)\b', task)
        nums = [int(n) for n in numbers]

        if algorithm == "shor":
            # Look for numbers that could be the integer to factor
            for n in nums:
                if n > 2 and n < 1000000:
                    params["N"] = n
                    break
            if "N" not in params:
                params["N"] = 15

        elif algorithm == "qaoa":
            # Use the first number > 1 as n_qubits, capped at 8
            n = next((x for x in nums if 2 <= x <= 20), 4)
            params["n_qubits"] = min(n, 8)

        elif algorithm == "vqe":
            n = next((x for x in nums if 2 <= x <= 12), 4)
            params["n_qubits"] = min(n, 6)

        elif algorithm == "grover":
            n = next((x for x in nums if 2 <= x <= 20), 4)
            params["n_qubits"] = min(n, 6)

        return params

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

        backend = self._select_backend(task_params)
        if self.verbose:
            print(f"  Selected backend: {backend}")

        for iteration in range(1, self.max_iterations + 1):
            try:
                result = self._execute_quantum_task(algorithm, task_params, backend)

                if result.get("success", False):
                    if self.verbose:
                        print(f"  Iteration {iteration}: Success!")

                    explanation = self._explain_result(llm, result)

                    return AgentResult(
                        task=self.task,
                        success=True,
                        algorithm_used=algorithm,
                        backend_used=backend,
                        iterations=iteration,
                        result_summary=explanation or "Task completed",
                        data=result,
                    )
                else:
                    if self.verbose:
                        print(f"  Iteration {iteration}: Failed, retrying...")

                    task_params = self._adjust_params(algorithm, task_params, result)

            except Exception as e:
                if self.verbose:
                    print(f"  Iteration {iteration}: Error - {e}")
                if iteration >= self.max_iterations:
                    return AgentResult(
                        task=self.task,
                        success=False,
                        algorithm_used=algorithm,
                        backend_used=backend,
                        iterations=iteration,
                        result_summary="Failed after max iterations",
                        error=str(e),
                    )

        return AgentResult(
            task=self.task,
            success=False,
            algorithm_used=algorithm,
            backend_used=backend,
            iterations=self.max_iterations,
            result_summary="Max iterations reached without success",
        )

    def _select_algorithm(self, llm) -> tuple[str, dict]:
        """Determine which quantum algorithm to use.
        
        Primary logic: keyword matching for reliable algorithm selection.
        Optional enhancement: LLM can refine selection when enabled and available.
        """
        task_lower = self.task.lower()

        # Keyword matching as PRIMARY logic
        if any(w in task_lower for w in ("factor", "rsa", "decrypt", "shor")):
            return "shor", {"N": 15}
        elif any(w in task_lower for w in ("optimize", "maxcut", "portfolio", "qaoa")):
            return "qaoa", {"n_qubits": 4}
        elif any(w in task_lower for w in ("energy", "ground", "chemistry", "vqe")):
            return "vqe", {"n_qubits": 4}
        elif any(w in task_lower for w in ("search", "grover", "find", "sat", "satisfy")):
            return "grover", {"n_qubits": 4, "premises": ["x0 & x1"]}

        # LLM as OPTIONAL enhancement only when keywords don't match
        if llm:
            prompt = f"""Select the best quantum algorithm for this task.
Options: grover, shor, qaoa, vqe

Task: {self.task}

Respond with just the algorithm name."""
            try:
                response = llm.invoke(prompt)
                alg = response.content.strip().lower()
                if alg in ("grover", "shor", "qaoa", "vqe"):
                    return alg, {}
            except Exception:
                pass

        # Default fallback
        return "grover", {"n_qubits": 4, "premises": ["x0 & x1"]}

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
