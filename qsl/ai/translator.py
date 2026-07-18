"""
Problem Translator - Converts natural language into QSL Programs.

Uses LLM (OpenAI/LangChain) to parse problem descriptions into
structured QSLProgram definitions.
"""

import re
from typing import Optional
from ..compiler.program import QSLProgram


class ProblemTranslator:
    """
    Translate natural language problem descriptions into QSL programs.
    
    Falls back to rule-based parsing when LLM is unavailable.
    
    Example:
        "Crack RSA-15" -> QSLProgram for factor(15) with Shor algorithm
        
    Args:
        model: LLM model name (default: "gpt-4")
        api_key: OpenAI API key (default: env OPENAI_API_KEY)
    """
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self._llm = None
    
    def _get_llm(self):
        """Lazy-load LLM client (OpenAI or DeepSeek)."""
        if self._llm is not None:
            return self._llm
        
        try:
            from . import _create_llm
            self._llm = _create_llm(model=self.model, api_key=self.api_key)
        except Exception:
            pass
        
        return self._llm
    
    def translate(self, problem_text: str) -> QSLProgram:
        """
        Translate a natural language problem into a QSL program.
        """
        llm = self._get_llm()
        
        if llm:
            return self._llm_translate(llm, problem_text)
        return self._rule_based_translate(problem_text)
    
    def _llm_translate(self, llm, problem_text: str) -> QSLProgram:
        """Use LLM to parse problem."""
        prompt = f"""Parse the following quantum computing problem into a structured format.
Respond with JSON containing: name, n_qubits, premises (list of boolean expressions), algorithm.

Problem: {problem_text}

Response format:
{{"name": "problem name", "n_qubits": int, "premises": ["expr1", "expr2"], "algorithm": "grover"|"shor"|"qaoa"|"vqe"}}
"""
        import json
        response = llm.invoke(prompt)
        
        try:
            data = json.loads(response.content)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                return self._rule_based_translate(problem_text)
        
        return QSLProgram(
            name=data.get("name", "AI Problem"),
            n_qubits=data.get("n_qubits", 4),
            premises=data.get("premises", []),
            main_algorithm=data.get("algorithm", "grover"),
        )
    
    def _rule_based_translate(self, problem_text: str) -> QSLProgram:
        """Rule-based fallback parsing."""
        text_lower = problem_text.lower()
        
        # Detect problem type
        if any(w in text_lower for w in ["factor", "rsa", "decrypt"]):
            # Extract number closest to keyword (not just first number)
            numbers = re.findall(r'\d+', problem_text)
            N = 15
            if numbers:
                # Find the number closest to keywords "factor", "RSA", "decrypt"
                keywords_positions = []
                for kw in ["factor", "rsa", "decrypt", "factorization"]:
                    pos = text_lower.find(kw)
                    if pos >= 0:
                        keywords_positions.append(pos)
                if keywords_positions:
                    for m in re.finditer(r'\d+', problem_text):
                        num_val = int(m.group())
                        num_pos = m.start()
                        for kw_pos in keywords_positions:
                            distance = abs(num_pos - kw_pos)
                            if distance < 50:  # within 50 chars of keyword
                                N = num_val
                                break
                        if N != 15:
                            break
                if N == 15:
                    N = int(numbers[0])
            n_qubits = max(4, (N.bit_length() * 2))
            return QSLProgram(
                name=f"Factor {N}",
                n_qubits=n_qubits,
                premises=[f"x0 | x1"],
                main_algorithm="shor",
            )
        
        elif any(w in text_lower for w in ["sat", "satisfy", "cnf", "clause"]):
            return QSLProgram(
                name="SAT Problem",
                n_qubits=4,
                premises=["x0 | ~x1", "x1 | x2"],
                main_algorithm="grover",
            )
        
        elif any(w in text_lower for w in ["optimize", "maxcut", "portfolio"]):
            return QSLProgram(
                name="Optimization",
                n_qubits=6,
                premises=[],
                main_algorithm="qaoa",
            )
        
        elif any(w in text_lower for w in ["energy", "ground", "chemistry", "molecule"]):
            return QSLProgram(
                name="Energy Calculation",
                n_qubits=4,
                premises=[],
                main_algorithm="vqe",
            )
        
        else:
            # Default: Grover search
            return QSLProgram(
                name="Search Problem",
                n_qubits=3,
                premises=["x0 & x1"],
                main_algorithm="grover",
            )
