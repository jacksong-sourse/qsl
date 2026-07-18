"""
Result Explainer - Translates quantum results into natural language.

Uses LLM to convert counts/expectation values/circuits into
human-readable explanations with uncertainty analysis.
"""

import numpy as np
from typing import Optional, Any
from dataclasses import dataclass


@dataclass
class Explanation:
    """Structured explanation of a quantum result."""
    summary: str
    details: str
    confidence: float
    uncertainty_sources: list[str]
    raw_interpretation: str


class ResultExplainer:
    """
    Explain quantum computation results in natural language.
    
    Args:
        model: LLM model name
        api_key: API key (default: env)
    """
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self._llm = None
    
    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        try:
            from . import _create_llm
            self._llm = _create_llm(model=self.model, api_key=self.api_key)
        except Exception:
            pass
        return self._llm
    
    def explain(self, quantum_result: Any, context: str = "") -> Explanation:
        """
        Generate a natural language explanation of quantum results.
        
        Args:
            quantum_result: Result object (GroverResult, dict, counts, etc.)
            context: Additional context about the problem
            
        Returns:
            Explanation with summary, details, and confidence
        """
        result_str = self._serialize_result(quantum_result)
        
        llm = self._get_llm()
        
        if llm:
            return self._llm_explain(llm, result_str, context)
        else:
            return self._rule_based_explain(result_str, context)
    
    def _serialize_result(self, result: Any) -> str:
        """Convert result to a string representation."""
        if result is None:
            return "No result"
        
        if hasattr(result, 'summary'):
            return result.summary()
        
        if isinstance(result, dict):
            return str(result)
        
        if isinstance(result, (list, tuple)):
            return f"Results: {result[:10]}" + ("..." if len(result) > 10 else "")
        
        return str(result)
    
    def _llm_explain(self, llm, result_str: str, context: str) -> Explanation:
        """Use LLM to generate explanation."""
        prompt = f"""Explain the following quantum computation result in simple, 
non-technical language. Include uncertainty analysis.

Context: {context or 'General quantum computation'}
Result: {result_str}

Format: Provide a concise summary (1 sentence), detailed explanation (2-3 sentences),
and identify sources of uncertainty."""

        try:
            response = llm.invoke(prompt)
            text = response.content
            
            sentences = text.split('. ')
            summary = sentences[0] + '.' if sentences else text
            details = '. '.join(sentences[1:]) if len(sentences) > 1 else text
            
            return Explanation(
                summary=summary,
                details=details,
                confidence=0.9,
                uncertainty_sources=["quantum noise", "measurement error", "sampling"],
                raw_interpretation=text,
            )
        except Exception:
            return self._rule_based_explain(result_str, context)
    
    def _rule_based_explain(self, result_str: str, context: str) -> Explanation:
        """Rule-based explanation fallback."""
        summary = f"Quantum computation completed: {context}"[:100]
        details = result_str[:300]
        
        uncertainty_sources = []
        if "shots" in result_str.lower() or "measurement" in result_str.lower():
            uncertainty_sources.append("finite sampling (shot noise)")
        if "simulator" not in result_str.lower():
            uncertainty_sources.append("hardware noise")
        
        return Explanation(
            summary=summary,
            details=details,
            confidence=0.8,
            uncertainty_sources=uncertainty_sources or ["algorithmic approximation"],
            raw_interpretation=result_str,
        )
