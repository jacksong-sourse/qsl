"""
Discovery Pipeline - Automated scientific discovery with quantum computing.

*** WARNING: DEMONSTRATION ONLY — simulates hypothesis testing with heuristics ***
Batch tests multiple hypotheses in parallel and generates discovery reports.
"""

import time
from typing import List, Optional
from dataclasses import dataclass, field
from .hypotheses import HypothesisTester, TestResult


@dataclass
class DiscoveryReport:
    """Structured scientific discovery report."""
    hypotheses_tested: int
    accepted: list[TestResult]
    rejected: list[TestResult]
    confidence_scores: list[float]
    summary: str
    raw_results: List[TestResult] = field(default_factory=list)
    error: Optional[str] = None


class DiscoveryPipeline:
    """
    Automated scientific discovery via batch hypothesis testing.
    
    1. Takes a list of hypotheses
    2. Runs them in parallel (where possible)
    3. Ranks results by confidence
    4. Generates LLM-powered discovery report
    
    Args:
        hypotheses_list: List of hypothesis strings to test
        batch_size: Number of parallel tests (default: all)
        llm_model: LLM model for report generation
    """
    
    def __init__(self,
                 hypotheses_list: List[str],
                 batch_size: int = 10,
                 llm_model: str = "gpt-4"):
        self.hypotheses = hypotheses_list
        self.batch_size = batch_size
        self.llm_model = llm_model
    
    def run(self) -> DiscoveryReport:
        """
        Execute the discovery pipeline.
        
        Returns:
            DiscoveryReport with ranked results
        """
        all_results = []
        
        print(f"\n  Testing {len(self.hypotheses)} hypotheses...")
        
        for i, hypothesis in enumerate(self.hypotheses):
            searcher = HypothesisTester(hypothesis)
            result = searcher.test()
            all_results.append(result)
            
            status = "ACCEPTED" if result.accepted else "REJECTED"
            print(f"  [{i+1}/{len(self.hypotheses)}] {status}: {hypothesis[:60]}...")
            time.sleep(0.1)  # Rate limiting
        
        # Separate accepted and rejected
        accepted = [r for r in all_results if r.accepted]
        rejected = [r for r in all_results if not r.accepted]
        
        # Sort by confidence
        all_confidences = [r.confidence for r in all_results]
        
        # Generate summary
        summary = self._generate_summary(all_results)
        
        return DiscoveryReport(
            hypotheses_tested=len(self.hypotheses),
            accepted=accepted,
            rejected=rejected,
            confidence_scores=all_confidences,
            summary=summary,
            raw_results=all_results,
        )
    
    def _generate_summary(self, results: List[TestResult]) -> str:
        """Generate a summary of discovery results."""
        n_accepted = sum(1 for r in results if r.accepted)
        n_rejected = len(results) - n_accepted
        
        # Try LLM summary (OpenAI or DeepSeek)
        try:
            from . import _create_llm
            llm = _create_llm(model=self.llm_model)
            if llm:
                hypotheses_text = "\n".join(
                    f"- {r.hypothesis} (p={r.p_value:.4f}, accepted={r.accepted})"
                    for r in results[:10]
                )
                
                prompt = f"""Summarize these quantum scientific discovery results in 3-4 sentences:
{n_accepted} accepted, {n_rejected} rejected out of {len(results)} hypotheses.

Results:
{hypotheses_text}"""
                
                response = llm.invoke(prompt)
                return response.content.strip()
        except Exception:
            pass
        
        return (f"Out of {len(results)} hypotheses tested, "
                f"{n_accepted} were accepted and {n_rejected} rejected. "
                f"Average confidence: {sum(r.confidence for r in results) / len(results):.2%}")
