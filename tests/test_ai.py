"""
测试 AI 量子科学家模块。

覆盖:
    - ProblemTranslator: 自然语言 -> QSLProgram 翻译
    - QuantumAgent: 自主量子计算代理
    - HypothesisTester: 科学假设检验
    - DiscoveryPipeline: 自动化科学发现流水线
    - ResultExplainer: 量子结果自然语言解释
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from dataclasses import is_dataclass, fields


# ==============================================================================
# TestProblemTranslator
# ==============================================================================

class TestProblemTranslator:

    def test_rule_based_translate_factor(self):
        """_rule_based_translate handles factor problem -> QSLProgram with correct name."""
        from qsl.ai.translator import ProblemTranslator
        from qsl.compiler.program import QSLProgram

        translator = ProblemTranslator()
        result = translator._rule_based_translate("Factor 21 into primes")
        assert isinstance(result, QSLProgram)
        assert result.name == "Factor 21"
        assert result.main_algorithm == "shor"
        assert result.n_qubits >= 4

    def test_rule_based_translate_sat(self):
        """_rule_based_translate handles SAT problem -> grover program."""
        from qsl.ai.translator import ProblemTranslator
        from qsl.compiler.program import QSLProgram

        translator = ProblemTranslator()
        result = translator._rule_based_translate("SAT problem with 3 clauses")
        assert isinstance(result, QSLProgram)
        assert result.name == "SAT Problem"
        assert result.main_algorithm == "grover"
        assert result.n_qubits == 4
        assert "x0 | ~x1" in result.premises

    def test_rule_based_translate_optimization(self):
        """_rule_based_translate handles optimization problem -> qaoa program."""
        from qsl.ai.translator import ProblemTranslator
        from qsl.compiler.program import QSLProgram

        translator = ProblemTranslator()
        result = translator._rule_based_translate("optimize portfolio allocation")
        assert isinstance(result, QSLProgram)
        assert result.name == "Optimization"
        assert result.main_algorithm == "qaoa"
        assert result.n_qubits == 6

    def test_rule_based_translate_energy(self):
        """_rule_based_translate handles energy problem -> vqe program."""
        from qsl.ai.translator import ProblemTranslator
        from qsl.compiler.program import QSLProgram

        translator = ProblemTranslator()
        result = translator._rule_based_translate("Calculate ground state energy of molecule")
        assert isinstance(result, QSLProgram)
        assert result.name == "Energy Calculation"
        assert result.main_algorithm == "vqe"
        assert result.n_qubits == 4

    def test_rule_based_translate_unknown(self):
        """_rule_based_translate handles unknown problem -> grover default."""
        from qsl.ai.translator import ProblemTranslator
        from qsl.compiler.program import QSLProgram

        translator = ProblemTranslator()
        result = translator._rule_based_translate("do something random")
        assert isinstance(result, QSLProgram)
        assert result.name == "Search Problem"
        assert result.main_algorithm == "grover"
        assert result.n_qubits == 3

    def test_translate_exists_and_returns_qsl_program(self):
        """test translate method exists and returns QSLProgram."""
        from qsl.ai.translator import ProblemTranslator
        from qsl.compiler.program import QSLProgram

        translator = ProblemTranslator()
        assert hasattr(translator, "translate")
        assert callable(translator.translate)

        # With no API key, falls back to rule-based
        result = translator.translate("Factor 15")
        assert isinstance(result, QSLProgram)
        assert result.main_algorithm in ("shor", "grover", "qaoa", "vqe")


# ==============================================================================
# TestQuantumAgent
# ==============================================================================

class TestQuantumAgent:

    def test_initializes_with_task(self):
        """test initializes with task description."""
        from qsl.ai.agent import QuantumAgent

        agent = QuantumAgent(task_description="Find RSA factors", verbose=False)
        assert agent.task == "Find RSA factors"
        assert agent.max_iterations == 3

    def test_select_algorithm_factor(self):
        """_select_algorithm detects 'factor' keyword -> shor."""
        from qsl.ai.agent import QuantumAgent

        agent = QuantumAgent(task_description="factor the number 21", verbose=False)
        algorithm, params = agent._select_algorithm(None)
        assert algorithm == "shor"
        assert "N" in params

    def test_select_algorithm_optimize(self):
        """_select_algorithm detects 'optimize' keyword -> qaoa."""
        from qsl.ai.agent import QuantumAgent

        agent = QuantumAgent(task_description="optimize this portfolio", verbose=False)
        algorithm, params = agent._select_algorithm(None)
        assert algorithm == "qaoa"

    def test_select_algorithm_energy(self):
        """_select_algorithm detects 'energy' keyword -> vqe."""
        from qsl.ai.agent import QuantumAgent

        agent = QuantumAgent(task_description="find ground state energy", verbose=False)
        algorithm, params = agent._select_algorithm(None)
        assert algorithm == "vqe"

    def test_select_algorithm_default(self):
        """_select_algorithm defaults to grover."""
        from qsl.ai.agent import QuantumAgent

        agent = QuantumAgent(task_description="find something interesting", verbose=False)
        algorithm, params = agent._select_algorithm(None)
        assert algorithm == "grover"
        assert "premises" in params

    def test_select_backend_simulator(self):
        """_select_backend returns simulator for <=20 qubits."""
        from qsl.ai.agent import QuantumAgent

        agent = QuantumAgent(task_description="test", verbose=False)
        backend = agent._select_backend({"n_qubits": 4})
        assert backend == "simulator"

        backend = agent._select_backend({"n_qubits": 20})
        assert backend == "simulator"

        backend = agent._select_backend({"n_qubits": 21})
        assert backend != "simulator"

    def test_execute_grover_simple(self):
        """_execute_quantum_task grover works (simple SAT)."""
        from qsl.ai.agent import QuantumAgent

        agent = QuantumAgent(task_description="grover test", verbose=False)
        result = agent._execute_quantum_task(
            algorithm="grover",
            params={"n_qubits": 3, "premises": ["x0 & x1"]},
            backend="simulator",
        )
        assert isinstance(result, dict)
        assert "success" in result
        assert "solutions" in result

    def test_execute_shor_n15(self):
        """_execute_quantum_task shor works for N=15."""
        from qsl.ai.agent import QuantumAgent

        agent = QuantumAgent(task_description="shor test", verbose=False)
        result = agent._execute_quantum_task(
            algorithm="shor",
            params={"N": 15},
            backend="simulator",
        )
        assert isinstance(result, dict)
        assert "success" in result
        assert "factors" in result

    def test_agent_result_dataclass(self):
        """AgentResult dataclass is correct."""
        from qsl.ai.agent import AgentResult

        assert is_dataclass(AgentResult)

        result = AgentResult(
            task="test task",
            success=True,
            algorithm_used="grover",
            backend_used="simulator",
            iterations=1,
            result_summary="completed",
            data={"key": "value"},
        )
        assert result.task == "test task"
        assert result.success is True
        assert result.algorithm_used == "grover"
        assert result.backend_used == "simulator"
        assert result.iterations == 1
        assert result.result_summary == "completed"
        assert result.data == {"key": "value"}
        assert result.error is None

        # Test error case
        fail_result = AgentResult(
            task="fail",
            success=False,
            algorithm_used="shor",
            backend_used="simulator",
            iterations=3,
            result_summary="failed",
            error="something went wrong",
        )
        assert fail_result.error == "something went wrong"


# ==============================================================================
# TestHypothesisTester
# ==============================================================================

class TestHypothesisTester:

    def test_initializes_with_hypothesis(self):
        """test initializes with hypothesis text."""
        from qsl.ai.hypotheses import HypothesisTester

        tester = HypothesisTester("Compound C6H6 ground state energy < -230 Hartree")
        assert tester.hypothesis == "Compound C6H6 ground state energy < -230 Hartree"
        assert tester.alpha == 0.05

    def test_parse_energy_less_than(self):
        """_parse_hypothesis extracts energy < -230 correctly."""
        from qsl.ai.hypotheses import HypothesisTester

        tester = HypothesisTester("Compound C6H6 ground state energy < -230 Hartree")
        quantity, operator, threshold = tester._parse_hypothesis()
        assert quantity == "energy"
        assert operator == "<"
        assert threshold == -230.0

    def test_parse_greater_equal(self):
        """_parse_hypothesis extracts >= operator."""
        from qsl.ai.hypotheses import HypothesisTester

        tester = HypothesisTester("energy >= -100.5")
        quantity, operator, threshold = tester._parse_hypothesis()
        assert quantity == "energy"
        assert operator == ">="
        assert threshold == -100.5

    def test_test_energy_returns_testresult(self):
        """_test_energy returns TestResult with fields populated."""
        from qsl.ai.hypotheses import HypothesisTester, TestResult

        tester = HypothesisTester("energy < -1.0 Hartree")
        result = tester._test_energy(threshold=-1.0, operator="<")
        assert isinstance(result, TestResult)
        assert result.hypothesis == "energy < -1.0 Hartree"
        assert "p_value" in fields(result) or hasattr(result, "p_value")
        assert "accepted" in fields(result) or hasattr(result, "accepted")
        assert "confidence" in fields(result) or hasattr(result, "confidence")
        assert "method" in fields(result) or hasattr(result, "method")
        assert isinstance(result.p_value, float)
        assert isinstance(result.confidence, float)
        # Note: accepted could be True or False depending on the VQE result,
        # just verify it's a bool
        assert isinstance(result.accepted, bool)

    def test_test_method_runs(self):
        """test method runs and returns TestResult."""
        from qsl.ai.hypotheses import HypothesisTester, TestResult

        tester = HypothesisTester("energy < 0.0")
        result = tester.test()
        assert isinstance(result, TestResult)
        assert result.hypothesis == "energy < 0.0"
        assert isinstance(result.method, str)
        assert len(result.method) > 0


# ==============================================================================
# TestDiscoveryPipeline
# ==============================================================================

class TestDiscoveryPipeline:

    def test_initializes_with_hypotheses_list(self):
        """test initializes with hypotheses list."""
        from qsl.ai.discovery import DiscoveryPipeline

        hypotheses = [
            "energy < -1.0",
            "energy > -10.0",
            "energy < -100.0",
        ]
        pipeline = DiscoveryPipeline(hypotheses)
        assert pipeline.hypotheses == hypotheses
        assert pipeline.batch_size == 10

    def test_run_returns_discovery_report(self):
        """run returns DiscoveryReport with correct count."""
        from qsl.ai.discovery import DiscoveryPipeline, DiscoveryReport

        hypotheses = ["energy < -1.0", "energy > -200.0"]
        pipeline = DiscoveryPipeline(hypotheses)
        report = pipeline.run()
        assert isinstance(report, DiscoveryReport)
        assert report.hypotheses_tested == 2

    def test_discovery_report_fields(self):
        """DiscoveryReport has accepted/rejected/confidence_scores/raw_results."""
        from qsl.ai.discovery import DiscoveryPipeline, DiscoveryReport

        hypotheses = ["energy < -1.0", "energy > -0.1", "energy < -200.0"]
        pipeline = DiscoveryPipeline(hypotheses)
        report = pipeline.run()
        assert isinstance(report, DiscoveryReport)
        assert isinstance(report.accepted, list)
        assert isinstance(report.rejected, list)
        assert isinstance(report.confidence_scores, list)
        assert isinstance(report.raw_results, list)
        assert len(report.confidence_scores) == 3
        assert len(report.raw_results) == 3
        assert len(report.accepted) + len(report.rejected) == 3


# ==============================================================================
# TestResultExplainer
# ==============================================================================

class TestResultExplainer:

    def test_initializes(self):
        """test initializes."""
        from qsl.ai.explainer import ResultExplainer

        explainer = ResultExplainer()
        assert explainer.model == "gpt-4"

    def test_explain_grover_result_like_object(self):
        """explain handles GroverResult-like object."""
        from qsl.ai.explainer import ResultExplainer, Explanation

        # Create a mock object with summary() method (like GroverResult)
        class MockGroverResult:
            def summary(self):
                return "Found 2 solutions in 1 iteration"

        explainer = ResultExplainer()
        result = explainer.explain(MockGroverResult(), context="SAT search")
        assert isinstance(result, Explanation)
        assert len(result.summary) > 0
        assert len(result.details) > 0
        assert isinstance(result.confidence, float)
        assert isinstance(result.uncertainty_sources, list)
        assert len(result.raw_interpretation) > 0

    def test_explain_dict_result(self):
        """explain handles dict result."""
        from qsl.ai.explainer import ResultExplainer, Explanation

        explainer = ResultExplainer()
        result = explainer.explain(
            {"energy": -1.137, "solutions": ["01", "10"]},
            context="VQE energy",
        )
        assert isinstance(result, Explanation)
        assert isinstance(result.confidence, float)
        assert isinstance(result.uncertainty_sources, list)

    def test_explain_none(self):
        """explain handles None."""
        from qsl.ai.explainer import ResultExplainer, Explanation

        explainer = ResultExplainer()
        result = explainer.explain(None)
        assert isinstance(result, Explanation)
        assert "No result" in result.details or "No result" in result.raw_interpretation

    def test_explanation_dataclass_fields(self):
        """Explanation dataclass fields are correct."""
        from qsl.ai.explainer import Explanation

        assert is_dataclass(Explanation)

        explanation = Explanation(
            summary="test summary",
            details="test details",
            confidence=0.95,
            uncertainty_sources=["noise"],
            raw_interpretation="raw",
        )
        assert explanation.summary == "test summary"
        assert explanation.details == "test details"
        assert explanation.confidence == 0.95
        assert explanation.uncertainty_sources == ["noise"]
        assert explanation.raw_interpretation == "raw"
