"""qsl.ai.demos 演示模板与 qsl.ai.report 报告测试。"""

import pytest

from qsl.ai.demos import DEMOS, list_demos, run_demo
from qsl.ai.report import AgentReport
from qsl.ai.verifier import VerificationResult

EXPECTED_DEMOS = {
    "factor", "sat", "sudoku", "maxcut", "tsp",
    "graph_coloring", "grover", "ghz", "qrng", "bb84",
}

# 相对耗时较长的演示 (QAOA 变分优化)
SLOW_DEMOS = {"maxcut", "tsp"}


class TestDemoRegistry:
    def test_registry_has_ten_demos(self):
        assert set(DEMOS) == EXPECTED_DEMOS
        assert len(DEMOS) == 10

    def test_list_demos_returns_chinese_descriptions(self):
        items = list_demos()
        assert len(items) == 10
        names = {name for name, _ in items}
        assert names == EXPECTED_DEMOS
        for name, desc in items:
            assert isinstance(desc, str) and desc.strip(), f"{name} 缺少简介"

    def test_run_unknown_demo_raises(self):
        with pytest.raises(KeyError):
            run_demo("nonexistent")


@pytest.mark.parametrize(
    "name",
    [pytest.param(n, marks=pytest.mark.slow) if n in SLOW_DEMOS else n
     for n in sorted(EXPECTED_DEMOS)],
)
def test_demo_runs_and_verified(name):
    out = run_demo(name, verbose=False)
    assert set(out) >= {"task", "algorithm", "result", "verified",
                        "report_markdown"}
    assert out["verified"] is True, (
        f"demo {name} 验证失败:\n{out['report_markdown']}")
    md = out["report_markdown"]
    assert md.startswith("# 任务报告")
    assert "## 验证状态" in md
    assert "✅" in md


class TestAgentReport:
    def _sample_report(self):
        return AgentReport(
            task="测试任务: 分解 15",
            algorithm="shor",
            algorithm_reason="整数分解用 Shor",
            backend="simulator",
            circuit_text="QPE circuit text",
            result_summary="15 = 3 * 5",
            verification=VerificationResult(
                True, "分解正确: 15 = 3 * 5", {"product": 15}),
            iterations=2,
            decision_chain=[
                {"round": 1, "action": "选择算法", "outcome": "shor"},
                {"round": 2, "action": "执行并验证", "outcome": "成功"},
            ],
        )

    def test_markdown_contains_all_sections(self):
        md = self._sample_report().to_markdown()
        for heading in ("# 任务报告", "## 任务", "## 算法选择", "## 电路",
                        "## 结果", "## 验证状态", "## 决策链"):
            assert heading in md, f"缺少章节: {heading}"
        assert "✅" in md
        assert "| 轮次 | 动作 | 结果 |" in md
        assert "```text" in md

    def test_markdown_failed_verification_icon(self):
        report = self._sample_report()
        report.verification = VerificationResult(False, "回乘失败", {})
        md = report.to_markdown()
        assert "❌" in md

    def test_markdown_no_verification(self):
        report = self._sample_report()
        report.verification = None
        assert "未执行验证" in report.to_markdown()

    def test_save_writes_utf8_file(self, tmp_path):
        path = tmp_path / "report.md"
        returned = self._sample_report().save(path)
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert text.startswith("# 任务报告")
        assert "分解 15" in text
        assert str(returned) == str(path)

    def test_from_agent_result(self):
        from qsl.ai.agent import AgentResult

        agent_result = AgentResult(
            task="分解 15",
            success=True,
            algorithm_used="shor",
            backend_used="simulator",
            iterations=1,
            result_summary="15 = 3 * 5",
            data={"factors": [3, 5], "circuit_text": "shor circuit"},
        )
        verification = VerificationResult(True, "分解正确", {})
        report = AgentReport.from_agent_result(
            agent_result, verification=verification,
            algorithm_reason="关键词匹配")
        assert report.task == "分解 15"
        assert report.algorithm == "shor"
        assert report.backend == "simulator"
        assert report.circuit_text == "shor circuit"
        assert report.verification.passed
        assert report.decision_chain  # 默认生成一行
        md = report.to_markdown()
        assert "# 任务报告" in md
        assert "## 决策链" in md
