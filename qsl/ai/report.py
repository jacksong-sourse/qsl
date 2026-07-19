"""
结构化中文报告 — 把量子任务执行与验证结果渲染为 Markdown。

AgentReport 汇总一次任务的完整信息: 任务、算法选择理由、电路、
结果摘要、验证状态与决策链, 可渲染为中文 Markdown 报告并保存。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from .verifier import VerificationResult


@dataclass
class AgentReport:
    """一次量子任务的结构化报告。"""

    task: str
    algorithm: str
    algorithm_reason: str = ""
    backend: str = "simulator"
    circuit_text: str = ""
    result_summary: str = ""
    verification: Optional[VerificationResult] = None
    iterations: int = 1
    decision_chain: List[dict] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # ----------------------------------------------------------------
    # 渲染
    # ----------------------------------------------------------------

    def to_markdown(self) -> str:
        """渲染为中文结构化 Markdown 报告。"""
        lines: List[str] = []
        lines.append("# 任务报告")
        lines.append("")
        lines.append(f"- 生成时间: {self.timestamp}")
        lines.append(f"- 后端: {self.backend}")
        lines.append(f"- 迭代轮数: {self.iterations}")
        lines.append("")

        lines.append("## 任务")
        lines.append("")
        lines.append(self.task or "(空)")
        lines.append("")

        lines.append("## 算法选择")
        lines.append("")
        lines.append(f"- 算法: **{self.algorithm}**")
        if self.algorithm_reason:
            lines.append(f"- 选择理由: {self.algorithm_reason}")
        lines.append("")

        lines.append("## 电路")
        lines.append("")
        if self.circuit_text:
            lines.append("```text")
            lines.append(self.circuit_text.rstrip("\n"))
            lines.append("```")
        else:
            lines.append("(无电路描述)")
        lines.append("")

        lines.append("## 结果")
        lines.append("")
        lines.append(self.result_summary or "(无结果)")
        lines.append("")

        lines.append("## 验证状态")
        lines.append("")
        if self.verification is None:
            lines.append("- 状态: 未执行验证")
        else:
            icon = "✅" if self.verification.passed else "❌"
            status = "通过" if self.verification.passed else "未通过"
            lines.append(f"- 状态: {icon} {status}")
            lines.append(f"- 说明: {self.verification.message}")
            if self.verification.details:
                lines.append("- 细节:")
                for key, value in self.verification.details.items():
                    if key == "clause_results":
                        continue
                    lines.append(f"  - {key}: {value}")
        lines.append("")

        lines.append("## 决策链")
        lines.append("")
        if self.decision_chain:
            lines.append("| 轮次 | 动作 | 结果 |")
            lines.append("| --- | --- | --- |")
            for i, step in enumerate(self.decision_chain):
                rnd = step.get("round", step.get("轮次", i + 1))
                action = step.get("action", step.get("动作", ""))
                outcome = step.get("outcome", step.get("结果", ""))
                lines.append(f"| {rnd} | {action} | {outcome} |")
        else:
            lines.append("(无决策链记录)")
        lines.append("")

        return "\n".join(lines)

    def save(self, path: Union[str, Path]) -> Path:
        """把 Markdown 报告写入文件 (utf-8)。返回写入路径。"""
        path = Path(path)
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path

    # ----------------------------------------------------------------
    # 构造
    # ----------------------------------------------------------------

    @classmethod
    def from_agent_result(cls,
                          agent_result,
                          verification: Optional[VerificationResult] = None,
                          decision_chain: Optional[List[dict]] = None,
                          algorithm_reason: str = "") -> "AgentReport":
        """
        从 QuantumAgent 的 AgentResult 构造报告。

        参数:
            agent_result: qsl.ai.agent.AgentResult 实例
            verification: 可选的验证结果
            decision_chain: 可选决策链 ([{round, action, outcome}, ...])
            algorithm_reason: 算法选择理由
        """
        data = getattr(agent_result, "data", None) or {}
        if decision_chain is None:
            outcome = "成功" if getattr(agent_result, "success", False) else "失败"
            decision_chain = [{
                "round": getattr(agent_result, "iterations", 1),
                "action": f"执行 {getattr(agent_result, 'algorithm_used', '?')}",
                "outcome": outcome,
            }]
        return cls(
            task=getattr(agent_result, "task", ""),
            algorithm=getattr(agent_result, "algorithm_used", ""),
            algorithm_reason=algorithm_reason,
            backend=getattr(agent_result, "backend_used", "simulator"),
            circuit_text=data.get("circuit_text", ""),
            result_summary=getattr(agent_result, "result_summary", ""),
            verification=verification,
            iterations=getattr(agent_result, "iterations", 1),
            decision_chain=decision_chain,
        )
