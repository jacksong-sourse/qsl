"""
测试 LLM 抽象层 (qsl.ai.llm_provider) 与 QuantumAgent 的规则回退。

所有测试均可在无网络、无 API key 的环境下运行：
autouse fixture 会清除 LLM 相关环境变量，并阻断 urllib 的 HTTP 请求。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import pytest


@pytest.fixture(autouse=True)
def _no_llm_env(monkeypatch):
    """清除 LLM 环境变量并阻断 Ollama 探测的 HTTP 请求。"""
    for var in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "MOONSHOT_API_KEY",
                "DASHSCOPE_API_KEY", "QSL_LLM", "QSL_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)

    def _blocked(*args, **kwargs):
        raise OSError("network blocked in tests")

    monkeypatch.setattr("urllib.request.urlopen", _blocked)
    yield


# ==============================================================================
# TestCreateProvider
# ==============================================================================

class TestCreateProvider:

    def test_returns_none_without_keys(self):
        """create_provider 在无 key 且 Ollama 不可达时返回 None。"""
        from qsl.ai.llm_provider import create_provider
        assert create_provider() is None

    def test_qsl_llm_ollama_unavailable(self, monkeypatch):
        """QSL_LLM=ollama 且无本地服务: available()==False, 不抛异常。"""
        from qsl.ai.llm_provider import create_provider, OllamaProvider
        monkeypatch.setenv("QSL_LLM", "ollama")
        provider = create_provider()
        assert isinstance(provider, OllamaProvider)
        assert provider.available() is False

    def test_priority_deepseek_over_openai(self, monkeypatch):
        """自动探测优先级: DeepSeek key > OpenAI key。"""
        from qsl.ai.llm_provider import create_provider, DeepSeekProvider
        monkeypatch.setenv("DEEPSEEK_API_KEY", "dk-test")
        monkeypatch.setenv("OPENAI_API_KEY", "ok-test")
        provider = create_provider()
        assert isinstance(provider, DeepSeekProvider)
        assert provider.api_key == "dk-test"
        assert provider.name == "deepseek"

    def test_qsl_llm_env_overrides_keys(self, monkeypatch):
        """QSL_LLM 指定的 provider 优先于 key 自动探测。"""
        from qsl.ai.llm_provider import create_provider, KimiProvider
        monkeypatch.setenv("QSL_LLM", "kimi")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "dk-test")
        provider = create_provider()
        assert isinstance(provider, KimiProvider)
        assert provider.name == "kimi"

    def test_explicit_name_with_kwargs(self):
        from qsl.ai.llm_provider import create_provider, OpenAIProvider
        provider = create_provider("openai", api_key="sk-test", model="gpt-4o")
        assert isinstance(provider, OpenAIProvider)
        assert provider.api_key == "sk-test"
        assert provider.model == "gpt-4o"
        assert provider.name == "openai"

    def test_unknown_name_raises(self):
        from qsl.ai.llm_provider import create_provider
        with pytest.raises(ValueError):
            create_provider("nonexistent-llm")

    def test_available_false_without_key(self):
        """无凭据时 available() 为 False（不抛异常）。"""
        from qsl.ai.llm_provider import DeepSeekProvider, QwenProvider
        assert DeepSeekProvider().available() is False
        assert QwenProvider().available() is False


# ==============================================================================
# TestLLMConfig
# ==============================================================================

class TestLLMConfig:

    def test_from_env(self, monkeypatch):
        from qsl.ai.llm_provider import LLMConfig
        monkeypatch.setenv("QSL_LLM", "deepseek")
        monkeypatch.setenv("QSL_LLM_MODEL", "deepseek-reasoner")
        cfg = LLMConfig.from_env()
        assert cfg.provider == "deepseek"
        assert cfg.model == "deepseek-reasoner"

    def test_from_env_defaults(self):
        from qsl.ai.llm_provider import LLMConfig
        cfg = LLMConfig.from_env()
        assert cfg.provider is None
        assert cfg.model is None

    def test_config_create(self, monkeypatch):
        from qsl.ai.llm_provider import LLMConfig, OllamaProvider
        monkeypatch.setenv("QSL_LLM", "ollama")
        cfg = LLMConfig.from_env()
        provider = cfg.create()
        assert isinstance(provider, OllamaProvider)

    def test_default_provider_singleton(self):
        from qsl.ai import llm_provider as lp

        class Dummy(lp.LLMProvider):
            @property
            def name(self):
                return "dummy"

            def available(self):
                return True

            def complete(self, prompt, system=None, temperature=0.2,
                         max_tokens=1024):
                return "ok"

        lp.set_default_provider(Dummy())
        try:
            assert lp.get_default_provider().name == "dummy"
        finally:
            lp.set_default_provider(None)
        assert lp.get_default_provider() is None


# ==============================================================================
# TestOllamaProvider (mock HTTP)
# ==============================================================================

class TestOllamaProvider:

    def test_complete_with_mocked_urlopen(self, monkeypatch):
        """monkeypatch urlopen 返回假 JSON, 验证 urllib 调用与解析。"""
        from qsl.ai.llm_provider import OllamaProvider

        captured = {}

        class FakeResponse:
            status = 200

            def __init__(self, payload):
                self._payload = payload

            def read(self):
                return json.dumps(self._payload).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout=None, **kwargs):
            url = req.full_url if hasattr(req, "full_url") else req
            captured["url"] = url
            captured["timeout"] = timeout
            if url.endswith("/api/chat"):
                captured["body"] = json.loads(req.data.decode("utf-8"))
                return FakeResponse({"message": {"content": "grover"}})
            return FakeResponse({"models": []})

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        provider = OllamaProvider(model="qwen2.5:7b")
        assert provider.available() is True
        assert captured["url"].endswith("/api/tags")
        assert captured["timeout"] == 1

        text = provider.complete("选择算法", system="只回算法名")
        assert text == "grover"
        assert captured["url"].endswith("/api/chat")
        assert captured["body"]["model"] == "qwen2.5:7b"
        assert captured["body"]["stream"] is False
        assert captured["body"]["messages"][0] == {
            "role": "system", "content": "只回算法名"}
        assert captured["body"]["messages"][-1] == {
            "role": "user", "content": "选择算法"}

    def test_available_false_on_connection_error(self):
        from qsl.ai.llm_provider import OllamaProvider
        assert OllamaProvider().available() is False


# ==============================================================================
# TestRuleRouting (规则路由表)
# ==============================================================================

class TestRuleRouting:

    @pytest.mark.parametrize("task,expected", [
        # shor: 分解/因数/factor/rsa/质因数/因子/factorize/crack/解密
        ("分解 15", "shor"),
        ("因数分解 21", "shor"),
        ("质因数分解 77", "shor"),
        ("factor 35", "shor"),
        ("破解 RSA 加密", "shor"),
        # qaoa: 优化/最大割/旅行商/maxcut/tsp/portfolio/调度/着色/覆盖...
        ("最大割问题", "qaoa"),
        ("旅行商 6 城市", "qaoa"),
        ("optimize portfolio allocation", "qaoa"),
        ("图着色问题", "qaoa"),
        ("schedule coloring problem", "qaoa"),
        # vqe: 基态/能量/分子/化学/哈密顿/ground state/energy/本征值...
        ("求 H2 基态能量", "vqe"),
        ("ground state of molecule", "vqe"),
        ("计算分子的哈密顿量本征值", "vqe"),
        # grover: 搜索/查找/数据库/sat/数独/sudoku/search/3-sat/可满足
        ("搜索数据库", "grover"),
        ("grover 4 比特", "grover"),
        ("3-SAT 问题", "grover"),
        ("数独求解", "grover"),
        ("求解可满足性问题", "grover"),
    ])
    def test_rule_routing(self, task, expected):
        """中英文任务句 -> 正确的算法路由。"""
        from qsl.ai.agent import QuantumAgent
        agent = QuantumAgent(task_description=task, verbose=False)
        algorithm, _ = agent._select_algorithm(None)
        assert algorithm == expected


# ==============================================================================
# TestParamExtraction (参数抽取)
# ==============================================================================

class TestParamExtraction:

    @pytest.mark.parametrize("task,algorithm,key,expected", [
        ("分解 21", "shor", "N", 21),
        ("分解 15", "shor", "N", 15),
        ("factor 35", "shor", "N", 35),
        ("factor N=21", "shor", "N", 21),
        ("最大割 6 节点", "qaoa", "n_qubits", 6),
        ("用 8 比特搜索", "grover", "n_qubits", 8),
        ("n=6 的分子能量", "vqe", "n_qubits", 6),
        ("6 qubits search", "grover", "n_qubits", 6),
        ("旅行商 20 城市", "qaoa", "n_qubits", 12),  # 钳制到 12
    ])
    def test_extract(self, task, algorithm, key, expected):
        from qsl.ai.agent import QuantumAgent
        agent = QuantumAgent(task_description=task, verbose=False)
        params = agent._parse_task_params(algorithm)
        assert params[key] == expected

    def test_defaults_when_no_numbers(self):
        """无参数时返回合理默认值。"""
        from qsl.ai.agent import QuantumAgent
        agent = QuantumAgent(task_description="分解质因数", verbose=False)
        assert agent._parse_task_params("shor")["N"] == 15

        agent = QuantumAgent(task_description="求解组合优化", verbose=False)
        assert agent._parse_task_params("qaoa")["n_qubits"] == 4

    def test_shor_skips_years_and_single_digits(self):
        """年份(>1900)和 1 位数不作为 N。"""
        from qsl.ai.agent import QuantumAgent
        agent = QuantumAgent(task_description="2024 年请分解 91", verbose=False)
        assert agent._parse_task_params("shor")["N"] == 91

    def test_shor_rejects_perfect_power(self):
        """完全幂(如 27=3^3)不作为 N, 回退默认 15。"""
        from qsl.ai.agent import QuantumAgent
        agent = QuantumAgent(task_description="factor 27", verbose=False)
        assert agent._parse_task_params("shor")["N"] == 15


# ==============================================================================
# TestSuggestClarification
# ==============================================================================

class TestSuggestClarification:

    def test_shor_without_number_asks_in_chinese(self):
        """shor 任务缺少数字时触发中文追问。"""
        from qsl.ai.agent import QuantumAgent
        agent = QuantumAgent(task_description="帮我分解一个整数", verbose=False)
        hint = agent.suggest_clarification()
        assert hint is not None
        assert "分解" in hint

    def test_no_hint_when_params_present(self):
        from qsl.ai.agent import QuantumAgent
        agent = QuantumAgent(task_description="分解 15", verbose=False)
        assert agent.suggest_clarification() is None

    def test_qaoa_without_number_asks(self):
        from qsl.ai.agent import QuantumAgent
        agent = QuantumAgent(task_description="帮我优化一个组合问题", verbose=False)
        hint = agent.suggest_clarification()
        assert hint is not None
        assert "节点" in hint or "变量" in hint


# ==============================================================================
# TestAgentRuleFallback (端到端)
# ==============================================================================

class TestAgentRuleFallback:

    def test_run_shor_end_to_end_without_llm(self):
        """无 key 环境下 run() 不崩溃, 走规则路径分解 15。"""
        from qsl.ai.agent import QuantumAgent
        agent = QuantumAgent(task_description="分解 15", verbose=False)
        result = agent.run()
        assert result.algorithm_used == "shor"
        assert result.success is True
        assert set(result.data.get("factors", [])) == {3, 5}

    def test_init_llm_wraps_default_provider(self):
        """设置了全局默认 provider 时, _init_llm 返回 langchain 兼容 wrapper。"""
        from qsl.ai import llm_provider as lp

        class Dummy(lp.LLMProvider):
            @property
            def name(self):
                return "dummy"

            def available(self):
                return True

            def complete(self, prompt, system=None, temperature=0.2,
                         max_tokens=1024):
                return "shor"

        lp.set_default_provider(Dummy())
        try:
            from qsl.ai.agent import QuantumAgent
            agent = QuantumAgent(task_description="随便一个任务", verbose=False)
            llm = agent._init_llm()
            assert llm is not None
            assert llm.invoke("选择算法").content == "shor"
        finally:
            lp.set_default_provider(None)
