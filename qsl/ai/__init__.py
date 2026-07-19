"""QSL AI Quantum Scientist - LLM-powered quantum computing automation.

LLM access is abstracted via qsl.ai.llm_provider (OpenAI / DeepSeek / Kimi /
Qwen / Ollama). Set QSL_LLM to pick a provider explicitly, or just set the
corresponding API key env var (DEEPSEEK_API_KEY, MOONSHOT_API_KEY,
OPENAI_API_KEY, DASHSCOPE_API_KEY) and let create_provider auto-detect.
"""

import os


def _create_llm(model=None, api_key=None):
    """Create an LLM client, auto-detecting OpenAI vs DeepSeek.

    Priority: DEEPSEEK_API_KEY > OPENAI_API_KEY

    DeepSeek config:
        base_url: https://api.deepseek.com
        models:   deepseek-chat (V3), deepseek-reasoner (R1)

    Returns:
        ChatOpenAI instance, or None if no API key found.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None

    # DeepSeek takes priority if configured
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_key:
        key = api_key or deepseek_key
        model = model or "deepseek-chat"
        return ChatOpenAI(
            model=model,
            api_key=key,
            base_url="https://api.deepseek.com",
        )

    # Fall back to OpenAI
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return None

    model = model or "gpt-4"
    return ChatOpenAI(model=model, api_key=key)


try:
    from .llm_provider import (
        LLMProvider,
        OpenAIProvider,
        DeepSeekProvider,
        KimiProvider,
        QwenProvider,
        OllamaProvider,
        LLMConfig,
        create_provider,
        set_default_provider,
        get_default_provider,
    )
except ImportError:
    LLMProvider = None
    OpenAIProvider = None
    DeepSeekProvider = None
    KimiProvider = None
    QwenProvider = None
    OllamaProvider = None
    LLMConfig = None
    create_provider = None
    set_default_provider = None
    get_default_provider = None

try:
    from .translator import ProblemTranslator
except ImportError:
    ProblemTranslator = None

try:
    from .agent import QuantumAgent
except ImportError:
    QuantumAgent = None

try:
    from .hypotheses import HypothesisTester
except ImportError:
    HypothesisTester = None

try:
    from .discovery import DiscoveryPipeline
except ImportError:
    DiscoveryPipeline = None

try:
    from .explainer import ResultExplainer
except ImportError:
    ResultExplainer = None

__all__ = [
    "ProblemTranslator",
    "QuantumAgent",
    "HypothesisTester",
    "DiscoveryPipeline",
    "ResultExplainer",
    "LLMProvider",
    "OpenAIProvider",
    "DeepSeekProvider",
    "KimiProvider",
    "QwenProvider",
    "OllamaProvider",
    "LLMConfig",
    "create_provider",
    "set_default_provider",
    "get_default_provider",
]
