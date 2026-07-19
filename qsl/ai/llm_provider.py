"""
LLM Provider Abstraction - Unified interface over multiple LLM backends.

Providers (all lazily imported, credentials from env vars or constructor):
    - OpenAIProvider   (OPENAI_API_KEY)
    - DeepSeekProvider (DEEPSEEK_API_KEY, OpenAI-compatible endpoint)
    - KimiProvider     (MOONSHOT_API_KEY, OpenAI-compatible endpoint)
    - QwenProvider     (DASHSCOPE_API_KEY, OpenAI-compatible endpoint)
    - OllamaProvider   (local Ollama, stdlib urllib only, no third-party deps)

Use create_provider() for auto-detection, or set_default_provider() /
get_default_provider() to configure one provider globally. When no provider
is available, callers should fall back to rule-based logic.
"""

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name, e.g. 'openai', 'deepseek', 'ollama'."""

    @abstractmethod
    def complete(self, prompt: str, system: str = None,
                 temperature: float = 0.2, max_tokens: int = 1024) -> str:
        """Synchronously return the completion text for a prompt."""

    @abstractmethod
    def available(self) -> bool:
        """Check credentials / service reachability. Must not raise."""


class OpenAIProvider(LLMProvider):
    """OpenAI chat completions (openai>=1.0 SDK, lazily imported).

    Also serves as the base class for OpenAI-compatible endpoints
    (DeepSeek / Kimi / Qwen) via the ENV_KEY / DEFAULT_BASE_URL /
    PROVIDER_NAME class attributes.
    """

    ENV_KEY = "OPENAI_API_KEY"
    DEFAULT_BASE_URL: Optional[str] = None
    PROVIDER_NAME = "openai"

    def __init__(self, model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get(self.ENV_KEY)
        self.base_url = base_url if base_url is not None else self.DEFAULT_BASE_URL
        self._client = None

    @property
    def name(self) -> str:
        return self.PROVIDER_NAME

    def available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def complete(self, prompt: str, system: str = None,
                 temperature: float = 0.2, max_tokens: int = 1024) -> str:
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek (deepseek-chat / deepseek-reasoner), OpenAI-compatible."""

    ENV_KEY = "DEEPSEEK_API_KEY"
    DEFAULT_BASE_URL = "https://api.deepseek.com"
    PROVIDER_NAME = "deepseek"

    def __init__(self, model: str = "deepseek-chat", **kwargs):
        super().__init__(model=model, **kwargs)


class KimiProvider(OpenAIProvider):
    """Moonshot Kimi, OpenAI-compatible."""

    ENV_KEY = "MOONSHOT_API_KEY"
    DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
    PROVIDER_NAME = "kimi"

    def __init__(self, model: str = "moonshot-v1-8k", **kwargs):
        super().__init__(model=model, **kwargs)


class QwenProvider(OpenAIProvider):
    """Alibaba Tongyi Qwen (DashScope compatible-mode), OpenAI-compatible."""

    ENV_KEY = "DASHSCOPE_API_KEY"
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    PROVIDER_NAME = "qwen"

    def __init__(self, model: str = "qwen-turbo", **kwargs):
        super().__init__(model=model, **kwargs)


class OllamaProvider(LLMProvider):
    """Local Ollama server, stdlib urllib only (no third-party deps)."""

    def __init__(self, model: str = "qwen2.5:7b",
                 host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    @property
    def name(self) -> str:
        return "ollama"

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=1) as resp:
                return resp.status == 200
        except Exception:
            return False

    def complete(self, prompt: str, system: str = None,
                 temperature: float = 0.2, max_tokens: int = 1024) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        req = urllib.request.Request(
            self.host + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("message", {}).get("content", "")


_PROVIDER_CLASSES = {
    "openai": OpenAIProvider,
    "deepseek": DeepSeekProvider,
    "kimi": KimiProvider,
    "moonshot": KimiProvider,
    "qwen": QwenProvider,
    "dashscope": QwenProvider,
    "ollama": OllamaProvider,
}


def create_provider(name: str = None, **kwargs) -> Optional[LLMProvider]:
    """Create an LLM provider.

    When name is None, auto-detect by priority:
        QSL_LLM env var > DeepSeek key present > Kimi key present
        > OpenAI key present > Qwen (DashScope) key present
        > reachable local Ollama > None (caller uses rule-based fallback).
    """
    if name:
        cls = _PROVIDER_CLASSES.get(name.lower())
        if cls is None:
            raise ValueError(f"Unknown LLM provider: {name!r}")
        return cls(**kwargs)

    env_name = os.environ.get("QSL_LLM")
    if env_name:
        cls = _PROVIDER_CLASSES.get(env_name.lower())
        if cls is None:
            raise ValueError(f"Unknown LLM provider in QSL_LLM: {env_name!r}")
        return cls(**kwargs)

    if os.environ.get("DEEPSEEK_API_KEY"):
        return DeepSeekProvider(**kwargs)
    if os.environ.get("MOONSHOT_API_KEY"):
        return KimiProvider(**kwargs)
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIProvider(**kwargs)
    if os.environ.get("DASHSCOPE_API_KEY"):
        return QwenProvider(**kwargs)

    ollama = OllamaProvider(**kwargs)
    if ollama.available():
        return ollama
    return None


@dataclass
class LLMConfig:
    """LLM configuration: provider / model / api_key / base_url."""

    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Read config from QSL_LLM and QSL_LLM_MODEL env vars."""
        return cls(
            provider=os.environ.get("QSL_LLM"),
            model=os.environ.get("QSL_LLM_MODEL"),
        )

    def create(self) -> Optional[LLMProvider]:
        """Instantiate the configured provider (auto-detect when unset)."""
        kwargs = {}
        if self.model:
            kwargs["model"] = self.model
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return create_provider(self.provider, **kwargs)


_UNSET = object()
_default_provider = _UNSET


def set_default_provider(provider: Optional[LLMProvider]) -> None:
    """Set the global default provider (None forces rule-based fallback)."""
    global _default_provider
    _default_provider = provider


def get_default_provider() -> Optional[LLMProvider]:
    """Return the global default provider, auto-detecting on first use."""
    global _default_provider
    if _default_provider is _UNSET:
        _default_provider = create_provider()
    return _default_provider
