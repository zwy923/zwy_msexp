"""Adapter registry helpers for LLM inference."""

from thesis_exp.llm.adapters import LocalMockAdapter, OpenAICompatibleChatAdapter
from thesis_exp.llm.base import BaseLLMAdapter

ADAPTER_REGISTRY: dict[str, type[BaseLLMAdapter]] = {
    "mock": LocalMockAdapter,
    "openai_compatible": OpenAICompatibleChatAdapter,
}


def create_adapter(provider_name: str) -> BaseLLMAdapter:
    """Create an adapter by provider name."""

    adapter_class = ADAPTER_REGISTRY.get(provider_name)
    if adapter_class is None:
        raise KeyError(f"Unknown provider adapter: {provider_name}")
    return adapter_class()


def list_adapter_types() -> list[str]:
    """Return registered adapter names."""

    return sorted(ADAPTER_REGISTRY)
