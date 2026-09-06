from llm.base import LLMEvaluator
from llm.gemini import GeminiEvaluator
from llm.mock import MockLLMEvaluator


def create_llm_evaluator(
    provider: str = "mock",
) -> LLMEvaluator:
    provider = provider.lower().strip()

    if provider == "mock":
        return MockLLMEvaluator()

    if provider == "gemini":
        return GeminiEvaluator()

    raise ValueError(
        f"Unsupported LLM provider: {provider}"
    )