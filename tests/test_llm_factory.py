import pytest

from llm.factory import create_llm_evaluator
from llm.gemini import GeminiEvaluator
from llm.mock import MockLLMEvaluator


def test_factory_creates_mock_evaluator():
    evaluator = create_llm_evaluator("mock")

    assert isinstance(evaluator, MockLLMEvaluator)


def test_factory_is_case_insensitive():
    evaluator = create_llm_evaluator("MOCK")

    assert isinstance(evaluator, MockLLMEvaluator)


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError):
        create_llm_evaluator("unknown")


def test_gemini_factory_requires_api_key(monkeypatch):
    monkeypatch.setattr(
        "llm.gemini.GEMINI_API_KEY",
        None,
    )

    with pytest.raises(ValueError):
        GeminiEvaluator()