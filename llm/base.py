from abc import ABC, abstractmethod

from llm.schemas import LLMEvaluation


class LLMEvaluator(ABC):
    """Provider-independent interface for assignment evaluation."""

    @abstractmethod
    def evaluate(self, evidence_package: dict) -> LLMEvaluation:
        """
        Evaluate a submission using an evidence package.

        Implementations may use Gemini, OpenAI, or a mock model.
        """
        raise NotImplementedError