import pytest

from llm.gemini import GeminiEvaluator
from pydantic import ValidationError

from llm.schemas import CriterionEvaluation


def test_parse_json():
    text = """
    {
        "criterion_id": "mae",
        "criterion_name": "MAE",
        "score": 1,
        "max_score": 1,
        "evidence": [
            "mean_absolute_error was detected."
        ],
        "missing_requirements": [],
        "reasoning": "The submission contains a valid MAE calculation.",
        "confidence": 0.95
    }
    """

    result = GeminiEvaluator._parse_json(text)

    assert result["criterion_id"] == "mae"
    assert result["score"] == 1
    assert result["confidence"] == 0.95


def test_parse_markdown_wrapped_json():
    text = """```json
{
    "criterion_id": "mae",
    "criterion_name": "MAE",
    "score": 1,
    "max_score": 1,
    "evidence": [],
    "missing_requirements": [],
    "reasoning": "Supported by evidence.",
    "confidence": 0.9
}
```"""

    result = GeminiEvaluator._parse_json(text)

    assert result["criterion_id"] == "mae"


def test_parse_invalid_json():
    with pytest.raises(ValueError):
        GeminiEvaluator._parse_json(
            "this is not valid JSON"
        )


def test_score_cannot_exceed_max_score():
    try:
        CriterionEvaluation(
            criterion_id="mae",
            criterion_name="MAE",
            score=2,
            max_score=1,
            evidence=[],
            missing_requirements=[],
            reasoning="Invalid score.",
            confidence=0.9,
        )
    except ValidationError:
        return

    raise AssertionError(
        "Expected ValidationError for score > max_score"
    )


def test_score_cannot_be_negative():
    try:
        CriterionEvaluation(
            criterion_id="mae",
            criterion_name="MAE",
            score=-1,
            max_score=1,
            evidence=[],
            missing_requirements=[],
            reasoning="Invalid score.",
            confidence=0.9,
        )
    except ValidationError:
        return

    raise AssertionError(
        "Expected ValidationError for negative score"
    )


def test_unexpected_criterion_id_is_rejected(monkeypatch):
    evaluator = object.__new__(GeminiEvaluator)

    criterion_evidence = {
        "criterion": {
            "id": "randomized_search",
            "name": "RandomizedSearchCV",
            "max_score": 4,
        }
    }

    class FakeResponse:
        text = """
        {
            "criterion_id": "mae",
            "criterion_name": "MAE",
            "score": 1,
            "max_score": 1,
            "evidence": [],
            "missing_requirements": [],
            "reasoning": "Wrong criterion.",
            "confidence": 0.9
        }
        """

    class FakeModels:
        def generate_content(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    evaluator.client = FakeClient()
    evaluator.model = "test"

    with pytest.raises(ValueError, match="unexpected criterion_id"):
        evaluator.evaluate_criterion(criterion_evidence)