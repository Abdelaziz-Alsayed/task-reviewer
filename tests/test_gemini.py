import pytest

from llm.gemini import GeminiEvaluator


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