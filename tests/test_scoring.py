import pytest

from evaluation.scoring import (
    calculate_ai_suggested_score,
    validate_criterion_results,
)
from llm.schemas import CriterionEvaluation


def make_result(
    criterion_id: str,
    score: float,
    max_score: float,
) -> CriterionEvaluation:
    return CriterionEvaluation(
        criterion_id=criterion_id,
        criterion_name=criterion_id,
        score=score,
        max_score=max_score,
        evidence=["Test evidence."],
        missing_requirements=[],
        reasoning="Test reasoning.",
        confidence=0.9,
    )


@pytest.fixture
def rubric():
    return [
        {
            "id": "data_understanding",
            "name": "Data Understanding",
            "max_score": 3,
        },
        {
            "id": "evaluation",
            "name": "Evaluation",
            "max_score": 4,
        },
        {
            "id": "randomized_search",
            "name": "RandomizedSearchCV",
            "max_score": 4,
        },
    ]


def test_calculate_total_score(rubric):
    results = [
        make_result(
            "data_understanding",
            2,
            3,
        ),
        make_result(
            "evaluation",
            3,
            4,
        ),
        make_result(
            "randomized_search",
            4,
            4,
        ),
    ]

    result = calculate_ai_suggested_score(
        results,
        rubric,
    )

    assert result.total_score == 9
    assert result.total_marks == 11


def test_rejects_duplicate_criteria(rubric):
    results = [
        make_result(
            "data_understanding",
            2,
            3,
        ),
        make_result(
            "data_understanding",
            3,
            3,
        ),
        make_result(
            "evaluation",
            3,
            4,
        ),
        make_result(
            "randomized_search",
            4,
            4,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate criterion",
    ):
        validate_criterion_results(
            results,
            rubric,
        )


def test_rejects_unknown_criterion(rubric):
    results = [
        make_result(
            "data_understanding",
            2,
            3,
        ),
        make_result(
            "evaluation",
            3,
            4,
        ),
        make_result(
            "randomized_search",
            4,
            4,
        ),
        make_result(
            "unknown",
            1,
            1,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="unknown criterion",
    ):
        validate_criterion_results(
            results,
            rubric,
        )


def test_rejects_missing_criterion(rubric):
    results = [
        make_result(
            "data_understanding",
            2,
            3,
        ),
        make_result(
            "evaluation",
            3,
            4,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="did not return",
    ):
        validate_criterion_results(
            results,
            rubric,
        )


def test_rejects_wrong_max_score(rubric):
    results = [
        make_result(
            "data_understanding",
            2,
            99,
        ),
        make_result(
            "evaluation",
            3,
            4,
        ),
        make_result(
            "randomized_search",
            4,
            4,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Incorrect max_score",
    ):
        validate_criterion_results(
            results,
            rubric,
        )


def test_rejects_score_above_rubric_max(rubric):
    results = [
        make_result("data_understanding", 3, 3),
        make_result("evaluation", 3, 4),
        make_result("randomized_search", 4, 4),
    ]

    # Mutate after Pydantic validation to simulate malformed external data.
    results[0].score = 4

    with pytest.raises(
        ValueError,
        match="Score exceeds maximum",
    ):
        validate_criterion_results(results, rubric)