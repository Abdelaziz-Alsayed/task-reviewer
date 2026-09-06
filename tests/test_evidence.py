from evaluation.deterministic import CheckResult
from evaluation.evidence import (
    build_all_criterion_evidence,
    build_criterion_evidence,
    build_evidence_package,
    get_criterion_check_ids,
    normalize_check_results,
)
from llm.mock import MockLLMEvaluator


def test_get_criterion_check_ids():
    criterion = {
        "id": "evaluation",
        "name": "Evaluation",
        "max_score": 4,
        "checks": [
            {
                "id": "mae",
                "checks": ["mae"],
            },
            {
                "id": "rmse",
                "checks": ["rmse"],
            },
            {
                "id": "r2",
                "checks": ["r2"],
            },
        ],
    }

    check_ids = get_criterion_check_ids(criterion)

    assert check_ids == ["mae", "rmse", "r2"]


def test_normalize_check_results():
    results = {
        "mae": CheckResult(
            check_id="mae",
            passed=True,
            evidence="Found MAE.",
            confidence="high",
        ),
        "rmse": CheckResult(
            check_id="rmse",
            passed=False,
            evidence="RMSE not detected.",
            confidence="medium",
        ),
    }

    normalized = normalize_check_results(results)

    assert len(normalized) == 2
    assert normalized[0]["check_id"] in {"mae", "rmse"}

    mae_result = next(
        result
        for result in normalized
        if result["check_id"] == "mae"
    )

    assert mae_result["passed"] is True
    assert mae_result["evidence"] == "Found MAE."
    assert mae_result["confidence"] == "high"


def test_build_criterion_evidence_filters_checks():
    criterion = {
        "id": "evaluation",
        "name": "Evaluation",
        "max_score": 4,
        "checks": [
            {
                "id": "mae",
                "checks": ["mae"],
            },
            {
                "id": "rmse",
                "checks": ["rmse"],
            },
        ],
    }

    deterministic_results = {
        "mae": CheckResult(
            "mae",
            True,
            "Found MAE.",
            "high",
        ),
        "rmse": CheckResult(
            "rmse",
            True,
            "Found RMSE.",
            "medium",
        ),
        "r2": CheckResult(
            "r2",
            True,
            "Found R².",
            "medium",
        ),
    }

    evidence = build_criterion_evidence(
        criterion=criterion,
        deterministic_results=deterministic_results,
        extracted_content={},
    )

    assert len(evidence["deterministic_evidence"]) == 2

    returned_checks = {
        result["check_id"]
        for result in evidence["deterministic_evidence"]
    }

    assert returned_checks == {"mae", "rmse"}


def test_build_all_criterion_evidence():
    rubric = [
        {
            "id": "evaluation",
            "name": "Evaluation",
            "max_score": 2,
            "checks": [
                {
                    "id": "mae",
                    "checks": ["mae"],
                },
            ],
        },
        {
            "id": "cross_validation",
            "name": "Cross-Validation",
            "max_score": 1,
            "checks": [
                {
                    "id": "kfold",
                    "checks": ["cross_validation"],
                },
            ],
        },
    ]

    deterministic_results = {
        "mae": CheckResult(
            "mae",
            True,
            "Found MAE.",
        ),
        "cross_validation": CheckResult(
            "cross_validation",
            True,
            "Found CV.",
        ),
    }

    results = build_all_criterion_evidence(
        rubric=rubric,
        deterministic_results=deterministic_results,
        extracted_content={},
    )

    assert len(results) == 2

    assert (
        results[0]["deterministic_evidence"][0]["check_id"]
        == "mae"
    )

    assert (
        results[1]["deterministic_evidence"][0]["check_id"]
        == "cross_validation"
    )


def test_build_evidence_package():
    assignment = {
        "id": "week05_regression",
        "name": "Week 05 - Regression",
        "total_marks": 25,
    }

    package = build_evidence_package(
        student_name="Demo Student",
        assignment=assignment,
        rubric=[],
        extracted_content={
            "python": "example code"
        },
        deterministic_results={},
    )

    assert package["student"]["name"] == "Demo Student"
    assert package["assignment"]["total_marks"] == 25


def test_mock_llm():
    rubric = [
        {
            "id": "evaluation",
            "name": "Evaluation",
            "max_score": 4,
        }
    ]

    evaluator = MockLLMEvaluator()

    result = evaluator.evaluate(
        {
            "rubric": rubric,
        }
    )

    assert result.model == "mock"
    assert len(result.criterion_results) == 1
    assert result.criterion_results[0].score == 4