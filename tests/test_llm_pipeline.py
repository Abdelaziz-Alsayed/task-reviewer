from evaluation.evidence import build_evidence_package
from evaluation.deterministic import CheckResult
from llm.factory import create_llm_evaluator


def test_mock_llm_evaluates_evidence_package():
    rubric = [
        {
            "id": "evaluation",
            "name": "Evaluation",
            "max_score": 4,
            "checks": [
                {
                    "id": "mae",
                    "description": "Calculates MAE.",
                    "points": 1,
                    "checks": ["mae"],
                }
            ],
        }
    ]

    deterministic_results = {
        "mae": CheckResult(
            check_id="mae",
            passed=True,
            evidence="MAE detected.",
            confidence="high",
        )
    }

    extracted_content = {
        "files": [
            {
                "path": "submission.py",
                "type": "python",
                "text": "mae = mean_absolute_error(y_test, y_pred)",
            }
        ]
    }

    package = build_evidence_package(
        student_name="Test Student",
        assignment={
            "id": "week05_regression",
            "name": "Week 05 - Regression",
            "total_marks": 25,
        },
        rubric=rubric,
        extracted_content=extracted_content,
        deterministic_results=deterministic_results,
    )

    evaluator = create_llm_evaluator("mock")

    result = evaluator.evaluate(package)

    assert result.model == "mock"
    assert len(result.criterion_results) == 1
    assert result.criterion_results[0].criterion_id == "evaluation"
    assert result.criterion_results[0].score == 4