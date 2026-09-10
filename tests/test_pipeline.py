from pathlib import Path

import pytest

from evaluation.pipeline import evaluate_submission
from llm.mock import MockLLMEvaluator


@pytest.fixture
def assignment():
    return {
        "id": "week05_regression",
        "name": "Week 05 - Regression",
        "total_marks": 11,
    }


@pytest.fixture
def rubric():
    return [
        {
            "id": "data_understanding",
            "name": "Data Understanding",
            "max_score": 3,
            "checks": [],
        },
        {
            "id": "evaluation",
            "name": "Evaluation",
            "max_score": 4,
            "checks": [],
        },
        {
            "id": "randomized_search",
            "name": "RandomizedSearchCV",
            "max_score": 4,
            "checks": [],
        },
    ]


def test_pipeline_runs_end_to_end(
    tmp_path: Path,
    assignment,
    rubric,
):
    submission = tmp_path / "submission"
    submission.mkdir()

    (submission / "solution.py").write_text(
        """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("train.csv")

X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
print(mae)
"""
    )

    result = evaluate_submission(
        submission_path=submission,
        student_name="Test Student",
        assignment=assignment,
        rubric=rubric,
        llm_evaluator=MockLLMEvaluator(),
    )

    assert result.student_name == "Test Student"
    assert result.submission_path == str(submission)

    assert result.extracted_content
    assert result.deterministic_results
    assert result.evidence_package

    assert result.ai_evaluation.model == "mock"

    assert len(result.ai_evaluation.criterion_results) == 3

    assert result.ai_suggested_score.total_score == 11
    assert result.ai_suggested_score.total_marks == 11


def test_pipeline_preserves_criterion_results(
    tmp_path: Path,
    assignment,
    rubric,
):
    submission = tmp_path / "submission"
    submission.mkdir()

    (submission / "solution.py").write_text(
        """
import pandas as pd

df = pd.read_csv("train.csv")
print(df.head())
print(df.describe())
"""
    )

    result = evaluate_submission(
        submission_path=submission,
        student_name="Test Student",
        assignment=assignment,
        rubric=rubric,
        llm_evaluator=MockLLMEvaluator(),
    )

    criterion_ids = [
        criterion.criterion_id
        for criterion in result.ai_evaluation.criterion_results
    ]

    assert criterion_ids == [
        "data_understanding",
        "evaluation",
        "randomized_search",
    ]


def test_pipeline_uses_supplied_llm(
    tmp_path: Path,
    assignment,
    rubric,
):
    submission = tmp_path / "submission"
    submission.mkdir()

    (submission / "solution.py").write_text(
        "print('hello')"
    )

    mock = MockLLMEvaluator()

    result = evaluate_submission(
        submission_path=submission,
        student_name="Test Student",
        assignment=assignment,
        rubric=rubric,
        llm_evaluator=mock,
    )

    assert result.ai_evaluation.model == "mock"
    assert all(
        criterion.confidence == 0.0
        for criterion in result.ai_evaluation.criterion_results
    )