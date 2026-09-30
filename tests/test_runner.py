from pathlib import Path

from database.db import get_evaluation, init_db
from evaluation.runner import run_evaluation
from llm.mock import MockLLMEvaluator


def test_run_evaluation_persists_result(
    tmp_path: Path,
):
    assignment = {
        "id": "week05_regression",
        "name": "Week 05 - Regression",
        "total_marks": 11,
    }

    rubric = [
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

    submission = tmp_path / "submission"
    submission.mkdir()

    (submission / "solution.py").write_text(
        """
import pandas as pd

df = pd.read_csv("train.csv")

X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

print(df.head())
print(df.describe())
"""
    )

    db_path = tmp_path / "runner.db"

    result = run_evaluation(
        submission_path=submission,
        student_name="Runner Student",
        assignment=assignment,
        rubric=rubric,
        llm_evaluator=MockLLMEvaluator(),
        db_path=db_path,
    )

    assert result.student_name == "Runner Student"
    assert result.submission_path == str(submission)

    evaluation, criteria = get_evaluation(
        db_path,
        evaluation_id=1,
    )

    assert evaluation is not None
    assert evaluation["name"] == "Runner Student"
    assert evaluation["assignment_id"] == "week05_regression"
    assert evaluation["status"] == "evaluated"

    assert evaluation["ai_score"] == 11
    assert evaluation["max_score"] == 11

    assert len(criteria) == 3