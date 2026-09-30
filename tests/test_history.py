from pathlib import Path

from database.db import init_db, save_evaluation_result
from evaluation.history import (
    get_evaluation_details,
    get_evaluation_history,
)
from evaluation.pipeline import EvaluationResult
from evaluation.scoring import AISuggestedScore
from llm.schemas import CriterionEvaluation, LLMEvaluation


def build_test_result() -> EvaluationResult:
    criterion_results = [
        CriterionEvaluation(
            criterion_id="data_understanding",
            criterion_name="Data Understanding",
            score=3,
            max_score=3,
            evidence=["Dataset was loaded and explored."],
            missing_requirements=[],
            reasoning="The criterion requirements are supported.",
            confidence=0.95,
        ),
        CriterionEvaluation(
            criterion_id="evaluation",
            criterion_name="Evaluation",
            score=3,
            max_score=4,
            evidence=["MAE and RMSE were calculated."],
            missing_requirements=[
                "R2 interpretation is missing."
            ],
            reasoning="Most evaluation requirements are present.",
            confidence=0.8,
        ),
    ]

    llm_evaluation = LLMEvaluation(
        criterion_results=criterion_results,
        model="mock",
        overall_notes=[],
    )

    suggested_score = AISuggestedScore(
        total_score=6,
        total_marks=7,
        criterion_scores=criterion_results,
    )

    return EvaluationResult(
        student_name="History Student",
        submission_path="/tmp/history_submission",
        extracted_content={},
        deterministic_results={},
        evidence_package={},
        ai_evaluation=llm_evaluation,
        ai_suggested_score=suggested_score,
    )


def test_get_evaluation_history(tmp_path: Path):
    db_path = tmp_path / "history.db"

    init_db(db_path)

    result = build_test_result()

    save_evaluation_result(
        path=db_path,
        result=result,
        assignment_id="week05_regression",
    )

    history = get_evaluation_history(db_path)

    assert len(history) == 1

    evaluation = history[0]

    assert evaluation["name"] == "History Student"
    assert evaluation["assignment_id"] == "week05_regression"
    assert evaluation["ai_score"] == 6
    assert evaluation["max_score"] == 7
    assert evaluation["status"] == "evaluated"


def test_get_evaluation_details(tmp_path: Path):
    db_path = tmp_path / "details.db"

    init_db(db_path)

    result = build_test_result()

    evaluation_id = save_evaluation_result(
        path=db_path,
        result=result,
        assignment_id="week05_regression",
    )

    evaluation, criteria = get_evaluation_details(
        db_path,
        evaluation_id,
    )

    assert evaluation is not None

    assert evaluation["name"] == "History Student"
    assert evaluation["assignment_id"] == "week05_regression"

    assert len(criteria) == 2

    assert criteria[0]["criterion"] == "data_understanding"
    assert criteria[0]["ai_score"] == 3

    assert criteria[1]["criterion"] == "evaluation"
    assert criteria[1]["ai_score"] == 3