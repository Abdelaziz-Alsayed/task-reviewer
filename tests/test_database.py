import json

from database.db import (
    get_evaluation,
    init_db,
    list_evaluations,
    save_evaluation_result,
)
from evaluation.pipeline import EvaluationResult
from evaluation.scoring import AISuggestedScore
from llm.schemas import CriterionEvaluation, LLMEvaluation


def build_test_result():
    criterion_results = [
        CriterionEvaluation(
            criterion_id="data_understanding",
            criterion_name="Data Understanding",
            score=3,
            max_score=3,
            evidence=[
                "Dataset loading detected.",
                "Target identified.",
            ],
            missing_requirements=[],
            reasoning="The criterion requirements were satisfied.",
            confidence=0.95,
        ),
        CriterionEvaluation(
            criterion_id="evaluation",
            criterion_name="Evaluation",
            score=3,
            max_score=4,
            evidence=[
                "MAE detected.",
                "RMSE detected.",
            ],
            missing_requirements=[
                "R² interpretation not clearly detected."
            ],
            reasoning="Most evaluation requirements were supported.",
            confidence=0.8,
        ),
    ]

    ai_evaluation = LLMEvaluation(
        criterion_results=criterion_results,
        model="mock",
        overall_notes=[],
    )

    ai_score = AISuggestedScore(
        total_score=6,
        total_marks=7,
        criterion_scores=criterion_results,
    )

    return EvaluationResult(
        student_name="Test Student",
        submission_path="/tmp/test_submission",
        extracted_content={},
        deterministic_results={},
        evidence_package={},
        ai_evaluation=ai_evaluation,
        ai_suggested_score=ai_score,
    )


def test_save_and_get_evaluation(tmp_path):
    db_path = tmp_path / "test.db"

    init_db(db_path)

    result = build_test_result()

    evaluation_id = save_evaluation_result(
        path=db_path,
        result=result,
        assignment_id="week05_regression",
    )

    assert evaluation_id > 0

    evaluation, criteria = get_evaluation(
        db_path,
        evaluation_id,
    )

    assert evaluation["name"] == "Test Student"
    assert evaluation["assignment_id"] == "week05_regression"
    assert evaluation["status"] == "evaluated"
    assert evaluation["ai_score"] == 6
    assert evaluation["max_score"] == 7
    assert evaluation["final_score"] is None

    assert len(criteria) == 2

    first = criteria[0]

    assert first["criterion"] == "data_understanding"
    assert first["ai_score"] == 3
    assert first["max_score"] == 3
    assert first["confidence"] == 0.95

    evidence = json.loads(first["evidence"])

    assert "Dataset loading detected." in evidence


def test_list_evaluations(tmp_path):
    db_path = tmp_path / "test.db"

    init_db(db_path)

    result = build_test_result()

    save_evaluation_result(
        path=db_path,
        result=result,
        assignment_id="week05_regression",
    )

    evaluations = list_evaluations(db_path)

    assert len(evaluations) == 1
    assert evaluations[0]["name"] == "Test Student"
    assert evaluations[0]["ai_score"] == 6
    assert evaluations[0]["max_score"] == 7


def test_resaving_evaluation_replaces_criterion_results(tmp_path):
    db_path = tmp_path / "test.db"

    init_db(db_path)

    result = build_test_result()

    first_id = save_evaluation_result(
        path=db_path,
        result=result,
        assignment_id="week05_regression",
    )

    second_id = save_evaluation_result(
        path=db_path,
        result=result,
        assignment_id="week05_regression",
    )

    assert second_id == first_id

    evaluation, criteria = get_evaluation(
        db_path,
        first_id,
    )

    assert evaluation["ai_score"] == 6
    assert len(criteria) == 2