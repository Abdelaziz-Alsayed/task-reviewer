from __future__ import annotations

from pathlib import Path
from typing import Any

from database.db import init_db, save_evaluation_result
from evaluation.pipeline import EvaluationResult, evaluate_submission
from llm.base import LLMEvaluator


def run_evaluation(
    submission_path: str | Path,
    student_name: str,
    assignment: dict[str, Any],
    rubric: list[dict[str, Any]],
    llm_evaluator: LLMEvaluator,
    db_path: str | Path,
) -> EvaluationResult:
    """
    Evaluate one submission and persist the complete result.

    Workflow:
        submission
        -> evaluation pipeline
        -> SQLite persistence
    """

    submission_path = Path(submission_path)

    if not submission_path.exists():
        raise FileNotFoundError(
            f"Submission path does not exist: {submission_path}"
        )

    init_db(db_path)

    result = evaluate_submission(
        submission_path=submission_path,
        student_name=student_name,
        assignment=assignment,
        rubric=rubric,
        llm_evaluator=llm_evaluator,
    )

    save_evaluation_result(
        path=db_path,
        result=result,
        assignment_id=assignment["id"],
    )

    return result