from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.deterministic import run_checks
from evaluation.evidence import build_evidence_package
from evaluation.scoring import AISuggestedScore, calculate_ai_suggested_score
from extraction.extractor import extract_submission
from llm.base import LLMEvaluator
from llm.schemas import LLMEvaluation


@dataclass
class EvaluationResult:
    student_name: str
    submission_path: str
    extracted_content: dict[str, Any]
    deterministic_results: dict[str, Any]
    evidence_package: dict[str, Any]
    ai_evaluation: LLMEvaluation
    ai_suggested_score: AISuggestedScore


def evaluate_submission(
    submission_path: str | Path,
    student_name: str,
    assignment: dict[str, Any],
    rubric: list[dict[str, Any]],
    llm_evaluator: LLMEvaluator,
) -> EvaluationResult:
    """
    Run the complete evaluation workflow for one student submission.

    The pipeline coordinates:
    extraction -> deterministic checks -> evidence package
    -> LLM evaluation -> score calculation.
    """

    submission_path = Path(submission_path)

    # 1. Extract the student's submission.
    extracted_content = extract_submission(submission_path)

    # 2. Run deterministic checks.
    deterministic_results = run_checks(extracted_content)

    # 3. Build the evidence package.
    evidence_package = build_evidence_package(
        student_name=student_name,
        assignment=assignment,
        rubric=rubric,
        extracted_content=extracted_content,
        deterministic_results=deterministic_results,
    )

    # 4. Evaluate the evidence using the selected LLM.
    ai_evaluation = llm_evaluator.evaluate(evidence_package)

    # 5. Calculate the AI suggested score.
    ai_suggested_score = calculate_ai_suggested_score(
        results=ai_evaluation.criterion_results,
        rubric=rubric,
    )

    return EvaluationResult(
        student_name=student_name,
        submission_path=str(submission_path),
        extracted_content=extracted_content,
        deterministic_results=deterministic_results,
        evidence_package=evidence_package,
        ai_evaluation=ai_evaluation,
        ai_suggested_score=ai_suggested_score,
    )