from __future__ import annotations

from dataclasses import asdict
from typing import Any

from evaluation.deterministic import CheckResult
from evaluation.evidence_selector import select_extracted_evidence


def get_criterion_check_ids(criterion: dict[str, Any]) -> list[str]:
    """
    Return the deterministic check IDs associated with a rubric criterion.

    The mapping comes directly from rubric.yaml.

    Example:
        Evaluation
        -> mae
        -> rmse
        -> r2
        -> metric_interpretation
    """

    check_ids: list[str] = []

    for rubric_check in criterion.get("checks", []):
        check_ids.extend(rubric_check.get("checks", []))

    return check_ids


def normalize_check_results(
    deterministic_results: dict[str, CheckResult]
    | list[CheckResult],
) -> list[dict[str, Any]]:
    """
    Convert deterministic CheckResult objects into JSON-friendly dictionaries.

    This creates the representation that can later be passed to an LLM.
    """

    if isinstance(deterministic_results, dict):
        results = deterministic_results.values()
    else:
        results = deterministic_results

    normalized = []

    for result in results:
        if isinstance(result, CheckResult):
            normalized.append(asdict(result))
        elif isinstance(result, dict):
            normalized.append(result)
        else:
            raise TypeError(
                f"Unsupported deterministic result type: {type(result)}"
            )

    return normalized


def build_evidence_package(
    student_name: str,
    assignment: dict[str, Any],
    rubric: list[dict[str, Any]],
    extracted_content: dict[str, Any],
    deterministic_results: dict[str, CheckResult] | list[CheckResult],
) -> dict[str, Any]:
    normalized_results = normalize_check_results(
        deterministic_results
    )

    criterion_evidence = build_all_criterion_evidence(
        rubric=rubric,
        deterministic_results=deterministic_results,
        extracted_content=extracted_content,
    )

    return {
        "student": {
            "name": student_name,
        },
        "assignment": assignment,
        "rubric": rubric,
        "deterministic_results": normalized_results,
        "criterion_evidence": criterion_evidence,
    }


def build_criterion_evidence(
    criterion: dict[str, Any],
    deterministic_results: dict[str, CheckResult]
    | list[CheckResult],
    extracted_content: dict[str, Any],
) -> dict[str, Any]:
    """
    Build evidence specifically for one rubric criterion.

    Only deterministic checks configured for that criterion
    are included.
    """

    required_check_ids = set(
        get_criterion_check_ids(criterion)
    )

    normalized_results = normalize_check_results(
        deterministic_results
    )

    relevant_results = [
        result
        for result in normalized_results
        if result.get("check_id") in required_check_ids
    ]

    return {
        "criterion": {
            "id": criterion.get("id"),
            "name": criterion.get("name"),
            "max_score": criterion.get("max_score"),
            "checks": criterion.get("checks", []),
        },
        "deterministic_evidence": relevant_results,
        "submission_evidence": select_extracted_evidence(
            criterion=criterion,
            extracted_content=extracted_content,
        ),
    }


def build_all_criterion_evidence(
    rubric: list[dict[str, Any]],
    deterministic_results: dict[str, CheckResult]
    | list[CheckResult],
    extracted_content: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build one criterion-specific evidence package for every
    rubric criterion.
    """

    return [
        build_criterion_evidence(
            criterion=criterion,
            deterministic_results=deterministic_results,
            extracted_content=extracted_content,
        )
        for criterion in rubric
    ]