from __future__ import annotations

from typing import Any


def get_criterion_check_ids(criterion: dict[str, Any]) -> list[str]:
    """
    Extract all deterministic check IDs associated with a rubric criterion.

    Example:
        randomized_search ->
        [
            "randomized_search",
            "large_search_space",
            "randomized_search_cv",
            "best_params",
        ]
    """

    check_ids = []

    for rubric_check in criterion.get("checks", []):
        check_ids.extend(rubric_check.get("checks", []))

    return check_ids


def build_evidence_package(
    student_name: str,
    assignment: dict[str, Any],
    rubric: list[dict[str, Any]],
    extracted_content: dict[str, Any],
    deterministic_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build the complete evidence package.

    This package is used internally by the evaluation pipeline.
    Individual LLM calls should normally receive criterion-specific
    evidence rather than the entire package.
    """

    return {
        "student": {
            "name": student_name,
        },
        "assignment": assignment,
        "rubric": rubric,
        "deterministic_results": deterministic_results,
        "extracted_content": extracted_content,
    }


def build_criterion_evidence(
    criterion: dict[str, Any],
    deterministic_results: list[dict[str, Any]],
    extracted_content: dict[str, Any],
) -> dict[str, Any]:
    """
    Build evidence for a single rubric criterion.

    Only deterministic results related to the criterion's configured
    checks are included.
    """

    required_check_ids = get_criterion_check_ids(criterion)

    relevant_results = [
        result
        for result in deterministic_results
        if result.get("check") in required_check_ids
    ]

    return {
        "criterion": {
            "id": criterion.get("id"),
            "name": criterion.get("name"),
            "max_score": criterion.get("max_score"),
            "checks": criterion.get("checks", []),
        },
        "deterministic_evidence": relevant_results,
        "extracted_content": extracted_content,
    }


def build_all_criterion_evidence(
    rubric: list[dict[str, Any]],
    deterministic_results: list[dict[str, Any]],
    extracted_content: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build a separate evidence package for every rubric criterion.
    """

    return [
        build_criterion_evidence(
            criterion=criterion,
            deterministic_results=deterministic_results,
            extracted_content=extracted_content,
        )
        for criterion in rubric
    ]