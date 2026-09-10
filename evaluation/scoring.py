from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm.schemas import CriterionEvaluation


@dataclass
class AISuggestedScore:
    total_score: float
    total_marks: float
    criterion_scores: list[CriterionEvaluation]


def validate_criterion_results(
    results: list[CriterionEvaluation],
    rubric: list[dict[str, Any]],
) -> None:
    rubric_ids = [
        str(criterion["id"])
        for criterion in rubric
    ]

    result_ids = [
        str(result.criterion_id)
        for result in results
    ]

    if len(result_ids) != len(set(result_ids)):
        raise ValueError(
            "Duplicate criterion results detected."
        )

    unknown_ids = set(result_ids) - set(rubric_ids)

    if unknown_ids:
        raise ValueError(
            "LLM returned unknown criterion IDs: "
            + ", ".join(sorted(unknown_ids))
        )

    missing_ids = set(rubric_ids) - set(result_ids)

    if missing_ids:
        raise ValueError(
            "LLM did not return results for criteria: "
            + ", ".join(sorted(missing_ids))
        )

    rubric_max_scores = {
        str(criterion["id"]): float(
            criterion["max_score"]
        )
        for criterion in rubric
    }

    for result in results:
        expected_max = rubric_max_scores[
            str(result.criterion_id)
        ]

        if result.max_score != expected_max:
            raise ValueError(
                f"Incorrect max_score for criterion "
                f"{result.criterion_id}: "
                f"expected {expected_max}, "
                f"got {result.max_score}"
            )

        if result.score < 0:
            raise ValueError(
                f"Negative score for criterion "
                f"{result.criterion_id}"
            )

        if result.score > expected_max:
            raise ValueError(
                f"Score exceeds maximum for criterion "
                f"{result.criterion_id}"
            )


def calculate_ai_suggested_score(
    results: list[CriterionEvaluation],
    rubric: list[dict[str, Any]],
) -> AISuggestedScore:
    validate_criterion_results(
        results=results,
        rubric=rubric,
    )

    total_marks = sum(
        float(criterion["max_score"])
        for criterion in rubric
    )

    total_score = sum(
        float(result.score)
        for result in results
    )

    return AISuggestedScore(
        total_score=total_score,
        total_marks=total_marks,
        criterion_scores=results,
    )