from __future__ import annotations

from pathlib import Path
from typing import Any

from database.db import get_evaluation, list_evaluations


def get_evaluation_history(
    db_path: str | Path,
) -> list[dict[str, Any]]:
    """
    Return all stored evaluations in a UI-friendly format.
    """

    rows = list_evaluations(db_path)

    return [dict(row) for row in rows]


def get_evaluation_details(
    db_path: str | Path,
    evaluation_id: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """
    Return one evaluation and its criterion results.
    """

    evaluation, criteria = get_evaluation(
        db_path,
        evaluation_id,
    )

    evaluation_dict = (
        dict(evaluation)
        if evaluation is not None
        else None
    )

    criteria_dicts = [
        dict(criterion)
        for criterion in criteria
    ]

    return evaluation_dict, criteria_dicts