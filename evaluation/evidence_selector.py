from __future__ import annotations

import re
from typing import Any


CHECK_KEYWORDS: dict[str, list[str]] = {
    "data_loading": [
        "read_csv",
        "read_excel",
        "load_",
        "fetch_openml",
    ],
    "dataset_exploration": [
        ".head(",
        ".info(",
        ".describe(",
        ".shape",
        "value_counts(",
    ],
    "target_identification": [
        "target",
        "target_column",
        "saleprice",
    ],
    "xy_split": [
        "train_test_split",
        "X_train",
        "X_test",
        "y_train",
        "y_test",
    ],
    "target_analysis": [
        "target.describe",
        "y.describe",
        "target.hist",
        "y.hist",
        "target.value_counts",
        "y.value_counts",
    ],
    "missing_values": [
        "fillna(",
        "SimpleImputer",
        "isna(",
        "isnull(",
        "dropna(",
    ],
    "categorical_handling": [
        "OneHotEncoder",
        "OrdinalEncoder",
        "get_dummies(",
        "ColumnTransformer",
        "handle_unknown",
        "catboost",
    ],
    "no_preprocessing_leakage": [
        "Pipeline(",
        "make_pipeline(",
        "fit_transform(",
        "StandardScaler",
        "MinMaxScaler",
        "SimpleImputer",
    ],
    "baseline": [
        "DummyRegressor",
        "mean(y_train)",
        "y_train.mean(",
        "baseline",
    ],
    "linear_regression": [
        "LinearRegression(",
    ],
    "flexible_regression": [
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "RandomForest",
        "XGBRegressor",
        "LGBMRegressor",
        "SVR(",
        "HistGradientBoostingRegressor",
    ],
    "train_predict": [
        ".fit(",
        ".predict(",
    ],
    "mae": [
        "mean_absolute_error",
        "MAE",
    ],
    "rmse": [
        "root_mean_squared_error",
        "mean_squared_error",
        "RMSE",
    ],
    "r2": [
        "r2_score",
        "R2",
        "r²",
        "r^2",
    ],
    "metric_interpretation": [
        "lower is better",
        "higher is better",
        "best model",
        "metric",
        "MAE",
        "RMSE",
        "R2",
    ],
    "cross_validation": [
        "cross_val_score",
        "cross_validate",
        "KFold(",
        "RepeatedKFold",
        "GridSearchCV",
        "RandomizedSearchCV",
    ],
    "cv_mean": [
        "mean()",
        "mean_score",
        "cv_mean",
        "cv_scores.mean",
        "scores.mean",
    ],
    "cv_std": [
        "std()",
        "std_score",
        "cv_std",
        "scores.std",
        "cv_scores.std",
    ],
    "randomized_search": [
        "RandomizedSearchCV",
    ],
    "large_search_space": [
        "param_distributions",
        "RandomizedSearchCV",
    ],
    "randomized_search_cv": [
        "RandomizedSearchCV",
        "cv=",
    ],
    "best_params": [
        "best_params_",
        "best_score_",
        "best_estimator_",
    ],
    "actual_vs_predicted": [
        "actual",
        "pred",
        "y_test",
        "y_pred",
        "y_true",
        "scatter(",
        "plot(",
    ],
    "residual_visualization": [
        "residual",
        "scatter(",
        "hist(",
        "plot(",
    ],
    "model_comparison": [
        "compare",
        "comparison",
        "results =",
        "model_results",
        "results_df",
        "sort_values",
    ],
    "model_justification": [
        "best model",
        "final model",
        "selected model",
        "because",
        "justif",
    ],
}


def get_criterion_check_ids(criterion: dict[str, Any]) -> list[str]:
    """Return deterministic check IDs configured for a criterion."""

    check_ids: list[str] = []

    for rubric_check in criterion.get("checks", []):
        check_ids.extend(rubric_check.get("checks", []))

    return check_ids


def _get_file_content(file_data: dict[str, Any]) -> str:
    """
    Return the most useful textual representation of one extracted file.

    Notebook files prioritize code and markdown because those are
    separately available in the extractor.
    """

    file_type = file_data.get("type")

    if file_type == "notebook":
        parts = [
            file_data.get("code", ""),
            file_data.get("markdown", ""),
            file_data.get("outputs", ""),
        ]
        return "\n\n".join(part for part in parts if part)

    return file_data.get("text", "")


def _find_snippets(
    text: str,
    keywords: list[str],
    window: int = 500,
    max_snippets: int = 8,
) -> list[str]:
    """
    Find short context windows around relevant keywords.

    The selector is deliberately conservative. It does not try to
    understand the code semantically; it simply extracts useful
    evidence around recognizable terms.
    """

    snippets: list[str] = []
    lower_text = text.lower()

    for keyword in keywords:
        keyword_lower = keyword.lower()

        start = 0

        while len(snippets) < max_snippets:
            position = lower_text.find(keyword_lower, start)

            if position == -1:
                break

            snippet_start = max(0, position - window)
            snippet_end = min(
                len(text),
                position + len(keyword) + window,
            )

            snippet = text[snippet_start:snippet_end].strip()

            if snippet and snippet not in snippets:
                snippets.append(snippet)

            start = position + len(keyword)

    return snippets


def select_extracted_evidence(
    criterion: dict[str, Any],
    extracted_content: dict[str, Any],
    max_snippets_per_check: int = 8,
) -> dict[str, Any]:
    """
    Select extracted submission evidence relevant to one criterion.

    Returns evidence grouped by deterministic check ID.
    """

    selected: dict[str, list[dict[str, str]]] = {}

    files = extracted_content.get("files", [])

    check_ids = get_criterion_check_ids(criterion)

    for check_id in check_ids:
        keywords = CHECK_KEYWORDS.get(check_id, [])

        if not keywords:
            continue

        check_evidence: list[dict[str, str]] = []

        for file_data in files:
            file_type = file_data.get("type")

            if file_type == "error":
                continue

            text = _get_file_content(file_data)

            if not text:
                continue

            snippets = _find_snippets(
                text=text,
                keywords=keywords,
                max_snippets=max_snippets_per_check,
            )

            for snippet in snippets:
                check_evidence.append(
                    {
                        "file": file_data.get("path", ""),
                        "type": file_type or "",
                        "snippet": snippet,
                    }
                )

        selected[check_id] = check_evidence

    return selected