from evaluation.evidence import build_criterion_evidence
from evaluation.evidence_selector import select_extracted_evidence
from evaluation.deterministic import CheckResult


def test_criterion_evidence_contains_only_relevant_checks():
    criterion = {
        "id": "randomized_search",
        "name": "RandomizedSearchCV",
        "max_score": 4,
        "checks": [
            {
                "id": "uses_randomized_search",
                "description": "Uses RandomizedSearchCV correctly.",
                "points": 1,
                "checks": ["randomized_search"],
            },
            {
                "id": "search_cv",
                "description": "Uses CV during randomized search.",
                "points": 1,
                "checks": ["randomized_search_cv"],
            },
        ],
    }

    deterministic_results = {
        "randomized_search": CheckResult(
            check_id="randomized_search",
            passed=True,
            evidence="RandomizedSearchCV detected.",
            confidence="high",
        ),
        "randomized_search_cv": CheckResult(
            check_id="randomized_search_cv",
            passed=True,
            evidence="RandomizedSearchCV uses cv=5.",
            confidence="high",
        ),
        "mae": CheckResult(
            check_id="mae",
            passed=True,
            evidence="MAE detected.",
            confidence="high",
        ),
    }

    extracted_content = {
        "files": [
            {
                "path": "submission.py",
                "type": "python",
                "text": """
from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(
    model,
    param_distributions=params,
    cv=5,
    n_iter=50
)

search.fit(X_train, y_train)

print(search.best_params_)
""",
            }
        ]
    }

    evidence = build_criterion_evidence(
        criterion=criterion,
        deterministic_results=deterministic_results,
        extracted_content=extracted_content,
    )

    check_ids = {
        item["check_id"]
        for item in evidence["deterministic_evidence"]
    }

    assert check_ids == {
        "randomized_search",
        "randomized_search_cv",
    }

    assert "submission_evidence" in evidence
    assert "RandomizedSearchCV" in str(evidence["submission_evidence"])


def test_submission_evidence_is_smaller_than_full_submission():
    criterion = {
        "id": "evaluation",
        "name": "Evaluation",
        "max_score": 4,
        "checks": [
            {
                "id": "mae",
                "description": "Calculates MAE.",
                "points": 1,
                "checks": ["mae"],
            }
        ],
    }

    extracted_content = {
        "files": [
            {
                "path": "submission.py",
                "type": "python",
                "text": (
                    "print('irrelevant section')\n" * 100
                    + "\nfrom sklearn.metrics import mean_absolute_error\n"
                    + "mae = mean_absolute_error(y_test, y_pred)\n"
                ),
            }
        ]
    }

    selected = select_extracted_evidence(
        criterion=criterion,
        extracted_content=extracted_content,
    )

    full_text = extracted_content["files"][0]["text"]

    selected_text = str(selected)

    assert len(selected_text) < len(full_text)
    assert "mean_absolute_error" in selected_text


def test_build_evidence_package_contains_criterion_evidence():
    from evaluation.evidence import build_evidence_package
    from evaluation.deterministic import CheckResult

    rubric = [
        {
            "id": "evaluation",
            "name": "Evaluation",
            "max_score": 4,
            "checks": [
                {
                    "id": "mae",
                    "description": "Calculates MAE.",
                    "points": 1,
                    "checks": ["mae"],
                }
            ],
        }
    ]

    deterministic_results = {
        "mae": CheckResult(
            check_id="mae",
            passed=True,
            evidence="mean_absolute_error detected.",
            confidence="high",
        )
    }

    extracted_content = {
        "files": [
            {
                "path": "submission.py",
                "type": "python",
                "text": (
                    "from sklearn.metrics import "
                    "mean_absolute_error\n"
                    "mae = mean_absolute_error(y_test, y_pred)"
                ),
            }
        ]
    }

    package = build_evidence_package(
        student_name="Test Student",
        assignment={
            "id": "week05_regression",
            "name": "Week 05 - Regression",
            "total_marks": 25,
        },
        rubric=rubric,
        extracted_content=extracted_content,
        deterministic_results=deterministic_results,
    )

    assert "criterion_evidence" in package
    assert len(package["criterion_evidence"]) == 1
    assert (
        package["criterion_evidence"][0]["criterion"]["id"]
        == "evaluation"
    )

    assert "extracted_content" not in package