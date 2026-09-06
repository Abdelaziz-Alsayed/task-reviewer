from evaluation.evidence_selector import (
    get_criterion_check_ids,
    select_extracted_evidence,
)


def test_get_criterion_check_ids():
    criterion = {
        "id": "randomized_search",
        "checks": [
            {
                "id": "uses_randomized_search",
                "checks": ["randomized_search"],
            },
            {
                "id": "best_params",
                "checks": ["best_params"],
            },
        ],
    }

    assert get_criterion_check_ids(criterion) == [
        "randomized_search",
        "best_params",
    ]


def test_select_extracted_evidence():
    criterion = {
        "id": "randomized_search",
        "name": "RandomizedSearchCV",
        "max_score": 4,
        "checks": [
            {
                "id": "uses_randomized_search",
                "checks": ["randomized_search"],
            },
            {
                "id": "best_params",
                "checks": ["best_params"],
            },
        ],
    }

    extracted_content = {
        "files": [
            {
                "path": "solution.py",
                "type": "python",
                "text": """
from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(
    model,
    param_distributions=params,
    cv=5
)

search.fit(X_train, y_train)

print(search.best_params_)
""",
            }
        ],
        "text": "",
    }

    evidence = select_extracted_evidence(
        criterion=criterion,
        extracted_content=extracted_content,
    )

    assert "randomized_search" in evidence
    assert "best_params" in evidence

    assert len(evidence["randomized_search"]) > 0
    assert len(evidence["best_params"]) > 0

    assert any(
        "RandomizedSearchCV" in item["snippet"]
        for item in evidence["randomized_search"]
    )

    assert any(
        "best_params_" in item["snippet"]
        for item in evidence["best_params"]
    )


def test_selector_ignores_extraction_errors():
    criterion = {
        "id": "evaluation",
        "checks": [
            {
                "id": "mae",
                "checks": ["mae"],
            }
        ],
    }

    extracted_content = {
        "files": [
            {
                "path": "broken.py",
                "type": "error",
                "text": "",
                "error": "Could not parse file",
            }
        ]
    }

    evidence = select_extracted_evidence(
        criterion=criterion,
        extracted_content=extracted_content,
    )

    assert evidence["mae"] == []