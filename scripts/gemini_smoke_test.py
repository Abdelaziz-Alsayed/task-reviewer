from evaluation.deterministic import CheckResult
from evaluation.evidence import build_criterion_evidence
from llm.factory import create_llm_evaluator


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
            "id": "broad_search_space",
            "description": "Defines a sufficiently broad search space.",
            "points": 1,
            "checks": ["large_search_space"],
        },
        {
            "id": "search_cv",
            "description": "Uses CV during randomized search.",
            "points": 1,
            "checks": ["randomized_search_cv"],
        },
        {
            "id": "best_params",
            "description": "Reports/interprets best parameters.",
            "points": 1,
            "checks": ["best_params"],
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
    "large_search_space": CheckResult(
        check_id="large_search_space",
        passed=True,
        evidence="param_distributions with multiple parameters detected.",
        confidence="medium",
    ),
    "randomized_search_cv": CheckResult(
        check_id="randomized_search_cv",
        passed=True,
        evidence="RandomizedSearchCV contains cv=5.",
        confidence="high",
    ),
    "best_params": CheckResult(
        check_id="best_params",
        passed=True,
        evidence="best_params_ detected.",
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

param_distributions = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
}

search = RandomizedSearchCV(
    model,
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring="neg_root_mean_squared_error",
    random_state=42,
)

search.fit(X_train, y_train)

print(search.best_params_)
print(search.best_score_)
""",
        }
    ]
}


criterion_evidence = build_criterion_evidence(
    criterion=criterion,
    deterministic_results=deterministic_results,
    extracted_content=extracted_content,
)


def main():
    evaluator = create_llm_evaluator("gemini")

    result = evaluator.evaluate_criterion(
        criterion_evidence
    )

    print("\n=== Gemini Evaluation ===\n")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()