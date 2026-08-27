import ast
import re
from dataclasses import dataclass


@dataclass
class CheckResult:
    check_id: str
    passed: bool
    evidence: str
    confidence: str = "medium"


def _code_text(extracted: dict) -> str:
    chunks = []
    for f in extracted.get("files", []):
        if f.get("type") in {"python", "notebook"}:
            chunks.append(f.get("code", f.get("text", "")))
    return "\n".join(chunks)


def _all_text(extracted: dict) -> str:
    return extracted.get("text", "")


def _has_any(text: str, patterns: list[str]) -> bool:
    low = text.lower()
    return any(p.lower() in low for p in patterns)


def check_code(code: str, full_text: str | None = None) -> dict[str, CheckResult]:
    results = {}
    normalized = code.replace("\\", "/")

    results["data_loading"] = CheckResult("data_loading", _has_any(code, ["read_csv(", "read_excel(", "load_", "fetch_openml("]), "Found a recognizable data-loading operation." if _has_any(code, ["read_csv(", "read_excel(", "load_", "fetch_openml("]) else "No recognizable data-loading operation found.")
    results["dataset_exploration"] = CheckResult("dataset_exploration", _has_any(code, [".head(", ".info(", ".describe(", ".shape", "value_counts("]), "Found dataset exploration operations." if _has_any(code, [".head(", ".info(", ".describe(", ".shape", "value_counts("]) else "No common dataset exploration operation found.")
    results["target_identification"] = CheckResult("target_identification", bool(re.search(r"\b(target|target_column|y)\s*=", code, re.I)), "Found an explicit target assignment." if re.search(r"\b(target|target_column|y)\s*=", code, re.I) else "No explicit target assignment detected.")
    results["xy_split"] = CheckResult("xy_split", bool(re.search(r"X\s*=.*(?:drop|\[|iloc)|(?:train_test_split|X_train|X_test|y_train|y_test)", code, re.I | re.S)), "Found an X/y or train/test separation pattern." if re.search(r"X\s*=.*(?:drop|\[|iloc)|(?:train_test_split|X_train|X_test|y_train|y_test)", code, re.I | re.S) else "No clear X/y separation pattern detected.")
    results["target_analysis"] = CheckResult("target_analysis", _has_any(code, ["target.describe", "y.describe", "target.hist", "y.hist", "target.value_counts", "y.value_counts"]), "Found target analysis." if _has_any(code, ["target.describe", "y.describe", "target.hist", "y.hist", "target.value_counts", "y.value_counts"]) else "No explicit target analysis detected.")

    results["missing_values"] = CheckResult("missing_values", _has_any(code, ["fillna(", "SimpleImputer", "isna(", "isnull(", "dropna("]), "Found missing-value handling." if _has_any(code, ["fillna(", "SimpleImputer", "isna(", "isnull(", "dropna("]) else "No missing-value handling detected.")
    results["categorical_handling"] = CheckResult("categorical_handling", _has_any(code, ["OneHotEncoder", "OrdinalEncoder", "get_dummies(", "ColumnTransformer", "catboost", "handle_unknown"]), "Found categorical-feature handling." if _has_any(code, ["OneHotEncoder", "OrdinalEncoder", "get_dummies(", "ColumnTransformer", "catboost", "handle_unknown"]) else "No common categorical-feature handling detected.")

    results["baseline"] = CheckResult("baseline", _has_any(code, ["DummyRegressor", "mean(y_train)", "y_train.mean(", "baseline"]), "Found a baseline pattern." if _has_any(code, ["DummyRegressor", "mean(y_train)", "y_train.mean(", "baseline"]) else "No clear baseline pattern detected.")
    results["linear_regression"] = CheckResult("linear_regression", _has_any(code, ["LinearRegression("]), "Found LinearRegression." if _has_any(code, ["LinearRegression("]) else "LinearRegression not detected.")
    results["flexible_regression"] = CheckResult("flexible_regression", _has_any(code, ["RandomForestRegressor", "GradientBoostingRegressor", "RandomForest", "XGBRegressor", "LGBMRegressor", "SVR(", "RandomForestRegressor(", "HistGradientBoostingRegressor"]), "Found a more flexible regression model." if _has_any(code, ["RandomForestRegressor", "GradientBoostingRegressor", "RandomForest", "XGBRegressor", "LGBMRegressor", "SVR(", "HistGradientBoostingRegressor"]) else "No common flexible regression model detected.")
    results["train_predict"] = CheckResult("train_predict", bool(re.search(r"\.fit\s*\(|\.predict\s*\(", code)), "Found fit/predict workflow." if bool(re.search(r"\.fit\s*\(|\.predict\s*\(", code)) else "No fit/predict calls detected.")

    results["mae"] = CheckResult("mae", _has_any(code, ["mean_absolute_error", "MAE", "mean(abs(", "mean_absolute_percentage_error"]), "Found MAE-related calculation." if _has_any(code, ["mean_absolute_error", "MAE", "mean(abs(", "mean_absolute_percentage_error"]) else "MAE not detected.")
    results["rmse"] = CheckResult("rmse", _has_any(code, ["root_mean_squared_error", "sqrt(mean_squared_error", "RMSE"]), "Found RMSE-related calculation." if _has_any(code, ["root_mean_squared_error", "sqrt(mean_squared_error", "RMSE"]) else "RMSE not detected.")
    results["r2"] = CheckResult("r2", _has_any(code, ["r2_score", "R2", "r²", "r^2"]), "Found R²-related calculation." if _has_any(code, ["r2_score", "R2", "r²", "r^2"]) else "R² not detected.")
    results["metric_interpretation"] = CheckResult("metric_interpretation", _has_any(full_text or code, ["mae", "rmse", "r2", "metric", "lower is better", "higher is better", "best model"]), "Found language suggesting metric interpretation/comparison." if _has_any(full_text or code, ["mae", "rmse", "r2", "metric", "lower is better", "higher is better", "best model"]) else "Metric interpretation requires contextual review; no clear interpretation detected.", "low")

    results["cross_validation"] = CheckResult("cross_validation", _has_any(code, ["cross_val_score", "cross_validate", "KFold(", "RepeatedKFold", "GridSearchCV", "RandomizedSearchCV"]), "Found a cross-validation pattern." if _has_any(code, ["cross_val_score", "cross_validate", "KFold(", "RepeatedKFold", "GridSearchCV", "RandomizedSearchCV"]) else "Cross-validation not detected.")
    results["cv_mean"] = CheckResult("cv_mean", _has_any(code, ["mean()", "mean_score", "cv_mean", "cv_scores.mean", "scores.mean"]), "Found a likely mean CV aggregation." if _has_any(code, ["mean()", "mean_score", "cv_mean", "cv_scores.mean", "scores.mean"]) else "No clear mean CV aggregation detected.")
    results["cv_std"] = CheckResult("cv_std", _has_any(code, ["std()", "std_score", "cv_std", "scores.std", "cv_scores.std"]), "Found a likely CV standard-deviation aggregation." if _has_any(code, ["std()", "std_score", "cv_std", "scores.std", "cv_scores.std"]) else "No clear CV standard deviation detected.")

    results["randomized_search"] = CheckResult("randomized_search", "RandomizedSearchCV" in code, "RandomizedSearchCV detected." if "RandomizedSearchCV" in code else "RandomizedSearchCV not detected.")
    results["randomized_search_cv"] = CheckResult("randomized_search_cv", bool(re.search(r"RandomizedSearchCV\s*\(.*?cv\s*=", code, re.I | re.S)), "RandomizedSearchCV includes an explicit cv parameter." if bool(re.search(r"RandomizedSearchCV\s*\(.*?cv\s*=", code, re.I | re.S)) else "No explicit cv parameter detected inside RandomizedSearchCV.")
    results["best_params"] = CheckResult("best_params", _has_any(code, ["best_params_", "best_score_", "best_estimator_"]), "Found best-parameter/search-result reporting." if _has_any(code, ["best_params_", "best_score_", "best_estimator_"]) else "No best-parameter/search-result reporting detected.")
    results["large_search_space"] = _large_search_space(code)

    results["actual_vs_predicted"] = CheckResult("actual_vs_predicted", bool(re.search(r"actual.*pred|pred.*actual|y_test.*y_pred|y_true.*y_pred", code, re.I | re.S)) and _has_any(code, ["scatter(", "plot(", "plt."]), "Found an actual-vs-predicted plotting pattern." if bool(re.search(r"actual.*pred|pred.*actual|y_test.*y_pred|y_true.*y_pred", code, re.I | re.S)) and _has_any(code, ["scatter(", "plot(", "plt."]) else "Actual-vs-predicted visualization not confidently detected.")
    results["residual_visualization"] = CheckResult("residual_visualization", bool(re.search(r"residual", code, re.I)) and _has_any(code, ["scatter(", "hist(", "plot(", "plt."]), "Found residual visualization." if bool(re.search(r"residual", code, re.I)) and _has_any(code, ["scatter(", "hist(", "plot(", "plt."]) else "Residual visualization not confidently detected.")
    results["model_comparison"] = CheckResult("model_comparison", _has_any(code, ["compare", "comparison", "results =", "model_results", "results_df", "sort_values"]), "Found a likely model-comparison pattern." if _has_any(code, ["compare", "comparison", "results =", "model_results", "results_df", "sort_values"]) else "No clear model comparison detected.")
    results["model_justification"] = CheckResult("model_justification", _has_any(full_text or code, ["best model", "final model", "selected model", "because", "justif"]), "Found language suggesting model selection/justification." if _has_any(full_text or code, ["best model", "final model", "selected model", "because", "justif"]) else "No clear model-selection justification detected.", "low")

    results["no_preprocessing_leakage"] = _leakage_check(code)
    return results


def _large_search_space(code: str) -> CheckResult:
    if "RandomizedSearchCV" not in code:
        return CheckResult("large_search_space", False, "RandomizedSearchCV not detected.")
    # Heuristic: count parameter keys between common dict brackets near param_distributions.
    matches = re.findall(r"param_distributions\s*=\s*\{(.*?)\}", code, re.I | re.S)
    if not matches:
        # Also support direct positional dictionaries.
        matches = re.findall(r"RandomizedSearchCV\s*\(.*?\{(.*?)\}.*?\)", code, re.I | re.S)
    count = 0
    for block in matches:
        count = max(count, len(re.findall(r"['\"]?[A-Za-z_][A-Za-z0-9_]*['\"]?\s*:", block)))
    passed = count >= 4
    return CheckResult("large_search_space", passed, f"Detected approximately {count} hyperparameter dimensions; Phase 1 threshold is 4+.")


def _leakage_check(code: str) -> CheckResult:
    patterns = [
        r"(?:StandardScaler|MinMaxScaler|RobustScaler|SimpleImputer|PolynomialFeatures)\s*\(.*?\)\s*\.fit_transform\s*\(\s*X\s*\)",
        r"(?:scaler|imputer)\s*=.*\n.*\.fit_transform\s*\(\s*X\s*\)",
    ]
    if any(re.search(p, code, re.I | re.S) for p in patterns) and re.search(r"cross_val_score|cross_validate|KFold|RandomizedSearchCV", code, re.I):
        return CheckResult("no_preprocessing_leakage", False, "Potential leakage: preprocessing appears to be fit/transformed on the full feature matrix before cross-validation/search. Review manually.", "high")
    if _has_any(code, ["Pipeline(", "make_pipeline("]) and re.search(r"cross_val_score|cross_validate|RandomizedSearchCV", code, re.I):
        return CheckResult("no_preprocessing_leakage", True, "Pipeline detected alongside cross-validation/search; this is a positive signal against preprocessing leakage.", "medium")
    return CheckResult("no_preprocessing_leakage", True, "No deterministic leakage pattern detected. This is not proof that leakage is absent.", "low")


def run_checks(extracted: dict) -> dict[str, CheckResult]:
    return check_code(_code_text(extracted), extracted.get("text", ""))
