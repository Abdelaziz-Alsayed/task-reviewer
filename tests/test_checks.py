from evaluation.deterministic import check_code


def test_core_regression_checks():
    code = '''
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
linear = LinearRegression()
forest = RandomForestRegressor()
search = RandomizedSearchCV(model, param_distributions={"n_estimators": [100,200], "max_depth": [3,5], "min_samples_split": [2,4], "max_features": ["sqrt", 1.0]}, cv=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=KFold(5))
print(cv_scores.mean(), cv_scores.std(), search.best_params_)
'''
    checks = check_code(code)
    for key in ["randomized_search", "randomized_search_cv", "large_search_space", "mae", "rmse", "r2", "cross_validation", "cv_mean", "cv_std", "linear_regression", "flexible_regression", "train_predict"]:
        assert checks[key].passed, key
