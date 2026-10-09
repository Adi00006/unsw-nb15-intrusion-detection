"""Model zoo and hyper-parameter search spaces."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from . import config as cfg

MODEL_NAMES = {
    "lr": "LogisticRegression",
    "dt": "DecisionTree",
    "rf": "RandomForest",
    "xgb": "XGBoost",
    "mlp": "MLP",
}


def make_model(key: str, n_classes: int, n_estimators: int = 300):
    rs = cfg.RANDOM_STATE
    if key == "lr":
        return LogisticRegression(max_iter=300, class_weight="balanced")
    if key == "dt":
        return DecisionTreeClassifier(class_weight="balanced", random_state=rs)
    if key == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=-1, class_weight="balanced", random_state=rs
        )
    if key == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=100, random_state=rs
        )
    if key == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            learning_rate=0.1,
            tree_method="hist",
            n_jobs=-1,
            random_state=rs,
            eval_metric="logloss" if n_classes == 2 else "mlogloss",
        )
    raise ValueError(f"Unknown model key: {key}")


PARAM_GRIDS = {
    "lr": {"clf__C": [0.1, 1, 10]},
    "dt": {"clf__max_depth": [10, 20, None], "clf__min_samples_leaf": [1, 2, 5]},
    "rf": {
        "clf__n_estimators": [200, 300, 500],
        "clf__max_depth": [None, 20, 40],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2"],
    },
    "xgb": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [6, 8, 10],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__subsample": [0.7, 0.9, 1.0],
        "clf__colsample_bytree": [0.7, 0.9, 1.0],
    },
    "mlp": {
        "clf__hidden_layer_sizes": [(64,), (128, 64), (256, 128)],
        "clf__alpha": [1e-4, 1e-3, 1e-2],
        "clf__learning_rate_init": [1e-3, 5e-4],
    },
}
