"""Model and resampling pipeline factory."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from .config import SEED

MODELS = ("dummy", "lr", "rf", "knn", "xgb", "catboost")
STRATEGIES = ("weights", "smote", "undersample", "none")


def estimator(model: str, strategy: str = "weights", *, fast: bool = False):
    if model == "dummy":
        return DummyClassifier(strategy="prior")
    if model == "lr":
        return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced" if strategy == "weights" else None,
                                  random_state=SEED)
    if model == "rf":
        return RandomForestClassifier(n_estimators=80 if fast else 300, min_samples_leaf=2,
            class_weight="balanced_subsample" if strategy == "weights" else None,
            random_state=SEED, n_jobs=-1)
    if model == "knn":
        return KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)
    if model == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=40 if fast else 300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
            scale_pos_weight=1.0 if strategy != "weights" else 10.0,
            tree_method="hist", n_jobs=-1, random_state=SEED, eval_metric="logloss")
    if model == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=50 if fast else 500, depth=6, learning_rate=0.05,
            auto_class_weights="Balanced" if strategy == "weights" else None,
            thread_count=-1, task_type="CPU", random_seed=SEED, verbose=False, allow_writing_files=False)
    raise ValueError(f"Unknown model {model!r}; choose from {MODELS}")


def make_pipeline(model: str, strategy: str = "weights", *, fast: bool = False) -> Pipeline:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}")
    steps = []
    if model in {"lr", "knn"}:
        steps.append(("scale", StandardScaler()))
    if strategy == "smote":
        if model == "knn":
            raise ValueError("SMOTE is excluded for KNN because it makes CPU prediction prohibitively slow")
        steps.append(("resample", SMOTE(random_state=SEED, k_neighbors=5)))
    elif strategy == "undersample":
        steps.append(("resample", RandomUnderSampler(random_state=SEED)))
    steps.append(("model", estimator(model, strategy, fast=fast)))
    return Pipeline(steps)
