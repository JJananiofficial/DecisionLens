"""Model catalogue for DecisionLens: linear, tree, boosting and margin-based classifiers."""
from __future__ import annotations

import warnings
import time
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model")

SIMPLE = "Logistic Regression"
DECISION_TREE = "Decision Tree"
RANDOM_FOREST = "Random Forest"
RANDOM_FOREST_CAL = "Random Forest (calibrated)"
EXTRA_TREES = "Extra Trees"
EXPLAINABLE = "Gradient Boosting"
EXPLAINABLE_CAL = "Gradient Boosting (calibrated)"
HIST_GB = "Histogram Gradient Boosting"
HIST_GB_CAL = "Histogram Gradient Boosting (calibrated)"
SVM = "Linear SVM (calibrated)"
BALANCED = "Stress test: GB class-balanced"
BALANCED_CAL = "Stress test: GB class-balanced + calibrated"

CORE_MODELS = (
    SIMPLE, DECISION_TREE, RANDOM_FOREST, RANDOM_FOREST_CAL, EXTRA_TREES,
    EXPLAINABLE, EXPLAINABLE_CAL, HIST_GB, HIST_GB_CAL, SVM,
)
TREE_MODELS = (DECISION_TREE, RANDOM_FOREST, RANDOM_FOREST_CAL, EXTRA_TREES, EXPLAINABLE, EXPLAINABLE_CAL)


def make_preprocessor(num, cat, scale: bool) -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [("num", Pipeline(numeric_steps), num),
         ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def build_simple(num, cat, seed=42):
    return Pipeline([("prep", make_preprocessor(num, cat, True)),
                     ("clf", LogisticRegressionCV(Cs=8, cv=5, scoring="neg_log_loss",
                                                  max_iter=2000, random_state=seed, n_jobs=-1))])


def build_decision_tree(num, cat, seed=42):
    return Pipeline([("prep", make_preprocessor(num, cat, False)),
                     ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=25,
                                                    class_weight=None, random_state=seed))])


def build_random_forest(num, cat, seed=42):
    return Pipeline([("prep", make_preprocessor(num, cat, False)),
                     ("clf", RandomForestClassifier(n_estimators=220, max_features="sqrt",
                         min_samples_leaf=8, n_jobs=-1, random_state=seed, class_weight=None))])


def build_extra_trees(num, cat, seed=42):
    return Pipeline([("prep", make_preprocessor(num, cat, False)),
                     ("clf", ExtraTreesClassifier(n_estimators=220, max_features="sqrt",
                         min_samples_leaf=8, n_jobs=-1, random_state=seed, class_weight=None))])


def build_explainable(num, cat, seed=42):
    return Pipeline([("prep", make_preprocessor(num, cat, False)),
                     ("clf", GradientBoostingClassifier(n_estimators=350, learning_rate=0.04,
                         max_depth=3, subsample=0.8, min_samples_leaf=25,
                         n_iter_no_change=20, validation_fraction=0.15, random_state=seed))])


def build_hist_gb(num, cat, seed=42):
    return Pipeline([("prep", make_preprocessor(num, cat, False)),
                     ("clf", HistGradientBoostingClassifier(max_iter=220, learning_rate=0.08,
                         max_leaf_nodes=15, min_samples_leaf=25, l2_regularization=1.0,
                         early_stopping=True, random_state=seed))])


def build_svm(num, cat, seed=42):
    # SVC is calibrated on the dedicated calibration partition in fit_all().
    return Pipeline([("prep", make_preprocessor(num, cat, True)),
                     ("clf", LinearSVC(C=0.1, class_weight=None, max_iter=5000,
                                      random_state=seed))])


def calibrate(fitted_pipeline, X_cal, y_cal):
    """Fit sigmoid/isotonic calibration on the dedicated calibration partition only."""
    method = "isotonic" if len(y_cal) >= 1500 else "sigmoid"
    return CalibratedClassifierCV(FrozenEstimator(fitted_pipeline), method=method).fit(X_cal, y_cal)


def fit_all(X_train, y_train, X_cal, y_cal, num, cat, seed=42, stress_test=False, return_timings=False):
    builders = [
        (SIMPLE, build_simple), (DECISION_TREE, build_decision_tree),
        (RANDOM_FOREST, build_random_forest), (EXTRA_TREES, build_extra_trees),
        (EXPLAINABLE, build_explainable), (HIST_GB, build_hist_gb),
    ]
    out = {}
    fitted = {}
    fit_seconds = {}
    for name, builder in builders:
        started = time.perf_counter()
        model = builder(num, cat, seed).fit(X_train, y_train)
        fit_seconds[name] = time.perf_counter() - started
        fitted[name] = model
        out[name] = model

    # Calibrated variants are distinct probability-producing models. Calibration sees only X_cal/y_cal.
    for name, base_name in ((RANDOM_FOREST_CAL, RANDOM_FOREST),
                            (EXPLAINABLE_CAL, EXPLAINABLE),
                            (HIST_GB_CAL, HIST_GB)):
        started = time.perf_counter()
        out[name] = calibrate(fitted[base_name], X_cal, y_cal)
        fit_seconds[name] = time.perf_counter() - started

    started = time.perf_counter()
    svm = build_svm(num, cat, seed).fit(X_train, y_train)
    out[SVM] = calibrate(svm, X_cal, y_cal)
    fit_seconds[SVM] = time.perf_counter() - started

    if stress_test:
        bal = build_explainable(num, cat, seed)
        started = time.perf_counter()
        bal.fit(X_train, y_train, clf__sample_weight=compute_sample_weight("balanced", y_train))
        fit_seconds[BALANCED] = time.perf_counter() - started
        out[BALANCED] = bal
        started = time.perf_counter()
        out[BALANCED_CAL] = calibrate(bal, X_cal, y_cal)
        fit_seconds[BALANCED_CAL] = time.perf_counter() - started
    return (out, fit_seconds) if return_timings else out


def predict_proba(model, X):
    return model.predict_proba(X)[:, 1]
