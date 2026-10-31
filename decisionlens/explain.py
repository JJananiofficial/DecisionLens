"""Feature explanations: TreeSHAP for the boosted model, LinearSHAP for the baseline,
plus an independent permutation-importance cross-check."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance


def _transform(pipe, X):
    return pipe.named_steps["prep"].transform(X)


def tree_explanation(pipe, X) -> shap.Explanation:
    """TreeSHAP values in log-odds space for a fitted gradient-boosting pipeline."""
    Xt = _transform(pipe, X)
    clf = pipe.named_steps["clf"]
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., -1]
    base = float(np.ravel(explainer.expected_value)[-1])
    return shap.Explanation(values=sv, base_values=np.full(len(Xt), base),
                            data=Xt.to_numpy(), feature_names=list(Xt.columns))


def linear_explanation(pipe, X_background, X) -> shap.Explanation:
    """LinearSHAP values (log-odds) for the logistic-regression pipeline."""
    bg, Xt = _transform(pipe, X_background), _transform(pipe, X)
    expl = shap.LinearExplainer(pipe.named_steps["clf"], shap.maskers.Independent(bg, max_samples=len(bg)))
    exp = expl(Xt)
    exp.feature_names = list(Xt.columns)
    return exp


def aggregate_by_feature(exp: shap.Explanation, cat: list[str]) -> pd.DataFrame:
    """Sum one-hot SHAP columns back to the original feature (e.g. channel_email + channel_social -> channel)."""
    df = pd.DataFrame(exp.values, columns=exp.feature_names)
    out = {}
    for col in df.columns:
        owner = next((c for c in cat if col.startswith(f"{c}_")), col)
        out[owner] = out.get(owner, 0) + df[col]
    return pd.DataFrame(out)


def global_importance(agg: pd.DataFrame) -> pd.Series:
    return agg.abs().mean().sort_values(ascending=False)


def importance_agreement(a: pd.Series, b: pd.Series, k: int = 5) -> dict:
    """How similar are two importance rankings? Spearman rank correlation + top-k overlap."""
    common = a.index.intersection(b.index)
    spearman = float(a[common].rank().corr(b[common].rank()))
    overlap = len(set(a.index[:k]) & set(b.index[:k])) / k
    return {"spearman": spearman, "top_k_overlap": overlap, "k": k}


def permutation_check(model, X, y, n_repeats: int = 5, seed: int = 0) -> pd.Series:
    """Model-agnostic importance (drop in ROC AUC when a feature is shuffled) on held-out data."""
    r = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=seed, n_jobs=-1)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


def additivity_error(pipe, exp: shap.Explanation, X) -> float:
    """Sanity check: base value + sum(SHAP) must reproduce the model's log-odds output."""
    Xt = _transform(pipe, X)
    logit = pipe.named_steps["clf"].decision_function(Xt)
    return float(np.max(np.abs(exp.base_values + exp.values.sum(axis=1) - logit)))


def _unwrap_estimator(model):
    """Return (base pipeline, optional fitted calibration object)."""
    if hasattr(model, "calibrated_classifiers_"):
        calibrated = model.calibrated_classifiers_[0]
        frozen = calibrated.estimator
        base = getattr(frozen, "estimator", frozen)
        return base, calibrated
    return model, None


def model_agnostic_explanation(pipe, X_background: pd.DataFrame, X: pd.DataFrame,
                              max_evals: int | None = None) -> shap.Explanation:
    """Permutation-SHAP on the numeric preprocessed feature matrix.

    Preprocessing is fitted on training data only. Explaining the transformed matrix avoids
    mixed string/numeric masker issues and supports SVM and calibrated model wrappers.
    """
    base_pipe, calibrated = _unwrap_estimator(pipe)
    prep = base_pipe.named_steps["prep"]
    clf = base_pipe.named_steps["clf"]
    bg = prep.transform(X_background)
    sample = prep.transform(X)
    if isinstance(bg, pd.DataFrame):
        feature_names = list(bg.columns)
        bg_values = bg.to_numpy(dtype=float)
        sample_values = sample.to_numpy(dtype=float)
    else:
        feature_names = [f"feature_{i}" for i in range(bg.shape[1])]
        bg_values, sample_values = np.asarray(bg, dtype=float), np.asarray(sample, dtype=float)

    def predict_positive(values):
        frame = pd.DataFrame(values, columns=feature_names)
        if hasattr(clf, "predict_proba"):
            raw = np.asarray(clf.predict_proba(frame))[:, 1]
        else:
            raw = np.asarray(clf.decision_function(frame)).reshape(-1)
        if calibrated is not None:
            calibrator = calibrated.calibrators[0]
            raw = calibrator.predict(raw)
        return np.asarray(raw, dtype=float)

    masker = shap.maskers.Independent(bg_values)
    explainer = shap.Explainer(predict_positive, masker, algorithm="permutation")
    budget = max_evals or (2 * bg_values.shape[1] + 1)
    result = explainer(sample_values, max_evals=budget)
    result.feature_names = feature_names
    result.data = sample_values
    return result


def explanation_for_model(pipe, X_background: pd.DataFrame, X: pd.DataFrame,
                          model_name: str = "") -> shap.Explanation:
    """Use TreeSHAP for supported uncalibrated tree models; otherwise permutation-SHAP."""
    base_pipe, calibrated = _unwrap_estimator(pipe)
    clf = base_pipe.named_steps.get("clf") if hasattr(base_pipe, "named_steps") else None
    supported_tree = clf is not None and clf.__class__.__name__ in {
        "DecisionTreeClassifier", "RandomForestClassifier", "ExtraTreesClassifier",
        "GradientBoostingClassifier",
    }
    if supported_tree and calibrated is None:
        Xt = _transform(base_pipe, X)
        explainer = shap.TreeExplainer(clf)
        values = explainer.shap_values(Xt)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[..., -1]
        base = float(np.ravel(explainer.expected_value)[-1])
        return shap.Explanation(values=values, base_values=np.full(len(Xt), base),
                                data=Xt.to_numpy(), feature_names=list(Xt.columns))
    return model_agnostic_explanation(pipe, X_background, X)
