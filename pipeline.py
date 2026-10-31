"""End-to-end experiment: split -> train -> calibrate -> predict -> decide -> compare."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from . import decision as dec
from . import evaluation as ev
from . import models as mdl
from .data import TRUE_PROB

SPLITS = ("train", "cal", "val", "test")


@dataclass
class Experiment:
    target: str
    num: list
    cat: list
    data: dict                      # split -> (X, y)
    models: dict
    probs: dict                     # model -> split -> np.ndarray
    true_prob_test: Optional[np.ndarray] = None
    seed: int = 42
    notes: list = field(default_factory=list)
    fit_seconds: dict = field(default_factory=dict)

    @property
    def model_names(self):
        return list(self.models)


def make_splits(df: pd.DataFrame, target: str, seed: int = 42, drop=()):
    """50% train / 15% calibration / 15% threshold-validation / 20% test (all stratified).

    Separate calibration and validation sets matter: a threshold tuned on the same data used to
    calibrate would be optimistically biased, and the test set is touched only for final scoring.
    """
    X = df.drop(columns=[target, *drop])
    y = df[target]
    idx = np.arange(len(df))
    i_rest, i_test = train_test_split(idx, test_size=0.20, stratify=y, random_state=seed)
    i_train, i_hold = train_test_split(i_rest, train_size=0.625, stratify=y.iloc[i_rest], random_state=seed)
    i_cal, i_val = train_test_split(i_hold, test_size=0.5, stratify=y.iloc[i_hold], random_state=seed)
    return {n: (X.iloc[i], y.iloc[i].to_numpy()) for n, i in
            zip(SPLITS, (i_train, i_cal, i_val, i_test))}, i_test


def run_experiment(df: pd.DataFrame, target: str, num: list, cat: list, seed: int = 42,
                   stress_test: bool = False) -> Experiment:
    has_truth = TRUE_PROB in df.columns
    drop = (TRUE_PROB,) if has_truth else ()
    data, i_test = make_splits(df, target, seed, drop)
    feats = num + cat
    data = {k: (X[feats], y) for k, (X, y) in data.items()}
    models, fit_seconds = mdl.fit_all(*data["train"], *data["cal"], num, cat, seed,
                                      stress_test, return_timings=True)
    probs = {m: {s: mdl.predict_proba(models[m], data[s][0]) for s in SPLITS} for m in models}
    truth = df[TRUE_PROB].iloc[i_test].to_numpy() if has_truth else None
    return Experiment(target, num, cat, data, models, probs, truth, seed, fit_seconds=fit_seconds)


def decision_summary(exp: Experiment, cm: dec.CostModel) -> pd.DataFrame:
    """Per model: threshold chosen on validation data -> realised outcome on the untouched test set.

    Two policies per model: 'tuned' (profit-optimal on validation) and 'break-even' (cost/value,
    which is only optimal if the probabilities are calibrated).
    """
    y_val, y_test = exp.data["val"][1], exp.data["test"][1]
    rows = []
    for m in exp.model_names:
        for policy, thr in (("tuned on validation", dec.best_threshold(y_val, exp.probs[m]["val"], cm)),
                            ("break-even (cost/value)", cm.breakeven)):
            r = dec.evaluate_policy(y_test, exp.probs[m]["test"], thr, cm)
            rows.append({"model": m, "policy": policy, **r})
    base = {"threshold": np.nan}
    for name, sel in (("Contact everyone", np.ones_like(y_test)), ("Contact no one", np.zeros_like(y_test))):
        profit = float(dec.policy_gain(y_test, sel, cm).sum())
        rows.append({"model": name, "policy": "baseline", **base, "contacted": int(sel.sum()),
                     "share_contacted": float(sel.mean()), "conversions": int(y_test[sel == 1].sum()),
                     "precision": np.nan, "recall": np.nan, "profit": profit,
                     "profit_per_1000": profit / len(y_test) * 1000})
    if exp.true_prob_test is not None:
        r = dec.evaluate_policy(y_test, exp.true_prob_test, cm.breakeven, cm)
        rows.append({"model": "Oracle (true probabilities)", "policy": "upper bound", **r})
    return pd.DataFrame(rows)


def compare_models(exp: Experiment, cm: dec.CostModel, a: str, b: str, n_boot: int = 500) -> dict:
    """Paired-bootstrap comparison of two models on AUC, Brier score and realised profit."""
    y = exp.data["test"][1]
    pa, pb = exp.probs[a]["test"], exp.probs[b]["test"]
    yv = exp.data["val"][1]
    ta = dec.best_threshold(yv, exp.probs[a]["val"], cm)
    tb = dec.best_threshold(yv, exp.probs[b]["val"], cm)
    gain = dec.policy_gain(y, pa >= ta, cm) - dec.policy_gain(y, pb >= tb, cm)
    brier = lambda yy, pp: float(np.mean((pp - yy) ** 2))
    return {
        "auc_diff": ev.paired_bootstrap(lambda i: roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]), len(y), n_boot),
        "brier_diff": ev.paired_bootstrap(lambda i: brier(y[i], pa[i]) - brier(y[i], pb[i]), len(y), n_boot),
        "profit_diff_per_1000": ev.paired_bootstrap(lambda i: gain[i].mean() * 1000, len(y), n_boot),
        "disagreement": dec.decision_disagreement(pa, ta, pb, tb),
        "thresholds": (ta, tb),
    }


def build_findings(exp: Experiment, cm: dec.CostModel) -> list[str]:
    """Plain-English conclusions generated from the numbers (never hard-coded)."""
    a, b, c = mdl.EXPLAINABLE, mdl.SIMPLE, mdl.EXPLAINABLE_CAL
    y = exp.data["test"][1]
    out = []
    auc = {m: roc_auc_score(y, exp.probs[m]["test"]) for m in exp.model_names}
    cmp_ = compare_models(exp, cm, a, b)
    d = cmp_["auc_diff"]
    sig = "statistically clear" if d["lo"] > 0 or d["hi"] < 0 else "not statistically clear"
    out.append(f"**Ranking quality.** {a} AUC {auc[a]:.3f} vs {b} {auc[b]:.3f} "
               f"(difference {d['mean']:+.3f}, 95% CI {d['lo']:+.3f} to {d['hi']:+.3f}; {sig}).")
    ece = {m: ev.expected_calibration_error(y, exp.probs[m]["test"]) for m in exp.model_names}
    out.append(f"**Calibration.** ECE: {b} {ece[b]:.3f}, {a} {ece[a]:.3f}, {c} {ece[c]:.3f}. "
               f"{'Calibrating improved the boosted model.' if ece[c] < ece[a] else 'Calibration did not improve the boosted model here.'}")
    ds = decision_summary(exp, cm)
    tuned = ds[(ds.policy == "tuned on validation") & ds.model.isin(exp.model_names)].set_index("model")
    # Select the candidate using validation outcomes only. The test set is then used
    # solely to report the selected policy's final performance.
    y_val = exp.data["val"][1]
    validation_profit = {}
    validation_thresholds = {}
    for model_name in exp.model_names:
        threshold = dec.best_threshold(y_val, exp.probs[model_name]["val"], cm)
        validation_thresholds[model_name] = threshold
        validation_profit[model_name] = dec.evaluate_policy(
            y_val, exp.probs[model_name]["val"], threshold, cm
        )["profit_per_1000"]
    best = max(validation_profit, key=validation_profit.get)
    out.append(f"**Business value.** Validation selected {best} (validation profit "
               f"{validation_profit[best]:,.0f} per 1,000). Its untouched test-set result was "
               f"{tuned.loc[best, 'profit_per_1000']:,.0f} per 1,000 customers, contacting "
               f"{tuned.loc[best, 'share_contacted']:.0%} of them.")
    p = cmp_["profit_diff_per_1000"]
    psig = "a real difference" if p["lo"] > 0 or p["hi"] < 0 else "within noise (the CI includes zero)"
    out.append(f"**Is the extra complexity worth it?** {a} vs {b}: profit difference {p['mean']:+,.0f} per 1,000 "
               f"customers (95% CI {p['lo']:+,.0f} to {p['hi']:+,.0f}) - {psig}.")
    be = ds[(ds.policy == "break-even (cost/value)")].set_index("model")
    gap = be.loc[c, "profit_per_1000"] - be.loc[a, "profit_per_1000"]
    out.append(f"**Why calibration matters for decisions.** Using the textbook break-even threshold "
               f"({cm.breakeven:.0%}), the calibrated model earns {gap:+,.0f} per 1,000 customers vs the uncalibrated one.")
    dg = cmp_["disagreement"]
    out.append(f"**Do the models decide differently?** They disagree on {dg['share_disagree']:.1%} of customers.")
    return out



def robustness_table(exp: Experiment, seed: int = 123) -> pd.DataFrame:
    """Descriptive stress test: missingness and numeric noise on held-out features.

    These perturbations are diagnostics only; they are not used for model selection.
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss
    rng = np.random.default_rng(seed)
    X = exp.data["test"][0].copy()
    y = exp.data["test"][1]
    numeric = [c for c in exp.num if c in X.columns]
    scenarios = {"Clean test data": X.copy()}
    missing = X.copy()
    for col in X.columns:
        mask = rng.random(len(missing)) < 0.10
        missing.loc[mask, col] = np.nan
    scenarios["10% random feature missingness"] = missing
    noisy = X.copy()
    for col in numeric:
        values = pd.to_numeric(noisy[col], errors="coerce")
        scale = float(values.std(skipna=True) or 0.0)
        if scale:
            noise = rng.normal(0, 0.05 * scale, len(noisy))
            noisy[col] = values + noise
    scenarios["5% numeric measurement noise"] = noisy
    combined = noisy.copy()
    for col in X.columns:
        mask = rng.random(len(combined)) < 0.05
        combined.loc[mask, col] = np.nan
    scenarios["Combined perturbation"] = combined

    rows = []
    for model_name, model in exp.models.items():
        clean_prob = exp.probs[model_name]["test"]
        clean_auc = roc_auc_score(y, clean_prob)
        for scenario, frame in scenarios.items():
            if scenario == "Clean test data":
                p = clean_prob
            else:
                p = mdl.predict_proba(model, frame)
            rows.append({"Model": model_name, "Scenario": scenario,
                         "ROC AUC": roc_auc_score(y, p),
                         "AUC change vs clean": roc_auc_score(y, p) - clean_auc,
                         "Brier": brier_score_loss(y, p)})
    return pd.DataFrame(rows)
