"""Statistical evaluation: discrimination, calibration and bootstrap uncertainty."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    g = df.groupby("bin").agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"), count=("y", "size"))
    return g.reset_index(drop=True)


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """Equal-frequency ECE: weighted mean |observed rate - mean prediction| across bins."""
    tab = calibration_table(y, p, n_bins)
    return float((tab["count"] / tab["count"].sum() * (tab["observed_rate"] - tab["mean_predicted"]).abs()).sum())


def classification_metrics(y, p) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "ROC AUC": roc_auc_score(y, p),
        "PR AUC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "Log loss": log_loss(y, p),
        "ECE": expected_calibration_error(y, p),
    }


def bootstrap_ci(y, p, fn: Callable, n_boot: int = 300, seed: int = 0, alpha: float = 0.05):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], p[i]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap(stat_fn: Callable[[np.ndarray], float], n: int, n_boot: int = 500, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    """Paired bootstrap of a statistic computed on resampled row indices.

    Both models are scored on the *same* resampled customers, so the CI reflects the difference
    between models rather than sampling noise common to both.
    """
    rng = np.random.default_rng(seed)
    vals = np.array([stat_fn(rng.integers(0, n, n)) for _ in range(n_boot)])
    return {"mean": float(vals.mean()), "lo": float(np.quantile(vals, alpha / 2)),
            "hi": float(np.quantile(vals, 1 - alpha / 2)), "p_positive": float((vals > 0).mean())}


def metrics_table(y, probs: dict, n_boot: int = 300) -> pd.DataFrame:
    rows = []
    for name, p in probs.items():
        m = classification_metrics(y, p)
        lo, hi = bootstrap_ci(y, p, roc_auc_score, n_boot=n_boot)
        m["ROC AUC 95% CI"] = f"{lo:.3f} to {hi:.3f}"
        rows.append(pd.Series(m, name=name))
    return pd.DataFrame(rows)
