"""Turning predicted probabilities into a business decision and measuring its value.

Decision: contact a customer iff predicted conversion probability >= threshold.
Each contact costs `cost_per_contact`; each conversion is worth `value_per_conversion`.
If probabilities were perfectly calibrated, the profit-maximising threshold would be exactly
cost / value (the break-even probability). Comparing that theory to what is actually optimal
shows how much miscalibration hurts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    cost_per_contact: float = 6.0
    value_per_conversion: float = 40.0

    @property
    def breakeven(self) -> float:
        return self.cost_per_contact / self.value_per_conversion


def policy_gain(y, selected, cm: CostModel) -> np.ndarray:
    """Per-customer profit contribution of a contact policy (0 if not contacted)."""
    return np.asarray(selected) * (np.asarray(y) * cm.value_per_conversion - cm.cost_per_contact)


def evaluate_policy(y, p, threshold: float, cm: CostModel) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    sel = p >= threshold
    contacted, conv = int(sel.sum()), int(y[sel].sum())
    profit = float(policy_gain(y, sel, cm).sum())
    return {
        "threshold": float(threshold), "contacted": contacted, "share_contacted": contacted / len(y),
        "conversions": conv, "precision": conv / contacted if contacted else np.nan,
        "recall": conv / y.sum() if y.sum() else np.nan, "profit": profit,
        "profit_per_1000": profit / len(y) * 1000,
    }


def profit_curve(y, p, cm: CostModel) -> pd.DataFrame:
    """Profit at every distinct threshold (contact customers with p >= threshold)."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    last = np.r_[ps[1:] != ps[:-1], True]
    k = np.arange(1, len(p) + 1)[last]
    cum = np.cumsum(ys)[last]
    df = pd.DataFrame({"threshold": ps[last], "contacted": k,
                       "profit": cum * cm.value_per_conversion - k * cm.cost_per_contact})
    df["share_contacted"] = df["contacted"] / len(p)
    none = pd.DataFrame({"threshold": [1.0 + 1e-9], "contacted": [0], "profit": [0.0], "share_contacted": [0.0]})
    return pd.concat([none, df], ignore_index=True)


def best_threshold(y, p, cm: CostModel) -> float:
    """Profit-maximising threshold, chosen on data the caller must keep separate from the test set."""
    c = profit_curve(y, p, cm)
    return float(c.loc[c["profit"].idxmax(), "threshold"])


def decile_table(y, p) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["decile"] = 10 - pd.qcut(df["p"].rank(method="first"), 10, labels=False)  # 1 = highest scores
    g = df.groupby("decile").agg(customers=("y", "size"), conversion_rate=("y", "mean"), mean_predicted=("p", "mean"))
    g["lift"] = g["conversion_rate"] / df["y"].mean()
    return g.reset_index()


def decision_disagreement(p_a, thr_a: float, p_b, thr_b: float) -> dict:
    a, b = np.asarray(p_a) >= thr_a, np.asarray(p_b) >= thr_b
    return {"share_disagree": float((a != b).mean()), "only_a": int((a & ~b).sum()), "only_b": int((~a & b).sum()),
            "both": int((a & b).sum())}


def expected_vs_realized(y, p, threshold: float, cm: CostModel) -> dict:
    """Profit the model *promises* for its contact list vs profit actually realised."""
    y, p = np.asarray(y), np.asarray(p)
    sel = p >= threshold
    expected = float(p[sel].sum() * cm.value_per_conversion - sel.sum() * cm.cost_per_contact)
    realized = float(policy_gain(y, sel, cm).sum())
    return {"expected": expected, "realized": realized, "overpromise": expected - realized}
