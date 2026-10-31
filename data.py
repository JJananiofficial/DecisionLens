"""Data generation and loading.

The default dataset is a *simulated* marketing-campaign dataset. Simulation is used on purpose:
because we know the true conversion probability of every customer, we can benchmark both models
against an "oracle" and check whether SHAP explanations recover the real drivers. The app also
accepts any binary-target CSV (e.g. the UCI Bank Marketing data) for real-data analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "converted"
TRUE_PROB = "true_probability"
NUMERIC = [
    "age", "tenure_months", "past_purchases", "avg_order_value",
    "days_since_last_purchase", "email_open_rate", "site_visits_30d", "discount_sensitivity",
]
CATEGORICAL = ["channel", "region"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_campaign_data(n: int = 20000, seed: int = 42, base_rate: float = 0.14) -> pd.DataFrame:
    """Simulate customers targeted by a marketing campaign.

    Conversion probability mixes linear effects with realistic non-linearities (recency decay,
    engagement x browsing interaction, discount sensitivity that only matters for big baskets,
    channel x age interaction) so a flexible model has genuine signal a linear one cannot capture.
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 71, n)
    tenure = np.clip(rng.gamma(2.0, 14.0, n), 1, 120)
    past = rng.poisson(np.clip(tenure / 10.0, 0.3, None))
    aov = np.clip(rng.lognormal(4.2, 0.5, n), 10, 600)
    days = np.clip(rng.exponential(60.0, n), 1, 365)
    open_rate = rng.beta(2, 5, n)
    visits = rng.poisson(3 + 6 * open_rate)
    disc = rng.beta(2, 3, n)
    channel = rng.choice(["email", "social", "search"], n, p=[0.5, 0.3, 0.2])
    region = rng.choice(["north", "south", "east", "west"], n)

    ch_eff = pd.Series(channel).map({"email": 0.2, "social": -0.2, "search": 0.5}).to_numpy()
    score = (
        0.8 * np.log1p(past)
        + 2.0 * open_rate
        + 0.06 * visits
        - 1.8 * (1 - np.exp(-days / 45.0))
        + 1.3 * disc * (aov > 90)
        + 1.5 * open_rate * np.tanh((visits - 5) / 3.0)
        + ch_eff
        + 0.9 * ((channel == "social") & (age < 30))
        - 0.0003 * (age - 40) ** 2
    )
    lo, hi = -12.0, 6.0  # solve intercept so the average conversion rate hits base_rate
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(score + mid).mean() > base_rate:
            hi = mid
        else:
            lo = mid
    p = _sigmoid(score + (lo + hi) / 2)
    y = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        "age": age, "tenure_months": tenure.round(1), "past_purchases": past,
        "avg_order_value": aov.round(2), "days_since_last_purchase": days.round(0),
        "email_open_rate": open_rate.round(3), "site_visits_30d": visits,
        "discount_sensitivity": disc.round(3), "channel": channel, "region": region,
        TARGET: y, TRUE_PROB: p,
    })
    # realistic missingness (completely at random) so imputation is actually exercised
    df.loc[rng.random(n) < 0.04, "email_open_rate"] = np.nan
    df.loc[rng.random(n) < 0.03, "avg_order_value"] = np.nan
    return df


def load_csv(file, target: str, positive_value=None) -> pd.DataFrame:
    """Load an uploaded CSV and coerce `target` to a 0/1 integer column."""
    df = pd.read_csv(file)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    vals = df[target].dropna().unique()
    if len(vals) != 2:
        raise ValueError(f"Target must be binary; found {len(vals)} distinct values.")
    if positive_value is not None:
        if positive_value not in vals:
            raise ValueError(f"Positive class '{positive_value}' is not present in the target column.")
        positive = positive_value
    elif set(map(str, vals)) <= {"0", "1"}:
        positive = 1
    else:
        positive = sorted(vals, key=str)[-1]  # backward-compatible default; UI asks explicitly
    out = df.dropna(subset=[target]).copy()
    out[target] = (out[target] == positive).astype(int)
    return out.reset_index(drop=True)


def infer_feature_types(df: pd.DataFrame, target: str, drop: tuple = ()):
    feats = [c for c in df.columns if c != target and c not in drop]
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in feats if c not in num]
    return num, cat
