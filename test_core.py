import numpy as np
import pytest

from decisionlens import data as D, decision as dec, evaluation as ev, explain as ex, models as M, pipeline as pl


@pytest.fixture(scope="module")
def exp():
    df = D.generate_campaign_data(n=6000, seed=7)
    return pl.run_experiment(df, D.TARGET, D.NUMERIC, D.CATEGORICAL, seed=7, stress_test=True)


# ---- data ----
def test_generator_hits_base_rate_and_is_reproducible():
    a, b = D.generate_campaign_data(4000, 1), D.generate_campaign_data(4000, 1)
    assert abs(a[D.TARGET].mean() - 0.14) < 0.03
    assert a.equals(b)
    assert a[D.TRUE_PROB].between(0, 1).all() and a.isna().any().any()


def test_load_csv_binary_coercion(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("a,y\n1,no\n2,yes\n3,no\n")
    out = D.load_csv(f, "y")
    assert out["y"].tolist() == [0, 1, 0]
    with pytest.raises(ValueError):
        f.write_text("a,y\n1,a\n2,b\n3,c\n"); D.load_csv(f, "y")


# ---- splits: no leakage ----
def test_splits_disjoint_stratified_and_exclude_truth(exp):
    idx = [set(exp.data[s][0].index) for s in pl.SPLITS]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not idx[i] & idx[j]
    assert D.TRUE_PROB not in exp.data["train"][0].columns
    rates = [exp.data[s][1].mean() for s in pl.SPLITS]
    assert max(rates) - min(rates) < 0.01


# ---- evaluation ----
def test_metrics_sanity():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    perfect = y * 0.98 + 0.01
    m = ev.classification_metrics(y, perfect)
    assert m["ROC AUC"] == 1.0 and m["Brier"] < 0.001
    assert ev.expected_calibration_error(y, np.full(2000, y.mean()) + rng.normal(0, 1e-6, 2000)) < 0.03


def test_ece_detects_miscalibration():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.5, 8000)
    y = (rng.random(8000) < p).astype(int)
    assert ev.expected_calibration_error(y, p) < 0.02
    assert ev.expected_calibration_error(y, np.clip(p * 2, 0, 1)) > 0.1


# ---- decision logic ----
def test_profit_curve_matches_bruteforce():
    rng = np.random.default_rng(2)
    p, y = rng.random(500), rng.integers(0, 2, 500)
    cm = dec.CostModel(5, 30)
    c = dec.profit_curve(y, p, cm)
    for thr in c.threshold.iloc[::50]:
        assert dec.evaluate_policy(y, p, thr, cm)["profit"] == pytest.approx(c.loc[c.threshold == thr, "profit"].iloc[0])
    assert dec.best_threshold(y, p, cm) in set(c.threshold)


def test_breakeven_is_optimal_for_calibrated_probabilities():
    rng = np.random.default_rng(3)
    p = rng.beta(2, 8, 200000)
    y = (rng.random(len(p)) < p).astype(int)
    cm = dec.CostModel(6, 40)
    assert abs(dec.best_threshold(y, p, cm) - cm.breakeven) < 0.02


def test_contact_none_is_zero_and_all_matches_formula(exp):
    cm = dec.CostModel()
    s = pl.decision_summary(exp, cm).set_index("model")
    y = exp.data["test"][1]
    assert s.loc["Contact no one", "profit"] == 0
    assert s.loc["Contact everyone", "profit"] == pytest.approx(y.sum() * 40 - len(y) * 6)


# ---- models ----
def test_models_beat_chance_and_oracle_is_upper_bound(exp):
    y = exp.data["test"][1]
    for m in exp.model_names:
        assert ev.classification_metrics(y, exp.probs[m]["test"])["ROC AUC"] > 0.65
    s = pl.decision_summary(exp, dec.CostModel())
    oracle = s.loc[s.model == "Oracle (true probabilities)", "profit"].iloc[0]
    best_real = s.loc[s.model.isin(exp.model_names), "profit"].max()
    # an upper bound in expectation; on one finite test sample allow a little noise
    assert oracle >= best_real - 0.05 * abs(oracle)


def test_stress_test_distorts_then_calibration_repairs(exp):
    y = exp.data["test"][1]
    raw = ev.expected_calibration_error(y, exp.probs[M.BALANCED]["test"])
    fixed = ev.expected_calibration_error(y, exp.probs[M.BALANCED_CAL]["test"])
    assert raw > 0.15 and fixed < raw / 3


# ---- explanations ----
def test_shap_additivity_and_agreement(exp):
    X = exp.data["test"][0].iloc[:300]
    t = ex.tree_explanation(exp.models[M.EXPLAINABLE], X)
    l = ex.linear_explanation(exp.models[M.SIMPLE], exp.data["train"][0].iloc[:200], X)
    assert ex.additivity_error(exp.models[M.EXPLAINABLE], t, X) < 1e-6
    assert ex.additivity_error(exp.models[M.SIMPLE], l, X) < 1e-6
    gt = ex.global_importance(ex.aggregate_by_feature(t, exp.cat))
    gl = ex.global_importance(ex.aggregate_by_feature(l, exp.cat))
    assert ex.importance_agreement(gt, gl)["spearman"] > 0.6
    assert set(D.NUMERIC + D.CATEGORICAL) == set(gt.index)


def test_comparison_and_findings_run(exp):
    cm = dec.CostModel()
    r = pl.compare_models(exp, cm, M.EXPLAINABLE, M.SIMPLE, n_boot=100)
    assert r["profit_diff_per_1000"]["lo"] <= r["profit_diff_per_1000"]["hi"]
    assert len(pl.build_findings(exp, cm)) == 6


def test_advanced_model_catalogue_and_fit_timings(exp):
    expected = {
        M.SIMPLE, M.DECISION_TREE, M.RANDOM_FOREST, M.RANDOM_FOREST_CAL,
        M.EXTRA_TREES, M.EXPLAINABLE, M.EXPLAINABLE_CAL,
        M.HIST_GB, M.HIST_GB_CAL, M.SVM,
    }
    assert expected.issubset(set(exp.model_names))
    assert set(exp.fit_seconds).issuperset(expected)
    assert all(exp.fit_seconds[name] >= 0 for name in expected)


def test_model_agnostic_shap_supports_calibrated_svm(exp):
    X = exp.data["test"][0].iloc[:3]
    background = exp.data["train"][0].iloc[:25]
    explanation = ex.explanation_for_model(exp.models[M.SVM], background, X, M.SVM)
    assert explanation.values.shape[0] == len(X)
    assert explanation.values.shape[1] == len(explanation.feature_names)


def test_robustness_table_reports_clean_and_perturbed_scenarios(exp):
    table = pl.robustness_table(exp)
    assert {"Clean test data", "10% random feature missingness",
            "5% numeric measurement noise", "Combined perturbation"}.issubset(set(table["Scenario"]))
    assert table["AUC change vs clean"].notna().all()
