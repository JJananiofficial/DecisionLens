# DecisionLens

DecisionLens is a Streamlit decision-intelligence workspace for comparing predictive models on both statistical quality and business decision value.

## Model catalogue

- Logistic Regression
- Decision Tree
- Random Forest (+ calibrated variant)
- Extra Trees
- Gradient Boosting (+ calibrated variant)
- Histogram Gradient Boosting (+ calibrated variant)
- Linear Support Vector Machine (calibrated)
- Optional class-weighted Gradient Boosting calibration stress test

Calibration is fitted on a dedicated calibration partition. Decision thresholds and model selection use validation data. The test partition is reserved for final reporting.

## Explainability

- TreeSHAP for supported uncalibrated tree estimators.
- Model-agnostic permutation-SHAP for calibrated models, Linear SVM and other unsupported tree implementations.
- SHAP values for generic calibrated models are in positive-class probability space; TreeSHAP values use the estimator's native output space. Do not compare their absolute magnitudes across methods.
- Preprocessing is fitted only on the training partition. SHAP uses the already-fitted preprocessing pipeline.

## Research features

- ROC-AUC, PR-AUC, Brier score, log loss and ECE
- Paired bootstrap model comparisons
- Business profit and threshold simulator
- Training and calibration time reporting
- Robustness stress tests for missing features and numeric measurement noise
- Subgroup diagnostics
- Reproducibility manifest and research report export

Robustness tests are descriptive diagnostics only and are not used for model selection. The business outcome is based on observed conversion, not causal treatment uplift.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run tests

```bash
PYTHONPATH=. pytest -q
```

## Project structure

```text
decisionlens/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/config.toml
├── decisionlens/
│   ├── data.py
│   ├── decision.py
│   ├── evaluation.py
│   ├── explain.py
│   ├── models.py
│   └── pipeline.py
└── tests/
    └── test_core.py
```

## Technology stack

- Python, pandas and NumPy for data handling
- scikit-learn for preprocessing, classification and calibration
- SHAP for model explanations
- Plotly and Streamlit for interactive visual analytics
- pytest for automated tests

## Reproducibility and interpretation notes

- Set the random seed in the application to reproduce a split and experiment configuration.
- Keep the test partition untouched until final evaluation.
- Calibration is fitted on the calibration partition; operating thresholds are selected on validation data.
- SHAP explains model behaviour, not causal effects.
- The campaign simulator uses observed outcomes and stated economic assumptions; it does not estimate incremental treatment uplift.
- Subgroup metrics are descriptive diagnostics, not a fairness certification.

## Development checks

```bash
python -m py_compile app.py decisionlens/*.py
PYTHONPATH=. pytest -q
```

## Data and privacy

Do not commit private, confidential, personal, or licensed datasets. The `.gitignore` excludes common local data files and Streamlit secrets by default. Use synthetic data for the public demonstration unless you have permission to publish a dataset.

## Future development

Potential extensions include additional model families, richer fairness metrics, model monitoring, cost-sensitive optimisation under uncertainty and deployment-oriented model governance.

## DecisionLens  
An independent portfolio project exploring the intersection of predictive analytics, explainable AI and business decision science.
![DecisionLens Fairness Analysis](screenshots/Fairness.png)

