# Fixed-budget PyTorch comparison

`model.py` implements the CPU MLP used as `safe_mlp`. The same base features,
cohort, split, preprocessor and match-level scoring are used for safe XGBoost.
Architecture, optimization budget and seed are fixed in code; no hyperparameter
search or holdout early stopping was performed. Train-period family validation
uses the same expanding folds as XGBoost. See results/validation.json and
results/runs/safe_mlp.json for executed configuration and metrics.

Run through `python -m portfolio.run --stage safe`. Existing prediction files are
reused. Determinism is checked in tests/test_portfolio.py for XGBoost and PyTorch.
