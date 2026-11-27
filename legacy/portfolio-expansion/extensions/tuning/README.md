# Training-only XGBoost search

`search.py` searches the additive safe-feature XGBoost model, preserving the
original Model 2 configuration and logic. Its untuned comparator is `safe_xgb`;
`safe_xgb_tuned` uses exactly the same inputs, cohort and final split.

Expanding folds and the fixed search budget are executable constants. All
preprocessing is fitted inside each fold. Both player perspectives stay in the
same date partition. Equal-year mean Brier selects parameters and the eligible
model family before final holdout scoring. The same training folds are reused
for family selection, so this is not a nested generalization estimate.

See results/tuning_details.json, results/tuning_trials.csv and
results/safe_comparisons.json for measured tuning and holdout comparisons.
