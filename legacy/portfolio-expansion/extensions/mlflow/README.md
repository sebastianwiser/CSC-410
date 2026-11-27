# Experiment tracking

Final fits are logged while they execute. `summarize.py` additionally logs saved
training fold/trial results and audit/SHAP artifacts without any new fitting or
holdout prediction, enriches final-fit parameters from fitted estimators, and
exports results/mlflow_runs.csv. Failed initial adapter fits remain visible.

```bash
python -m extensions.mlflow.summarize
mlflow ui --backend-store-uri ./mlruns
```

The MLflow store and trained binaries are ignored. Only text summaries and
prediction/audit artifacts are committed. On a fresh clone, prior raw tracking
runs are absent; a fresh full execution recreates them. Training-only artifact
rebuilds do not recreate original MLflow run IDs. To open historical run summaries
without rerunning experiments, inspect the committed CSV.
