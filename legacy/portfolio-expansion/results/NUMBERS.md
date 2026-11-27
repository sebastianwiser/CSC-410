# Numeric evidence ledger

All measured values below are generated from executed output files. Configuration values are recorded by the code that used them.
No approximate note figures are substituted for measurements. Per-match probabilities remain in results/predictions/ rather than being repeated here.

| Model | Match cohort | Accuracy | ROC AUC | Brier |
|---|---:|---:|---:|---:|
| logistic_baseline | 10,185 | 63.59% | 0.6995 | 0.2191 |
| model_1 | 11,568 | 64.25% | 0.7026 | 0.2185 |
| model_2 | 10,185 | 65.50% | 0.7193 | 0.2124 |
| model_3 | 10,185 | 65.78% | 0.7227 | 0.2115 |
| model_4 | 10,185 | 65.49% | 0.7196 | 0.2124 |
| model_5 | 10,185 | 65.45% | 0.7191 | 0.2125 |
| model_6 | 10,185 | 65.72% | 0.7243 | 0.2109 |
| point_in_time_serve | 11,568 | 64.89% | 0.7182 | 0.2134 |
| rank_baseline | 10,185 | 63.71% | 0.6371 | 0.3629 |
| safe_logistic | 11,568 | 64.75% | 0.7159 | 0.2144 |
| safe_mlp | 11,568 | 64.96% | 0.7173 | 0.2136 |
| safe_xgb | 11,568 | 64.89% | 0.7172 | 0.2138 |
| safe_xgb_tuned | 11,568 | 64.94% | 0.7171 | 0.2139 |

## Differences from requested approximate reference figures

References are user-supplied targets, not measured observations. Flag means the new measurement differs when rounded to the reference precision.

| Model | Measured accuracy | Difference from reference (percentage points) | Flag |
|---|---:|---:|---|
| logistic_baseline | 63.593520% | -0.006480 | no |
| model_2 | 65.498282% | -0.001718 | no |
| model_6 | 65.724104% | +0.024104 | no |

## Other numeric outputs

- `ablation.md`, `comparisons.json`: accuracy confidence intervals, paired differences, discordant counts, exact McNemar p-values and matched counts.
- `safe_comparisons.json`: same-source paired extension comparisons against untuned XGBoost.
- `validation.json`, `tuning_details.json`, `tuning_trials.csv`: all executed training fold scores, counts, trial values and parameters.
- `serve_audit.json`, `serve_audit_players.csv`: final per-stat errors/correlations/coverage and player-level disagreements, without a ranking filter. The initial rank-complete audit is retained under legacy/audits/.
- `shap_importance.csv`, `shap_metadata.json`: final global/per-surface mean absolute attribution and sample sizes; earlier winner-oriented stratified sample retained in legacy/shap/.
- `mlflow_runs.csv`, `runs/`: run parameters, measured metrics, elapsed times and statuses.
- `point_in_time_provenance.csv`, `alignment.csv`: historical-date bounds and source linkage dates/counts.
- `environment.json`, `tests.xml`, `data_sha256.json`: runtime, duplicate checks, test results, local input hashes.
- `reproduction_check.json`: tracked-files-only reproduction, training-only artifact rebuild, and unchanged saved metrics.

MLflow contains 54 logged runs: {'FINISHED': 48, 'FAILED': 6}. Failed adapter attempts occurred before holdout prediction.

## Skipped or unverified

- **bookmaker_benchmark**: Skipped: Tennis-Data download index returned HTTP Forbidden. Supplied processed odds were not independently re-downloaded or provenance-verified.
- **docker_build**: Not run: Docker CLI is absent. Dockerfile supplied; Python API smoke tests passed.
- **github_hosted_actions**: Workflow supplied; hosted execution not verified because this branch has not been pushed.
- **submodule_source_recovery**: Empty gitlinks had no .gitmodules remote URLs. Commit IDs preserved in legacy/submodules.json; source contents remain unavailable.
- **legacy_leakage_fix**: Deliberately excluded: changing original feature/model logic is prohibited. Full-period lookups and same-date history risks are documented; additive historical model is separately tested.
- **failed_attempts**: Initial legacy adapters failed before prediction because metadata duplicated surface columns; adapter fixed, original scripts unchanged. Initial SHAP sample exceeded a small surface cohort; bounded sampling fixed it.
- **upstream_source_license**: Upstream revision/license and LLM prompts/generation dates were not recoverable from checked-in CSVs alone. SHA256 inventory records local inputs, not upstream authenticity.
