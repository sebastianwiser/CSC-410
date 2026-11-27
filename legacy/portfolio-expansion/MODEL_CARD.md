# Model card

[TEAM NOTE: who built what]

## Intended use

Educational ATP singles win-probability comparison and a local, validated API demo. Not intended for wagering, financial decisions, medical conclusions, player injury judgments, or automated consequential decisions. No profitability claim is supported.

## Selected model and decision rule

safe_xgb_tuned, selected by minimum training-only expanding-window mean Brier (0.217186), before evaluating the held-out period. Final fitted parameters and feature lists are recorded in `results/runs/` and `results/validation.json`. Threshold is fixed at neutral probability. Deployment symmetrizes the two player-order probabilities.

Training: 2014–2020. Final test: 2021–2024. Training-period validation: 2018, 2019, 2020. Seed: 42. Test observations: 11,568 unique source matches. Accuracy 0.649378, percentile match-bootstrap confidence interval [0.640647, 0.658454], ROC AUC 0.717135, Brier 0.213881. Confidence level and bootstrap budget are executable in `portfolio/metrics.py`; only accuracy has a reported uncertainty interval. Test results are descriptive and did not select this model.

## Data and provenance

Checked-in ATP match CSVs supply ranks, age, ranking points, context, outcomes and serve counts. Processed match CSV supplies the separate legacy cohort. SHA256 hashes of local files are committed in `results/data_sha256.json`. Local hashes do not establish upstream license, revision or authenticity. The retrieved sources behind the processed file and absent submodules remain incompletely verified.

Serve, injury and playstyle reports may be LLM-sourced (`gemini_*` / `openai_*` and report JSON files); prompts, generation dates and primary sources are unverified. They are retained for legacy reproduction but excluded from the selected model. Audited serve errors are against pooled available ATP counts, not complete career ground truth.

## Historical-feature policy and evaluation limits

The selected model consumes rankings/age/context supplied as pre-match metadata and date-batched Elo, form and head-to-head signals. Match publication timestamps are not supplied. Outcome-derived historical source dates must precede the target tournament date. No same-date results enter historical features. Tournament start dates make this conservative; early holdout outcomes may enter later holdout histories without parameter refitting. Ranking publication-time correctness and live feed behavior have not been verified.

Legacy estimates preserve future-dependent static lookups, sequential same-date features, non-temporal calibration and mirrored calibration folds. They are not eligible for model selection. Cohort differences preclude comparing table accuracies as if all models saw identical matches. Cross-source paired significance depends on conservative approximate linkage. Repeated player/tournament observations reduce independence; bootstrap intervals and unadjusted p-values should be read cautiously. Small training-fold score differences may be noise; tuning/family selection is not nested.

## Transparency and engineering

Global/per-surface SHAP values in `results/shap.md` explain the oriented estimator margin, not directly the symmetrized probability. Correlated ranking and history variables can share or exchange attribution; no causal or player-level claim follows. Missing pre-file history, serve coverage, era shifts, geographic/tour-level coverage, name matching, and undocumented exclusions limit generalization. Per-subgroup calibration and fairness were not established.

The API validates finite ranges, distinct players and asserted history chronology, but caller-supplied profiles are not independently verified. It loads the training-selected artifact and uses the same symmetry rule. Tests cover chronological historical feature construction, future-data perturbation, deterministic predictions/metrics, unique saved predictions, and API smoke/validation. Docker and hosted CI were not run in this environment. Full gaps are recorded in `results/skipped_and_unverified.json`.
