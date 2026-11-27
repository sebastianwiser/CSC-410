# Reproduced class-project results

These values come from saved per-match predictions produced during the earlier repository review. They are not transcribed from the slides. No final class model was retrained or evaluated on new holdout predictions for this cleanup.

The executed split was train 2014–2020 / test 2021–2024. Each physical match contributes once: probabilities from both player orders were averaged symmetrically, then an outcome-independent match-ID hash selected the scoring orientation. Original model and feature-building functions were used without changing their logic.

| Experiment | Test matches | Accuracy | ROC AUC | Brier |
|---|---:|---:|---:|---:|
| Standalone logistic model | 11,568 | 64.25% | 0.7026 | 0.2185 |
| Logistic comparator on processed data | 10,185 | 63.59% | 0.6995 | 0.2191 |
| Rank-only comparator | 10,185 | 63.71% | 0.6371 | 0.3629 |
| XGBoost with Elo and form | 10,185 | 65.50% | 0.7193 | 0.2124 |
| Add supplied serve statistics | 10,185 | 65.78% | 0.7227 | 0.2115 |
| Add injury reports | 10,185 | 65.49% | 0.7196 | 0.2124 |
| Add playstyle categories | 10,185 | 65.45% | 0.7191 | 0.2125 |
| Combined model retained in the repository | 10,185 | 65.72% | 0.7243 | 0.2109 |

Higher accuracy and ROC AUC are better; lower Brier is better. The rank-only comparator outputs hard decisions rather than calibrated probabilities, so its Brier has a different interpretation.

## What this means for the presentation

The standalone logistic model uses the ATP match-file cohort (11,568 test matches). The logistic comparator, rank baseline and XGBoost experiments use the processed-data cohort (10,185 test matches). The presentation mixes these baseline contexts. The logistic-comparator row above is the appropriate same-data comparison for the improved-model experiment; the standalone script is kept separately.

The richer ranking/Elo/form experiment improved over its logistic comparator in this reproduction. Injury and playstyle additions did not improve accuracy. The serve and combined experiments show small descriptive changes, with no clear paired accuracy gain over the base XGBoost model:

| Addition versus base XGBoost | Accuracy change (percentage points) | Paired bootstrap interval | McNemar p |
|---|---:|---|---:|
| Add supplied serve statistics | +0.285 | [-0.226, +0.785] | 0.2923 |
| Add injury reports | -0.010 | [-0.275, +0.255] | 1.0000 |
| Add playstyle categories | -0.049 | [-0.383, +0.275] | 0.8195 |
| Combined model retained in the repository | +0.226 | [-0.314, +0.776] | 0.4432 |

These are the saved paired percentile match-bootstrap confidence intervals and exact two-sided McNemar tests. The interval level is 95%. They assume independent matches and are not adjusted for multiple comparisons; repeated players and tournaments make that assumption imperfect.

## Limits of the evidence

Original static player attributes and surface history use full-period data; combined-model clusters are also computed across the full period. Original sequential features may consume same-date results, and isotonic calibration uses non-temporal folds. Preserving the original scripts also preserves these issues. These rows should be read as reproductions of classroom experiments, not clean prospective performance estimates.

Serve, injury and playstyle reports may be LLM-sourced. Unknown collection dates, uneven coverage and uncertain source accuracy limit what can be concluded about their value. The deck's betting-market ideas remain motivations and proposed next steps; profitability was not established.

## Check the evidence

From the repository root:

```bash
python scripts/check_project.py
```

This uses the standard library to independently recompute accuracy, ROC AUC and Brier from the saved probabilities, check unique match counts, and verify that these tables match the recorded outputs. It does not fit models or request new predictions.

The [full statistical report](../legacy/portfolio-expansion/results/ablation.md), [raw metrics](../legacy/portfolio-expansion/results/metrics.json), [paired comparisons](../legacy/portfolio-expansion/results/comparisons.json) and [per-match predictions](../legacy/portfolio-expansion/results/predictions) are retained in the archive. Later extension runs are kept there as separate work and are excluded from this class-project table.
