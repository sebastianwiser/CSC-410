# SHAP interpretation

Selected model: safe_xgb_tuned. Explanation target: underlying oriented XGBoost raw margin (log odds); final probabilities also symmetrize player order.
Training-only outcome-independent player orientation; uniform match sample for global importance, independent bounded samples per surface. No extra test calls. Values describe model associations, not causal effects.
Rank/log-rank and Elo/form are correlated; attribution can shift between substitutes. Tree path dependence does not resolve confounding.
Per-surface samples can be small and do not establish reliable surface-specific mechanisms.

## global

| Feature | Mean absolute SHAP |
|---|---:|
| num__elo_diff | 0.387460 |
| num__log_rank_diff | 0.236917 |
| num__rank_diff | 0.143776 |
| num__age_diff | 0.064055 |
| num__pts_diff | 0.054299 |
| num__form_diff | 0.052625 |
| num__h2h_rate | 0.027398 |
| cat__best_of_3 | 0.021144 |

## carpet

| Feature | Mean absolute SHAP |
|---|---:|
| num__rank_diff | 0.665556 |
| num__log_rank_diff | 0.237561 |
| num__elo_diff | 0.091754 |
| num__age_diff | 0.067281 |
| num__form_diff | 0.056419 |
| cat__round_rr | 0.038055 |
| num__pts_diff | 0.025933 |
| cat__tourney_level_d | 0.023080 |

## clay

| Feature | Mean absolute SHAP |
|---|---:|
| num__elo_diff | 0.378273 |
| num__log_rank_diff | 0.234546 |
| num__rank_diff | 0.194097 |
| num__age_diff | 0.069477 |
| num__form_diff | 0.054660 |
| num__pts_diff | 0.052340 |
| num__h2h_rate | 0.023307 |
| cat__best_of_3 | 0.019965 |

## grass

| Feature | Mean absolute SHAP |
|---|---:|
| num__elo_diff | 0.451014 |
| num__log_rank_diff | 0.280394 |
| num__rank_diff | 0.154575 |
| num__pts_diff | 0.070784 |
| num__age_diff | 0.067591 |
| num__form_diff | 0.064402 |
| cat__best_of_3 | 0.032189 |
| num__h2h_rate | 0.027141 |

## hard

| Feature | Mean absolute SHAP |
|---|---:|
| num__elo_diff | 0.300414 |
| num__log_rank_diff | 0.259107 |
| num__rank_diff | 0.177605 |
| num__age_diff | 0.070527 |
| num__form_diff | 0.056155 |
| num__pts_diff | 0.044584 |
| num__h2h_rate | 0.026840 |
| cat__best_of_3 | 0.018354 |
