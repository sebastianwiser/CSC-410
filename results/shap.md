# SHAP interpretation

Selected model: safe_xgb_tuned. Explanation target: underlying oriented XGBoost raw margin (log odds); final probabilities also symmetrize player order.
Training-only stratified sample; no extra test calls. Values describe model associations, not causal effects.
Rank/log-rank and Elo/form are correlated; attribution can shift between substitutes. Tree path dependence does not resolve confounding.
Per-surface samples can be small and do not establish reliable surface-specific mechanisms.

## global

| Feature | Mean absolute SHAP |
|---|---:|
| num__elo_diff | 0.332183 |
| num__log_rank_diff | 0.253458 |
| num__rank_diff | 0.248753 |
| num__age_diff | 0.067097 |
| num__form_diff | 0.059337 |
| num__pts_diff | 0.053879 |
| cat__best_of_3 | 0.023698 |
| num__h2h_rate | 0.022865 |

## carpet

| Feature | Mean absolute SHAP |
|---|---:|
| num__rank_diff | 0.675609 |
| num__log_rank_diff | 0.239135 |
| num__elo_diff | 0.090314 |
| num__age_diff | 0.065185 |
| num__form_diff | 0.058263 |
| cat__round_rr | 0.043478 |
| num__pts_diff | 0.028838 |
| cat__tourney_level_d | 0.025039 |

## clay

| Feature | Mean absolute SHAP |
|---|---:|
| num__elo_diff | 0.379709 |
| num__log_rank_diff | 0.230978 |
| num__rank_diff | 0.194018 |
| num__age_diff | 0.070326 |
| num__pts_diff | 0.058752 |
| num__form_diff | 0.058703 |
| num__h2h_rate | 0.022732 |
| cat__best_of_3 | 0.021713 |

## grass

| Feature | Mean absolute SHAP |
|---|---:|
| num__elo_diff | 0.448304 |
| num__log_rank_diff | 0.275811 |
| num__rank_diff | 0.150380 |
| num__pts_diff | 0.067920 |
| num__age_diff | 0.067226 |
| num__form_diff | 0.063892 |
| cat__best_of_3 | 0.033427 |
| cat__surface_grass | 0.028147 |

## hard

| Feature | Mean absolute SHAP |
|---|---:|
| num__elo_diff | 0.295518 |
| num__log_rank_diff | 0.261105 |
| num__rank_diff | 0.177762 |
| num__age_diff | 0.064742 |
| num__form_diff | 0.055980 |
| num__pts_diff | 0.048112 |
| num__h2h_rate | 0.025130 |
| cat__best_of_3 | 0.019362 |
