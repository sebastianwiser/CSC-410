# Holdout ablation

Computed from saved, symmetrized per-match probabilities. Each physical source match is scored once.
Models using different source files have different cohorts; paired comparisons use only confidently aligned matches.
Legacy rows retain full-period feature leakage and non-temporal isotonic calibration. They are descriptive reproductions, not leakage-free estimates.

| Model | Test matches | Accuracy | Accuracy bootstrap CI | ROC AUC | Brier |
|---|---:|---:|---|---:|---:|
| logistic_baseline | 10185 | 0.635935 | [0.626605, 0.645069] | 0.699514 | 0.219130 |
| model_1 | 11568 | 0.642462 | [0.633904, 0.650847] | 0.702615 | 0.218546 |
| model_2 | 10185 | 0.654983 | [0.645754, 0.664310] | 0.719307 | 0.212440 |
| model_3 | 10185 | 0.657830 | [0.648500, 0.667160] | 0.722679 | 0.211493 |
| model_4 | 10185 | 0.654885 | [0.645754, 0.664116] | 0.719557 | 0.212421 |
| model_5 | 10185 | 0.654492 | [0.645653, 0.663819] | 0.719121 | 0.212511 |
| model_6 | 10185 | 0.657241 | [0.647815, 0.666375] | 0.724326 | 0.210894 |
| point_in_time_serve | 11568 | 0.648859 | [0.640128, 0.657590] | 0.718170 | 0.213392 |
| rank_baseline | 10185 | 0.637113 | [0.627585, 0.646343] | 0.637118 | 0.362887 |
| safe_logistic | 11568 | 0.647476 | [0.638913, 0.656725] | 0.715903 | 0.214363 |
| safe_mlp | 11568 | 0.649637 | [0.640906, 0.658629] | 0.717320 | 0.213616 |
| safe_xgb | 11568 | 0.648945 | [0.640212, 0.658022] | 0.717163 | 0.213835 |
| safe_xgb_tuned | 11568 | 0.649378 | [0.640647, 0.658454] | 0.717135 | 0.213881 |

## Paired accuracy comparisons

Percentile paired match bootstrap; exact two-sided binomial McNemar test on discordant matches.
Intervals assume independent matches; repeated players/tournaments violate that assumption. No multiple-comparison adjustment.

| Comparison | Aligned matches | Difference | Bootstrap CI | McNemar p |
|---|---:|---:|---|---:|
| logistic_baseline vs model_2 | 10185 | -0.019048 | [-0.027000, -0.011092] | 3.02318e-06 |
| logistic_baseline vs logistic_baseline | 10185 | +0.000000 | [+0.000000, +0.000000] | 1 |
| model_1 vs model_2 | 8700 | -0.014023 | [-0.022529, -0.005747] | 0.00148763 |
| model_1 vs logistic_baseline | 8700 | +0.004598 | [+0.000460, +0.008506] | 0.0256212 |
| model_2 vs model_2 | 10185 | +0.000000 | [+0.000000, +0.000000] | 1 |
| model_2 vs logistic_baseline | 10185 | +0.019048 | [+0.011092, +0.027000] | 3.02318e-06 |
| model_3 vs model_2 | 10185 | +0.002847 | [-0.002258, +0.007855] | 0.292313 |
| model_3 vs logistic_baseline | 10185 | +0.021895 | [+0.013844, +0.030145] | 1.38649e-07 |
| model_4 vs model_2 | 10185 | -0.000098 | [-0.002749, +0.002553] | 1 |
| model_4 vs logistic_baseline | 10185 | +0.018949 | [+0.011193, +0.026902] | 3.70309e-06 |
| model_5 vs model_2 | 10185 | -0.000491 | [-0.003829, +0.002752] | 0.819465 |
| model_5 vs logistic_baseline | 10185 | +0.018557 | [+0.010700, +0.026119] | 4.5515e-06 |
| model_6 vs model_2 | 10185 | +0.002258 | [-0.003144, +0.007757] | 0.443177 |
| model_6 vs logistic_baseline | 10185 | +0.021306 | [+0.012568, +0.029553] | 3.47544e-07 |
| point_in_time_serve vs model_2 | 8700 | -0.006667 | [-0.013796, +0.000345] | 0.0830746 |
| point_in_time_serve vs logistic_baseline | 8700 | +0.011954 | [+0.004483, +0.019080] | 0.000823555 |
| rank_baseline vs model_2 | 10185 | -0.017869 | [-0.026313, -0.009816] | 1.328e-05 |
| rank_baseline vs logistic_baseline | 10185 | +0.001178 | [-0.001868, +0.004124] | 0.483171 |
| safe_logistic vs model_2 | 8700 | -0.008736 | [-0.015862, -0.001264] | 0.0213231 |
| safe_logistic vs logistic_baseline | 8700 | +0.009885 | [+0.003563, +0.016322] | 0.00259953 |
| safe_mlp vs model_2 | 8700 | -0.006207 | [-0.013103, +0.001034] | 0.104808 |
| safe_mlp vs logistic_baseline | 8700 | +0.012414 | [+0.005632, +0.018966] | 0.000300088 |
| safe_xgb vs model_2 | 8700 | -0.006437 | [-0.013793, +0.000922] | 0.0941641 |
| safe_xgb vs logistic_baseline | 8700 | +0.012184 | [+0.005402, +0.018621] | 0.000286482 |
| safe_xgb_tuned vs model_2 | 8700 | -0.005862 | [-0.012989, +0.001494] | 0.125451 |
| safe_xgb_tuned vs logistic_baseline | 8700 | +0.012759 | [+0.005747, +0.019425] | 0.000232135 |

Rank baseline probabilities are hard decisions with ties at neutral probability; its Brier is therefore not calibrated.
The split, seed, bootstrap budget and fold definitions are executable constants in portfolio/.
Full precision metrics, discordant counts, and alignment exclusions are in the adjacent JSON/CSV artifacts.
