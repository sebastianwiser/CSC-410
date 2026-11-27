# ATP match prediction: reproducible evidence and additive extensions

Predict an ATP singles match winner from pre-match rankings and historical match information. This portfolio asks whether richer tennis features improve discrimination and probability quality over simple baselines. Original work remains in `CSC410Pres/`; original model scripts and their logic are unchanged. New experiments live in `portfolio/` and `extensions/`.

[TEAM NOTE: who built what]

## Results and protocol

Training uses the configured period ending before the holdout. The executed split is recorded in `portfolio/common.py` and each final run JSON: train 2014–2020, test 2021–2024. The processed-data cohort has **10,185 physical test matches**; the ATP-file cohort has **11,568**. These are distinct datasets, not mirrored-row counts. Natural-key duplicate checks found 0 and 0 duplicates respectively.

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

Legacy reproductions retain the original feature construction, including leakage risks. They are excluded from deployment selection. Model 1 uses the ATP-file cohort; the separate `logistic_baseline` uses Model 2's processed cohort and baseline factory. They are not interchangeable labels for the same experiment. The MLP, safe logistic and both safe XGBoost runs share the same base features and split. `point_in_time_serve` adds historical serve features to untuned safe XGBoost. It does not replace a legacy model.

Both player orders are used during training. At evaluation, their probabilities are symmetrized, then a deterministic match-ID hash chooses an outcome-independent orientation. Each match is scored once. No decision threshold is fitted on holdout data. Every final estimator made its holdout probability call once; reporting consumes saved predictions. Fixed historical estimator settings are reproduced, not retrospectively tuned.

Training-only expanding validation years: 2018, 2019, 2020. Selection minimizes equal-year mean Brier. Optuna ran 4 XGBoost trials; the MLP used a fixed configuration with no hyperparameter search. Selected eligible model: **safe_xgb_tuned**, training-fold Brier **0.217186**. Selection was persisted before final holdout evaluation. This reuses the training folds for tuning and family selection; it is not nested cross-validation.

## Honest findings

- Legacy Model 2 improved accuracy over its logistic comparator by 1.905 percentage points, paired bootstrap interval [1.109, 2.700]. Its full-period lookups prevent interpreting that gain as clean prospective evidence.
- Injury and playstyle additions did not improve legacy accuracy. Model 6's gain over Model 2 was 0.226 percentage points, with interval [-0.314, 0.776] and McNemar p=0.4432; there is no convincing accuracy gain in this comparison.
- Training-only tuning improved the selection criterion slightly, but its holdout accuracy difference from untuned safe XGBoost was only 0.043 percentage points, interval [-0.182, 0.268]. Holdout Brier did not improve. Rolling serve features improved Brier descriptively but did not improve accuracy; these small changes do not establish reliable superiority.
- Supplied serve estimates disagreed materially with available ATP aggregates for 63.25% of matched players under the declared >5-percentage-point any-stat rule. Time-window differences and missing ATP statistics can explain some disagreement; this does not prove fabrication.

See [full ablation and paired intervals](results/ablation.md), [serve audit](results/serve_audit.md), [SHAP](results/shap.md), and [numeric ledger](results/NUMBERS.md). Cross-source paired comparisons use a conservative one-to-one player-pair/surface/date-window alignment; ambiguous candidates are excluded. Those comparisons are conditional on the aligned subset and on the linkage assumptions, not full-cohort causal tests.

## Limitations

Serve, injury, and playstyle inputs may be LLM-sourced, including `gemini_*` / `openai_*` files. Generation prompts, dates, sources and health assertions are unverified. Static lifetime/surface lookups, full-period player attributes and Model 6 clusters use future information; original dynamic features also process same-date rows sequentially. Original isotonic calibration uses non-temporal folds and mirrored examples. None of those issues are silently repaired in the legacy scores.

The additive models avoid these historical-feature problems by batching dates and updating after the whole date group. ATP `tourney_date` is the tournament start, so this excludes useful within-event history. Ranking/age/context fields are treated as supplied pre-match metadata; their publication timestamps are not independently verified. Earlier holdout outcomes may update history for later holdout dates, reflecting sequential prediction with frozen model parameters. Missing serve statistics and unavailable pre-file history limit coverage.

Bootstrap intervals use matches as independent units; repeated players/events violate that assumption and may narrow intervals. Paired p-values are unadjusted for multiple comparisons. Cross-source names and dates are imperfect. SHAP describes the selected oriented estimator's raw margin, while the served probability also symmetrizes player order; correlated features make attribution unstable. These are retrospective datasets with unresolved upstream revision/licensing, not a live data feed.

This is an educational analysis and API demonstration, not a betting system or medical assessment. The API requires caller-supplied pre-match profiles and cannot authenticate their timing. Bookmaker benchmarking was skipped after the download request was forbidden. Docker image execution and hosted CI remain unverified.

## Reproduce

Run from the repository root. The tested Python version is 3.11.15; dependencies are fully version-pinned in the lock file. Original `requirements.txt` is preserved.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-portfolio.lock.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/atp-mpl
python -m portfolio.run
python evaluate.py
python -m extensions.llm_audit.audit
python -m portfolio.rebuild
python -m extensions.shap.explain
python -m extensions.mlflow.summarize
python -m portfolio.report
python -m pytest -q
```

Committed predictions are reused and not rescored by `portfolio.run`. `portfolio.rebuild` recreates the selected artifact from training rows only, which is needed in a fresh clone because model binaries are ignored. To explicitly reproduce all final predictions in an independent fresh execution, preserve the current evidence first with `mv results legacy/prior-results`, then run the commands above. Do not tune against the already reported holdout.

```bash
mlflow ui --backend-store-uri ./mlruns
uvicorn extensions.api.app:app --host 127.0.0.1 --port 8000
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data @extensions/api/example.json
docker build -f extensions/api/Dockerfile -t atp-portfolio .
docker run --rm -p 8000:8000 atp-portfolio
```

Open the address printed by MLflow for its local UI; open the API's `/docs` route for its validated schema. Raw MLflow stores and trained binaries stay local; [run summary](results/mlflow_runs.csv) is committed. Docker commands are provided for a machine with Docker installed and were not executed here. CI rebuilds the selected artifact on training data, runs tests, and evaluates committed prediction files.

## Repository preservation

Byte-identical root duplicates were moved to `legacy/root_duplicates/` with compatibility symlinks; differing root data and model scripts remain. The original README is archived. Editor metadata is untracked but remains on disk. Empty gitlinks were replaced by provenance notes and their original commit IDs retained. No remote branch was pushed.
