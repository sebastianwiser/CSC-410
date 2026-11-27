"""Generate narrative reports from executed artifacts; no numbers copied from notes."""
import json
from pathlib import Path
from portfolio.common import ROOT, RESULTS, write_json, SEED, VALIDATION_YEARS
from portfolio.metrics import paired
import pandas as pd


def read(name):
    return json.loads((RESULTS/name).read_text())


def main():
    m, c, v = read('metrics.json'), read('comparisons.json'), read('validation.json')
    audit, env, mlflow = read('serve_audit.json'), read('environment.json'), read('mlflow_summary.json')
    table = ['| Model | Match cohort | Accuracy | ROC AUC | Brier |', '|---|---:|---:|---:|---:|']
    for name, r in m.items():
        table.append(f"| {name} | {r['n_matches']:,} | {100*r['accuracy']:.2f}% | {r['roc_auc']:.4f} | {r['brier']:.4f} |")
    table = '\n'.join(table)
    # References below are user-supplied comparison targets, never measured results.
    references = {'logistic_baseline': .636, 'model_2': .655, 'model_6': .657}
    flags = {name: dict(measured_accuracy=m[name]['accuracy'],
                       delta_percentage_points=100*(m[name]['accuracy']-target),
                       differs_at_reference_precision=round(m[name]['accuracy']*100, 1) != round(target*100, 1))
             for name, target in references.items()}
    write_json(RESULTS/'reference_differences.json', flags)
    safe_predictions = {name: pd.read_csv(RESULTS/'predictions'/f'{name}.csv')
                        for name in ('safe_xgb', 'safe_xgb_tuned', 'safe_mlp', 'point_in_time_serve')}
    safe_comparisons = {f'{name} vs safe_xgb': paired(p, safe_predictions['safe_xgb'])
                        for name, p in safe_predictions.items()}
    write_json(RESULTS/'safe_comparisons.json', safe_comparisons)
    skips = {
        'bookmaker_benchmark': 'Skipped: Tennis-Data download index returned HTTP Forbidden. Supplied processed odds were not independently re-downloaded or provenance-verified.',
        'docker_build': 'Not run: Docker CLI is absent. Dockerfile supplied; Python API smoke tests passed.',
        'github_hosted_actions': 'Workflow supplied; hosted execution not verified because this branch has not been pushed.',
        'submodule_source_recovery': 'Empty gitlinks had no .gitmodules remote URLs. Commit IDs preserved in legacy/submodules.json; source contents remain unavailable.',
        'legacy_leakage_fix': 'Deliberately excluded: changing original feature/model logic is prohibited. Full-period lookups and same-date history risks are documented; additive historical model is separately tested.',
        'failed_attempts': 'Initial legacy adapters failed before prediction because metadata duplicated surface columns; adapter fixed, original scripts unchanged. Initial SHAP sample exceeded a small surface cohort; bounded sampling fixed it.',
        'upstream_source_license': 'Upstream revision/license and LLM prompts/generation dates were not recoverable from checked-in CSVs alone. SHA256 inventory records local inputs, not upstream authenticity.'}
    write_json(RESULTS/'skipped_and_unverified.json', skips)
    logistic_gain = c['model_2 vs logistic_baseline']
    m6_gain = c['model_6 vs model_2']
    safe_gain = safe_comparisons['safe_xgb_tuned vs safe_xgb']
    readme = f'''# ATP match prediction: reproducible evidence and additive extensions

Predict an ATP singles match winner from pre-match rankings and historical match information. This portfolio asks whether richer tennis features improve discrimination and probability quality over simple baselines. Original work remains in `CSC410Pres/`; original model scripts and their logic are unchanged. New experiments live in `portfolio/` and `extensions/`.

[TEAM NOTE: who built what]

## Results and protocol

Training uses the configured period ending before the holdout. The executed split is recorded in `portfolio/common.py` and each final run JSON: train 2014–2020, test 2021–2024. The processed-data cohort has **{m['model_2']['n_matches']:,} physical test matches**; the ATP-file cohort has **{m['model_1']['n_matches']:,}**. These are distinct datasets, not mirrored-row counts. Natural-key duplicate checks found {env['processed_test_duplicate_natural_keys']} and {env['raw_test_duplicate_natural_keys']} duplicates respectively.

{table}

Legacy reproductions retain the original feature construction, including leakage risks. They are excluded from deployment selection. Model 1 uses the ATP-file cohort; the separate `logistic_baseline` uses Model 2's processed cohort and baseline factory. They are not interchangeable labels for the same experiment. The MLP, safe logistic and both safe XGBoost runs share the same base features and split. `point_in_time_serve` adds historical serve features to untuned safe XGBoost. It does not replace a legacy model.

Both player orders are used during training. At evaluation, their probabilities are symmetrized, then a deterministic match-ID hash chooses an outcome-independent orientation. Each match is scored once. No decision threshold is fitted on holdout data. Every final estimator made its holdout probability call once; reporting consumes saved predictions. Fixed historical estimator settings are reproduced, not retrospectively tuned.

Training-only expanding validation years: {', '.join(map(str, VALIDATION_YEARS))}. Selection minimizes equal-year mean Brier. Optuna ran {len(read('tuning_details.json'))} XGBoost trials; the MLP used a fixed configuration with no hyperparameter search. Selected eligible model: **{v['selected']}**, training-fold Brier **{v['scores'][v['selected']]['mean_brier']:.6f}**. Selection was persisted before final holdout evaluation. This reuses the training folds for tuning and family selection; it is not nested cross-validation.

## Honest findings

- Legacy Model 2 improved accuracy over its logistic comparator by {100*logistic_gain['accuracy_difference']:.3f} percentage points, paired bootstrap interval [{100*logistic_gain['ci95'][0]:.3f}, {100*logistic_gain['ci95'][1]:.3f}]. Its full-period lookups prevent interpreting that gain as clean prospective evidence.
- Injury and playstyle additions did not improve legacy accuracy. Model 6's gain over Model 2 was {100*m6_gain['accuracy_difference']:.3f} percentage points, with interval [{100*m6_gain['ci95'][0]:.3f}, {100*m6_gain['ci95'][1]:.3f}] and McNemar p={m6_gain['mcnemar_p']:.4f}; there is no convincing accuracy gain in this comparison.
- Training-only tuning improved the selection criterion slightly, but its holdout accuracy difference from untuned safe XGBoost was only {100*safe_gain['accuracy_difference']:.3f} percentage points, interval [{100*safe_gain['ci95'][0]:.3f}, {100*safe_gain['ci95'][1]:.3f}]. Holdout Brier did not improve. Rolling serve features improved Brier descriptively but did not improve accuracy; these small changes do not establish reliable superiority.
- Supplied serve estimates disagreed materially with available ATP aggregates for {100*audit['material_player_fraction']:.2f}% of matched players under the declared >{audit['threshold_percentage_points']:g}-percentage-point any-stat rule. Time-window differences and missing ATP statistics can explain some disagreement; this does not prove fabrication.

See [full ablation and paired intervals](results/ablation.md), [serve audit](results/serve_audit.md), [SHAP](results/shap.md), and [numeric ledger](results/NUMBERS.md). Cross-source paired comparisons use a conservative one-to-one player-pair/surface/date-window alignment; ambiguous candidates are excluded. Those comparisons are conditional on the aligned subset and on the linkage assumptions, not full-cohort causal tests.

## Limitations

Serve, injury, and playstyle inputs may be LLM-sourced, including `gemini_*` / `openai_*` files. Generation prompts, dates, sources and health assertions are unverified. Static lifetime/surface lookups, full-period player attributes and Model 6 clusters use future information; original dynamic features also process same-date rows sequentially. Original isotonic calibration uses non-temporal folds and mirrored examples. None of those issues are silently repaired in the legacy scores.

The additive models avoid these historical-feature problems by batching dates and updating after the whole date group. ATP `tourney_date` is the tournament start, so this excludes useful within-event history. Ranking/age/context fields are treated as supplied pre-match metadata; their publication timestamps are not independently verified. Earlier holdout outcomes may update history for later holdout dates, reflecting sequential prediction with frozen model parameters. Missing serve statistics and unavailable pre-file history limit coverage.

Bootstrap intervals use matches as independent units; repeated players/events violate that assumption and may narrow intervals. Paired p-values are unadjusted for multiple comparisons. Cross-source names and dates are imperfect. SHAP describes the selected oriented estimator's raw margin, while the served probability also symmetrizes player order; correlated features make attribution unstable. These are retrospective datasets with unresolved upstream revision/licensing, not a live data feed.

This is an educational analysis and API demonstration, not a betting system or medical assessment. The API requires caller-supplied pre-match profiles and cannot authenticate their timing. Bookmaker benchmarking was skipped after the download request was forbidden. Docker image execution and hosted CI remain unverified.

## Reproduce

Run from the repository root. The tested Python version is {env['python']}; dependencies are fully version-pinned in the lock file. Original `requirements.txt` is preserved.

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
'''
    (ROOT/'README.md').write_text(readme)
    selected = m[v['selected']]
    lo, hi = selected['accuracy_ci95']
    card = f'''# Model card

[TEAM NOTE: who built what]

## Intended use

Educational ATP singles win-probability comparison and a local, validated API demo. Not intended for wagering, financial decisions, medical conclusions, player injury judgments, or automated consequential decisions. No profitability claim is supported.

## Selected model and decision rule

{v['selected']}, selected by minimum training-only expanding-window mean Brier ({v['scores'][v['selected']]['mean_brier']:.6f}), before evaluating the held-out period. Final fitted parameters and feature lists are recorded in `results/runs/` and `results/validation.json`. Threshold is fixed at neutral probability. Deployment symmetrizes the two player-order probabilities.

Training: 2014–2020. Final test: 2021–2024. Training-period validation: {', '.join(map(str, VALIDATION_YEARS))}. Seed: {SEED}. Test observations: {selected['n_matches']:,} unique source matches. Accuracy {selected['accuracy']:.6f}, percentile match-bootstrap confidence interval [{lo:.6f}, {hi:.6f}], ROC AUC {selected['roc_auc']:.6f}, Brier {selected['brier']:.6f}. Confidence level and bootstrap budget are executable in `portfolio/metrics.py`; only accuracy has a reported uncertainty interval. Test results are descriptive and did not select this model.

## Data and provenance

Checked-in ATP match CSVs supply ranks, age, ranking points, context, outcomes and serve counts. Processed match CSV supplies the separate legacy cohort. SHA256 hashes of local files are committed in `results/data_sha256.json`. Local hashes do not establish upstream license, revision or authenticity. The retrieved sources behind the processed file and absent submodules remain incompletely verified.

Serve, injury and playstyle reports may be LLM-sourced (`gemini_*` / `openai_*` and report JSON files); prompts, generation dates and primary sources are unverified. They are retained for legacy reproduction but excluded from the selected model. Audited serve errors are against pooled available ATP counts, not complete career ground truth.

## Historical-feature policy and evaluation limits

The selected model consumes rankings/age/context supplied as pre-match metadata and date-batched Elo, form and head-to-head signals. Match publication timestamps are not supplied. Outcome-derived historical source dates must precede the target tournament date. No same-date results enter historical features. Tournament start dates make this conservative; early holdout outcomes may enter later holdout histories without parameter refitting. Ranking publication-time correctness and live feed behavior have not been verified.

Legacy estimates preserve future-dependent static lookups, sequential same-date features, non-temporal calibration and mirrored calibration folds. They are not eligible for model selection. Cohort differences preclude comparing table accuracies as if all models saw identical matches. Cross-source paired significance depends on conservative approximate linkage. Repeated player/tournament observations reduce independence; bootstrap intervals and unadjusted p-values should be read cautiously. Small training-fold score differences may be noise; tuning/family selection is not nested.

## Transparency and engineering

Global/per-surface SHAP values in `results/shap.md` explain the oriented estimator margin, not directly the symmetrized probability. Correlated ranking and history variables can share or exchange attribution; no causal or player-level claim follows. Missing pre-file history, serve coverage, era shifts, geographic/tour-level coverage, name matching, and undocumented exclusions limit generalization. Per-subgroup calibration and fairness were not established.

The API validates finite ranges, distinct players and asserted history chronology, but caller-supplied profiles are not independently verified. It loads the training-selected artifact and uses the same symmetry rule. Tests cover chronological historical feature construction, future-data perturbation, deterministic predictions/metrics, unique saved predictions, and API smoke/validation. Docker and hosted CI were not run in this environment. Full gaps are recorded in `results/skipped_and_unverified.json`.
'''
    (ROOT/'MODEL_CARD.md').write_text(card)
    lines = ['# Numeric evidence ledger', '',
             'All measured values below are generated from executed output files. Configuration values are recorded by the code that used them.',
             'No approximate note figures are substituted for measurements. Per-match probabilities remain in results/predictions/ rather than being repeated here.', '', table, '',
             '## Differences from requested approximate reference figures', '',
             'References are user-supplied targets, not measured observations. Flag means the new measurement differs when rounded to the reference precision.', '',
             '| Model | Measured accuracy | Difference from reference (percentage points) | Flag |',
             '|---|---:|---:|---|']
    for name, r in flags.items():
        lines.append(f"| {name} | {100*r['measured_accuracy']:.6f}% | {r['delta_percentage_points']:+.6f} | {'yes' if r['differs_at_reference_precision'] else 'no'} |")
    lines += ['', '## Other numeric outputs', '',
              '- `ablation.md`, `comparisons.json`: accuracy confidence intervals, paired differences, discordant counts, exact McNemar p-values and matched counts.',
              '- `safe_comparisons.json`: same-source paired extension comparisons against untuned XGBoost.',
              '- `validation.json`, `tuning_details.json`, `tuning_trials.csv`: all executed training fold scores, counts, trial values and parameters.',
              '- `serve_audit.json`, `serve_audit_players.csv`: final per-stat errors/correlations/coverage and player-level disagreements, without a ranking filter. The initial rank-complete audit is retained under legacy/audits/.',
              '- `shap_importance.csv`, `shap_metadata.json`: global/per-surface mean absolute attribution and sample sizes.',
              '- `mlflow_runs.csv`, `runs/`: run parameters, measured metrics, elapsed times and statuses.',
              '- `point_in_time_provenance.csv`, `alignment.csv`: historical-date bounds and source linkage dates/counts.',
              '- `environment.json`, `tests.xml`, `data_sha256.json`: runtime, duplicate checks, test results, local input hashes.', '',
              f"MLflow contains {mlflow['run_count']} logged runs: {mlflow['status_counts']}. Failed adapter attempts occurred before holdout prediction.",
              '', '## Skipped or unverified', '']
    lines += [f'- **{key}**: {reason}' for key, reason in skips.items()]
    (RESULTS/'NUMBERS.md').write_text('\n'.join(lines)+'\n')
    print('Generated README, model card, numeric ledger, and skip report from executed artifacts.')


if __name__ == '__main__':
    main()
