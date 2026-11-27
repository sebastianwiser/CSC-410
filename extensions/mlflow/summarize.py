"""Log recorded training trials/folds and auxiliary evaluations; export text run summary.

This command consumes saved outputs and never fits, selects, or predicts again.
"""
import json
from pathlib import Path
import mlflow
import joblib
import pandas as pd
from portfolio.common import ROOT, RESULTS, write_json
from portfolio.metrics import metrics


def logged(name):
    runs = mlflow.search_runs(experiment_names=['ATP portfolio'],
                             filter_string=f"tags.mlflow.runName = '{name}' and attributes.status = 'FINISHED'")
    return not runs.empty


def record(name, params, values, artifacts=()):
    if logged(name):
        return
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(values)
        for p in artifacts:
            mlflow.log_artifact(str(p))


def main():
    mlflow.set_tracking_uri((ROOT/'mlruns').as_uri())
    mlflow.set_experiment('ATP portfolio')
    config = json.loads((RESULTS/'validation.json').read_text())
    for trial in json.loads((RESULTS/'tuning_details.json').read_text()):
        record(f"optuna_trial_{trial['number']}", dict(**trial['params'], stage='training-only walk-forward'),
               {'mean_brier': trial['value']}, [RESULTS/'tuning_details.json'])
        for fold in trial['folds']:
            record(f"optuna_trial_{trial['number']}_fold_{fold['year']}",
                   dict(**trial['params'], train_end=fold['year']-1, validation_year=fold['year']),
                   {'brier': fold['brier'], 'n_matches': fold['n_matches']})
    for name, result in config['scores'].items():
        for fold in result['folds']:
            record(f"{name}_validation_{fold['year']}",
                   {'train_end': fold['year']-1, 'validation_year': fold['year'], 'configuration': name},
                   {'brier': fold['brier'], 'n_matches': fold['n_matches']}, [RESULTS/'validation.json'])
    rank = pd.read_csv(RESULTS/'predictions/rank_baseline.csv')
    record('rank_baseline', {'rule': 'lower rank; neutral ties; symmetrized'},
           metrics(rank.label.to_numpy(), rank.probability.to_numpy()), [RESULTS/'predictions/rank_baseline.csv'])
    audit = json.loads((RESULTS/'serve_audit.json').read_text())
    record('serve_data_audit_all_available_stats', {'material_threshold_pp': audit['threshold_percentage_points']},
           {'material_player_fraction': audit['material_player_fraction'], 'matched_players': audit['matched_players'],
            **{f'{s}_{k}': v for s, d in audit['stats'].items() for k, v in d.items()}},
           [RESULTS/'serve_audit.json', RESULTS/'serve_audit_players.csv'])
    record('selected_model_shap', {'model': config['selected'], 'period': 'training only'}, {},
           [RESULTS/'shap_metadata.json', RESULTS/'shap_importance.csv', RESULTS/'shap.md'])
    # Enrich final-fit records from fitted estimator parameters, without prediction.
    client = mlflow.tracking.MlflowClient()
    for path in sorted((RESULTS/'runs').glob('*.json')):
        record_data = json.loads(path.read_text())
        artifact = RESULTS/'artifacts'/f"{record_data['name']}.joblib"
        if artifact.exists():
            bundle = joblib.load(artifact)
            params = {k: str(value) for k, value in bundle['model'].get_params(deep=True).items()
                      if value is None or isinstance(value, (str, int, float, bool))}
        else:
            params = record_data.get('estimator_parameters', {})
        # On a fresh clone, import saved results honestly rather than claiming a fresh fit.
        record(record_data['name'], dict(**params, imported_saved_output='true'),
               record_data['metrics'], [RESULTS/'predictions'/f"{record_data['name']}.csv"])
        local = mlflow.search_runs(experiment_names=['ATP portfolio'],
                  filter_string=f"tags.mlflow.runName = '{record_data['name']}' and attributes.status = 'FINISHED'")
        for key, value in params.items():
            client.log_param(local.iloc[0].run_id, key, value)
        if 'estimator_parameters' not in record_data:
            record_data['estimator_parameters'] = params
            write_json(path, record_data)
    runs = mlflow.search_runs(experiment_names=['ATP portfolio'])
    keep = [c for c in runs.columns if c in ('run_id', 'status', 'start_time', 'end_time', 'tags.mlflow.runName')
            or c.startswith(('params.', 'metrics.'))]
    runs[keep].sort_values('start_time').to_csv(RESULTS/'mlflow_runs.csv', index=False)
    write_json(RESULTS/'mlflow_summary.json', dict(run_count=len(runs),
               status_counts=runs.status.value_counts().to_dict(),
               note='Includes failed pre-prediction adapter attempts. Validation records are logged from saved training-only fold outputs.'))
    print(runs.status.value_counts().to_dict(), len(runs))


if __name__ == '__main__':
    main()
