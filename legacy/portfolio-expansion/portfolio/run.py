"""Run once in a fresh results directory; cached predictions prevent test rescoring."""
import argparse
import json
import time
import traceback
import joblib
import numpy as np
import pandas as pd
import mlflow
from portfolio.common import ROOT, RESULTS, SEED, write_json, fingerprints
from portfolio.legacy import build
from portfolio.predictions import canonical
from portfolio.metrics import metrics
from portfolio.models import make
from extensions.point_in_time.features import load_raw, build as point_features
from extensions.tuning.search import score, tune


def holdout(name, model, data, features, provenance, cv=None):
    path = RESULTS / 'predictions' / f'{name}.csv'
    if path.exists():
        print(f'{name}: using immutable saved predictions', flush=True)
        return
    train = data[data.year.between(2014, 2020)]
    test = data[data.year.between(2021, 2024)]
    with mlflow.start_run(run_name=name):
        mlflow.log_params({'seed': SEED, 'train_start': 2014, 'train_end': 2020,
                           'test_start': 2021, 'test_end': 2024, 'provenance': provenance,
                           'model': str(model.get_params(deep=False))})
        mlflow.log_params({k: str(value) for k, value in model.get_params(deep=True).items()
                           if value is None or isinstance(value, (str, int, float, bool))})
        start = time.monotonic()
        model.fit(train[features], train.label)
        pred = canonical(test, model.predict_proba(test[features])[:, 1])
        result = metrics(pred.label.to_numpy(), pred.probability.to_numpy())
        result.update(n_matches=len(pred), train_rows=len(train),
                      fit_and_predict_seconds=time.monotonic()-start)
        path.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(path, index=False)
        artifact = RESULTS / 'artifacts' / f'{name}.joblib'
        artifact.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': model, 'features': features, 'name': name}, artifact)
        write_json(RESULTS / 'runs' / f'{name}.json', {'name': name, 'metrics': result,
                    'features': features, 'provenance': provenance, 'validation': cv,
                    'mlflow_run_id': mlflow.active_run().info.run_id})
        mlflow.log_metrics(result)
        mlflow.log_artifact(str(path))
        if cv is not None:
            mlflow.log_metric('training_cv_brier', cv['mean_brier'])
        print(name, result, flush=True)


def safe():
    cache = RESULTS / 'cache' / 'point_features.pkl'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        data = pd.read_pickle(cache)
    else:
        data = point_features(load_raw())
        data.to_pickle(cache)
    data = data[data.year.between(2014, 2024)].copy()
    train = data[data.year <= 2020]
    assert (data.source_max_date.isna() | (data.source_max_date < data.date)).all()
    data[['match_id', 'date', 'source_max_date']].drop_duplicates().to_csv(
        RESULTS / 'point_in_time_provenance.csv', index=False)
    cv_path = RESULTS / 'validation.json'
    if cv_path.exists():
        selection = json.loads(cv_path.read_text())
    else:
        params, trials, details = tune(train)
        trials.to_csv(RESULTS / 'tuning_trials.csv', index=False)
        write_json(RESULTS / 'tuning_details.json', details)
        factories = {'safe_logistic': lambda: make('logistic'),
                     'safe_xgb': lambda: make(), 'safe_xgb_tuned': lambda: make(params=params),
                     'safe_mlp': lambda: make('mlp'),
                     'point_in_time_serve': lambda: make(serve=True)}
        scores = {}
        for name, factory in factories.items():
            value, folds = score(factory, train)
            scores[name] = {'mean_brier': value, 'folds': folds}
            print('Training-only validation:', name, value, flush=True)
        selection = {'params': params, 'scores': scores,
                     'selected': min(scores, key=lambda k: scores[k]['mean_brier']),
                     'criterion': 'minimum equal-year mean expanding-window Brier; training years only'}
        write_json(cv_path, selection)
    factories = {'safe_logistic': lambda: make('logistic'),
                 'safe_xgb': lambda: make(),
                 'safe_xgb_tuned': lambda: make(params=selection['params']),
                 'safe_mlp': lambda: make('mlp'),
                 'point_in_time_serve': lambda: make(serve=True)}
    # Selection is persisted before any final model sees the holdout.
    for name, factory in factories.items():
        model = factory()
        features = model.named_steps['pre'].transformers[0][2] + model.named_steps['pre'].transformers[1][2]
        holdout(name, model, data, features, 'date-batched ATP historical features', selection['scores'][name])
    write_json(RESULTS / 'selected_model.json', {'name': selection['selected'],
                    'artifact': f"results/artifacts/{selection['selected']}.joblib",
                    'selection': selection['criterion']})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['legacy', 'safe', 'all'], default='all')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri((ROOT / 'mlruns').as_uri())
    mlflow.set_experiment('ATP portfolio')
    write_json(RESULTS / 'data_sha256.json', fingerprints())
    failures = {}
    if args.stage in ('legacy', 'all'):
        for number in range(1, 7):
            name = f'model_{number}'
            try:
                if (RESULTS / 'predictions' / f'{name}.csv').exists():
                    continue
                m, data, features, factory = build(number)
                # Persist original features for audit, never commit cached binaries.
                (RESULTS / 'cache').mkdir(exist_ok=True)
                data.to_pickle(RESULTS / 'cache' / f'{name}.pkl')
                holdout(name, factory(), data, features, 'unchanged legacy feature/model functions; known leakage risk')
                if number == 2:
                    holdout('logistic_baseline', m.make_baseline_pipeline(), data,
                            m.BASELINE_NUM + m.BASELINE_CAT, 'legacy Model 2 logistic comparator')
                    test = data[data.year.between(2021, 2024)]
                    p = np.where(test.rank_diff < 0, 1., np.where(test.rank_diff > 0, 0., .5))
                    canonical(test, p).to_csv(RESULTS / 'predictions/rank_baseline.csv', index=False)
            except Exception:
                failures[name] = traceback.format_exc()
                print(failures[name], flush=True)
    write_json(RESULTS / f'failures_{args.stage}.json', failures)
    if args.stage in ('safe', 'all'):
        safe()


if __name__ == '__main__':
    main()
