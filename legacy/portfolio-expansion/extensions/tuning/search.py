"""Expanding-window selection, scoring one canonical perspective per match."""
import optuna
import numpy as np
from sklearn.metrics import brier_score_loss
from portfolio.predictions import canonical
from portfolio.common import VALIDATION_YEARS, SEED
from portfolio.models import make


def folds(data):
    for year in VALIDATION_YEARS:
        tr = data[data.year < year]
        va = data[(data.year == year) & (data.perspective == 0)]
        assert tr.date.max() < va.date.min()
        yield year, tr, va


def score(factory, train):
    out = []
    for year, tr, va in folds(train):
        m = factory()
        m.fit(tr, tr.label)
        valid = train[train.year == year]
        pred = canonical(valid, m.predict_proba(valid)[:, 1])
        out.append({'year': year, 'n_matches': len(pred),
                    'brier': float(brier_score_loss(pred.label, pred.probability))})
    return float(np.mean([x['brier'] for x in out])), out


def tune(train):
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial({'max_depth': 3, 'learning_rate': .05, 'n_estimators': 200})
    def objective(trial):
        params = dict(max_depth=trial.suggest_int('max_depth', 2, 5),
                      learning_rate=trial.suggest_float('learning_rate', .02, .1, log=True),
                      n_estimators=trial.suggest_categorical('n_estimators', [150, 200, 300]))
        value, detail = score(lambda: make(params=params), train)
        trial.set_user_attr('folds', detail)
        return value
    study.optimize(objective, n_trials=4)
    return study.best_params, study.trials_dataframe(), [dict(number=t.number, params=t.params,
            value=t.value, folds=t.user_attrs.get('folds')) for t in study.trials]
