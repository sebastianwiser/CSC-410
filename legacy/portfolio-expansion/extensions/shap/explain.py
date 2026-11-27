"""Explain the selected estimator on training-period rows, globally and by surface."""
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
import shap
from portfolio.common import RESULTS, SEED, write_json


def main():
    chosen = json.loads((RESULTS/'selected_model.json').read_text())
    bundle = joblib.load(RESULTS/'artifacts'/f"{chosen['name']}.joblib")
    model = bundle['model']
    data = pd.read_pickle(RESULTS/'cache/point_features.pkl')
    train = data[data.year.between(2014, 2020)].copy()
    # Choose each match view by ID hash, never by winner identity or label.
    flip = train.match_id.map(lambda mid: int(hashlib.sha256(mid.encode()).hexdigest()[-1], 16) % 2)
    train = train[train.perspective.eq(flip)]
    global_sample = train.sample(n=min(200, len(train)), random_state=SEED).assign(shap_group='global')
    surface_samples = [g.sample(n=min(40, len(g)), random_state=SEED).assign(shap_group=surface)
                       for surface, g in train.groupby('surface')]
    sample = pd.concat([global_sample, *surface_samples])
    pre, clf = model.named_steps['pre'], model.named_steps['clf']
    x = pre.transform(sample[bundle['features']])
    names = pre.get_feature_names_out()
    if hasattr(clf, 'get_booster'):
        explainer = shap.TreeExplainer(clf)
        values = explainer(x).values
        target = 'underlying oriented XGBoost raw margin (log odds); final probabilities also symmetrize player order'
    elif hasattr(clf, 'coef_'):
        explainer = shap.LinearExplainer(clf, pre.transform(train.sample(n=100, random_state=SEED)))
        values = explainer(x).values
        target = 'underlying oriented logistic raw margin; final probabilities also symmetrize player order'
    else:
        explainer = shap.Explainer(lambda z: clf.predict_proba(z)[:, 1],
                                  pre.transform(train.sample(n=20, random_state=SEED)), algorithm='permutation')
        values = explainer(x, max_evals=2*x.shape[1]+1).values
        target = 'underlying oriented probability; final probabilities also symmetrize player order'
    rows = []
    for surface in sample.shap_group.unique():
        mask = sample.shap_group.eq(surface).to_numpy()
        for feature, importance in zip(names, np.abs(values[mask]).mean(axis=0)):
            rows.append(dict(surface=surface, feature=feature, mean_absolute_shap=float(importance), n_rows=int(mask.sum())))
    pd.DataFrame(rows).to_csv(RESULTS/'shap_importance.csv', index=False)
    write_json(RESULTS/'shap_metadata.json', dict(model=chosen['name'], target=target,
               sample_rows=len(sample), global_rows=len(global_sample), period='training only',
               orientation='outcome-independent match-ID hash', global_sampling='uniform match sample', seed=SEED))
    lines = ['# SHAP interpretation', '', f"Selected model: {chosen['name']}. Explanation target: {target}.",
             'Training-only outcome-independent player orientation; uniform match sample for global importance, independent bounded samples per surface. No extra test calls. Values describe model associations, not causal effects.',
             'Rank/log-rank and Elo/form are correlated; attribution can shift between substitutes. Tree path dependence does not resolve confounding.',
             'Per-surface samples can be small and do not establish reliable surface-specific mechanisms.', '']
    frame = pd.DataFrame(rows)
    for surface in frame.surface.unique():
        lines += [f'## {surface}', '', '| Feature | Mean absolute SHAP |', '|---|---:|']
        for r in frame[frame.surface == surface].nlargest(8, 'mean_absolute_shap').itertuples():
            lines.append(f'| {r.feature} | {r.mean_absolute_shap:.6f} |')
        lines.append('')
    (RESULTS/'shap.md').write_text('\n'.join(lines))
    print(chosen['name'], len(sample), x.shape)


if __name__ == '__main__':
    main()
