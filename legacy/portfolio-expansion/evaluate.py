"""Evaluate saved predictions only. Never fits a model or reopens holdout features."""
import json
import re
import unicodedata
import numpy as np
import pandas as pd
from portfolio.common import RESULTS, write_json
from portfolio.metrics import metrics, bootstrap_mean, paired


def name_key(name):
    text = ''.join(c for c in unicodedata.normalize('NFKD', str(name)) if not unicodedata.combining(c))
    parts = text.lower().replace('.', '').split()
    if len(parts[-1]) <= 2:  # processed surname + initials
        surname, initial = ' '.join(parts[:-1]), parts[-1][0]
    else:
        surname, initial = ' '.join(parts[1:]), parts[0][0]
    return re.sub('[^a-z]', '', surname) + ':' + initial


def alignment(pred, reference):
    """Conservative one-to-one same-pair/surface match within a tournament fortnight.

    Reject ambiguities in either direction rather than guessing event identity.
    Same-source IDs are exact. Cross-source dates differ: ATP files use event starts.
    """
    a, b = pred.copy(), reference.copy()
    if set(a.match_id) == set(b.match_id):
        return a, {'method': 'exact source match IDs', 'matched': len(a), 'unmatched': 0}
    for d in (a, b):
        d['key1'] = d.player1.map(name_key)
        d['key2'] = d.player2.map(name_key)
        d['pair'] = [tuple(sorted(x)) for x in zip(d.key1, d.key2)]
        d['date'] = pd.to_datetime(d.date)
    candidates = a.merge(b, on=['pair', 'surface'], suffixes=('_a', '_b'))
    candidates['gap'] = (candidates.date_a-candidates.date_b).dt.days.abs()
    candidates = candidates[candidates.gap <= 14]
    counts_a = candidates.match_id_a.value_counts()
    counts_b = candidates.match_id_b.value_counts()
    good = candidates[candidates.match_id_a.map(counts_a).eq(1) & candidates.match_id_b.map(counts_b).eq(1)]
    lookup = good.set_index('match_id_a')
    a = a[a.match_id.isin(lookup.index)].copy()
    mapped = lookup.loc[a.match_id]
    flip = a.key1.to_numpy() != mapped.key1_b.to_numpy()
    a.loc[flip, 'label'] = 1-a.loc[flip, 'label']
    a.loc[flip, 'probability'] = 1-a.loc[flip, 'probability']
    a['match_id'] = mapped.match_id_b.to_numpy()
    return a, {'method': 'unique same-player-pair/surface within date tolerance; inspect alignment.csv',
               'matched': len(a), 'unmatched': len(pred)-len(a),
               'max_date_gap_days': int(good.gap.max()) if len(good) else None,
               'pairs': good[['match_id_a', 'match_id_b', 'gap']].to_dict('records')}


def main():
    predictions = {p.stem: pd.read_csv(p) for p in sorted((RESULTS/'predictions').glob('*.csv'))}
    summary, comparisons, mappings = {}, {}, []
    for name, p in predictions.items():
        assert p.match_id.is_unique
        assert p.probability.between(0, 1).all()
        result = metrics(p.label.to_numpy(), p.probability.to_numpy())
        correct = (p.probability.to_numpy() >= .5) == p.label.to_numpy()
        result.update(n_matches=len(p), accuracy_ci95=bootstrap_mean(correct))
        summary[name] = result
        for ref in ('model_2', 'logistic_baseline'):
            if ref not in predictions:
                continue
            aligned, audit = alignment(p, predictions[ref])
            for row in audit.pop('pairs', []):
                mappings.append(dict(model=name, reference=ref, **row))
            comparisons[f'{name} vs {ref}'] = {**paired(aligned, predictions[ref]), 'alignment': audit}
    write_json(RESULTS/'metrics.json', summary)
    write_json(RESULTS/'comparisons.json', comparisons)
    pd.DataFrame(mappings).to_csv(RESULTS/'alignment.csv', index=False)
    lines = ['# Holdout ablation', '',
             'Computed from saved, symmetrized per-match probabilities. Each physical source match is scored once.',
             'Models using different source files have different cohorts; paired comparisons use only confidently aligned matches.',
             'Legacy rows retain full-period feature leakage and non-temporal isotonic calibration. They are descriptive reproductions, not leakage-free estimates.', '',
             '| Model | Test matches | Accuracy | Accuracy bootstrap CI | ROC AUC | Brier |',
             '|---|---:|---:|---|---:|---:|']
    for name, r in summary.items():
        lo, hi = r['accuracy_ci95']
        lines.append(f"| {name} | {r['n_matches']} | {r['accuracy']:.6f} | [{lo:.6f}, {hi:.6f}] | {r['roc_auc']:.6f} | {r['brier']:.6f} |")
    lines += ['', '## Paired accuracy comparisons', '',
              'Percentile paired match bootstrap; exact two-sided binomial McNemar test on discordant matches.',
              'Intervals assume independent matches; repeated players/tournaments violate that assumption. No multiple-comparison adjustment.', '',
              '| Comparison | Aligned matches | Difference | Bootstrap CI | McNemar p |',
              '|---|---:|---:|---|---:|']
    for name, r in comparisons.items():
        if not r['n']:
            lines.append(f'| {name} | 0 | unavailable | unavailable | unavailable |')
            continue
        lo, hi = r['ci95']
        lines.append(f"| {name} | {r['n']} | {r['accuracy_difference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {r['mcnemar_p']:.6g} |")
    lines += ['', 'Rank baseline probabilities are hard decisions with ties at neutral probability; its Brier is therefore not calibrated.',
              'The split, seed, bootstrap budget and fold definitions are executable constants in portfolio/.',
              'Full precision metrics, discordant counts, and alignment exclusions are in the adjacent JSON/CSV artifacts.']
    (RESULTS/'ablation.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
