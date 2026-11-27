#!/usr/bin/env python3
"""Check the classroom repository and saved results using only the standard library.

This never trains a model, calls an AI service, or predicts on the holdout again.
"""
import ast
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'legacy/portfolio-expansion'
MODELS = [
    ('model_1', 'Standalone logistic model'),
    ('logistic_baseline', 'Logistic comparator on processed data'),
    ('rank_baseline', 'Rank-only comparator'),
    ('model_2', 'XGBoost with Elo and form'),
    ('model_3', 'Add supplied serve statistics'),
    ('model_4', 'Add injury reports'),
    ('model_5', 'Add playstyle categories'),
    ('model_6', 'Combined model retained in the repository'),
]


def load(name):
    return json.loads((ARCHIVE / 'results' / name).read_text())


def results_table():
    metrics = load('metrics.json')
    lines = ['| Experiment | Test matches | Accuracy | ROC AUC | Brier |',
             '|---|---:|---:|---:|---:|']
    for key, title in MODELS:
        m = metrics[key]
        lines.append(f"| {title} | {m['n_matches']:,} | {100*m['accuracy']:.2f}% | {m['roc_auc']:.4f} | {m['brier']:.4f} |")
    return '\n'.join(lines)


def comparisons_table():
    pairs = load('comparisons.json')
    lines = ['| Addition versus base XGBoost | Accuracy change (percentage points) | Paired bootstrap interval | McNemar p |',
             '|---|---:|---|---:|']
    for key, title in MODELS:
        if key not in {'model_3', 'model_4', 'model_5', 'model_6'}:
            continue
        result = pairs[f'{key} vs model_2']
        low, high = result['ci95']
        lines.append(f"| {title} | {100*result['accuracy_difference']:+.3f} | [{100*low:+.3f}, {100*high:+.3f}] | {result['mcnemar_p']:.4f} |")
    return '\n'.join(lines)


def verify_predictions(key, expected):
    with (ARCHIVE / 'results/predictions' / f'{key}.csv').open(newline='') as source:
        rows = list(csv.DictReader(source))
    ids = [r['match_id'] for r in rows]
    if len(ids) != len(set(ids)) or len(ids) != expected['n_matches']:
        raise ValueError(f'{key}: invalid unique-match count')
    labels = [int(r['label']) for r in rows]
    probabilities = [float(r['probability']) for r in rows]
    if not all(y in (0, 1) and 0 <= p <= 1 for y, p in zip(labels, probabilities)):
        raise ValueError(f'{key}: invalid saved labels/probabilities')
    if not all(2021 <= int(r['date'][:4]) <= 2024 for r in rows):
        raise ValueError(f'{key}: prediction outside the recorded holdout period')
    n = len(rows)
    accuracy = sum((p >= .5) == y for y, p in zip(labels, probabilities)) / n
    brier = math.fsum((p-y)**2 for y, p in zip(labels, probabilities)) / n
    # Mann-Whitney AUC with average ranks for exact probability ties.
    positives = sum(labels)
    rank, positive_rank_sum = 1, 0.
    for _, tied in itertools.groupby(sorted(zip(probabilities, labels)), key=lambda r: r[0]):
        group = list(tied)
        average_rank = rank + (len(group)-1)/2
        positive_rank_sum += average_rank * sum(y for _, y in group)
        rank += len(group)
    auc = (positive_rank_sum - positives*(positives+1)/2) / (positives*(n-positives))
    for metric, observed in [('accuracy', accuracy), ('roc_auc', auc), ('brier', brier)]:
        if not math.isclose(observed, expected[metric], abs_tol=1e-6):
            raise ValueError(f'{key}: {metric} disagrees with saved predictions')


def check_links(path):
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
        if re.match(r'^[a-z]+:', target) or target.startswith('#'):
            continue
        local = target.split('#')[0]
        if not (path.parent / local).exists():
            raise ValueError(f'{path.relative_to(ROOT)}: missing link {target}')


def main():
    manifest = json.loads((ROOT/'docs/project-manifest.json').read_text())
    for name, digest in manifest['files'].items():
        path = ROOT/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Original source or presentation changed: {name}')
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=name)
    metrics = load('metrics.json')
    for key, _ in MODELS:
        verify_predictions(key, metrics[key])
    note = (ROOT/'docs/results.md').read_text()
    if results_table() not in note or comparisons_table() not in note:
        raise ValueError('Results note does not match saved numerical evidence')
    for path in [ROOT/'README.md', ROOT/'legacy/README.md', *sorted((ROOT/'docs').glob('*.md'))]:
        check_links(path)
    pins = [line for line in (ROOT/'requirements.txt').read_text().splitlines() if line and not line.startswith('#')]
    if not all(re.fullmatch(r'[A-Za-z0-9_-]+==[0-9.]+', line) for line in pins):
        raise ValueError('Class dependency versions must be pinned')
    print('Original model/SQL sources and presentation: unchanged')
    print('Saved match counts, accuracy, ROC AUC and Brier: verified')
    print('Results note and local documentation links: verified')
    print('No model training, new holdout inference or AI requests were performed.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        sys.exit(str(exc))
