"""Audit static supplied serve estimates against observed ATP aggregate ratios.

These ATP files are the observed reference, not a claim of complete career truth.
Ratios pool numerators/denominators and exclude incomplete/invalid match-stat rows.
"""
import numpy as np
import pandas as pd
from extensions.point_in_time.features import load_raw, serve_counts
from portfolio.common import DATA, RESULTS, write_json

STATS = ['first_serve_pct', 'first_serve_win_pct', 'second_serve_win_pct']
MATERIAL_PP = 5.


def main():
    from collections import defaultdict
    values = defaultdict(list)
    raw = load_raw(require_ranks=False)
    for r in raw.itertuples(index=False):
        for name, prefix in ((r.winner_name, 'w'), (r.loser_name, 'l')):
            counts = serve_counts(r, prefix)
            if counts is not None:
                values[name].append(counts)
    rows = []
    for name, samples in values.items():
        v = np.sum(samples, axis=0)
        rows.append(dict(player_name=name, observed_matches=len(samples),
                         first_serve_pct=100*v[1]/v[0],
                         first_serve_win_pct=100*v[2]/v[1] if v[1] else np.nan,
                         second_serve_win_pct=100*v[3]/(v[0]-v[1]) if v[0]>v[1] else np.nan))
    truth = pd.DataFrame(rows)
    supplied = pd.concat([pd.read_csv(p) for p in sorted((DATA/'serve_data').glob('gemini_*.csv'))])
    # Keep each source row visible; do not silently average conflicting estimates.
    supplied['source_row'] = np.arange(len(supplied))
    merged = supplied.merge(truth, on='player_name', suffixes=('_supplied', '_atp'), validate='many_to_one')
    stats = {}
    for stat in STATS:
        x = pd.to_numeric(merged[f'{stat}_supplied'], errors='coerce')
        y = merged[f'{stat}_atp']
        valid = x.notna() & y.notna()
        err = (x[valid]-y[valid]).abs()
        stats[stat] = dict(n_rows=int(valid.sum()), mae_percentage_points=float(err.mean()),
                           pearson_correlation=float(x[valid].corr(y[valid])),
                           material_fraction=float((err > MATERIAL_PP).mean()))
        merged[f'{stat}_absolute_error_pp'] = (x-y).abs()
    # Player-level material disagreement: any stat in any supplied row exceeds threshold.
    cols = [s+'_absolute_error_pp' for s in STATS]
    material = merged.set_index('player_name')[cols].max(axis=1).groupby(level=0).max() > MATERIAL_PP
    report = dict(scope='all available valid ATP serve-stat rows; no ranking-completeness filter', threshold_percentage_points=MATERIAL_PP, supplied_rows=len(supplied),
                  matched_rows=len(merged), matched_players=int(merged.player_name.nunique()),
                  supplied_players=int(supplied.player_name.nunique()),
                  material_player_fraction=float(material.mean()), stats=stats)
    merged.to_csv(RESULTS/'serve_audit_players.csv', index=False)
    write_json(RESULTS/'serve_audit.json', report)
    lines = ['# Supplied serve data audit', '',
             'gemini files may be LLM-sourced. ATP counts are pooled over the available match files; scope differs from undocumented career estimates.',
             'Missing-stat matches are excluded. Exact full-name joins only; unmatched players are not assumed accurate.', '',
             '| Stat | Matched rows | MAE (percentage points) | Pearson r | Material fraction |',
             '|---|---:|---:|---:|---:|']
    for stat, v in stats.items():
        lines.append(f"| {stat} | {v['n_rows']} | {v['mae_percentage_points']:.6f} | {v['pearson_correlation']:.6f} | {v['material_fraction']:.6f} |")
    lines += ['', f"Material means absolute error > {MATERIAL_PP:g} percentage points (declared threshold, not learned).",
              f"Matched {report['matched_players']} of {report['supplied_players']} supplied players; player-level any-stat material disagreement fraction: {report['material_player_fraction']:.6f}.",
              'No inference about deliberate fabrication or individual player health is supported.']
    (RESULTS/'serve_audit.md').write_text('\n'.join(lines)+'\n')
    print(report)


if __name__ == '__main__':
    main()
