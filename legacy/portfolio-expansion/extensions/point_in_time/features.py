"""Date-batched historical ATP features; never consume same-date outcomes.

ATP tourney_date is a tournament start date, not the actual match date. Batching
whole tournament-date groups is conservative and deliberately excludes intra-event history.
"""
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from portfolio.common import DATA

BASE = ['rank_diff', 'log_rank_diff', 'pts_diff', 'age_diff', 'elo_diff', 'h2h_rate', 'form_diff']
SERVE = ['first_in_diff', 'first_win_diff', 'second_win_diff']
CAT = ['surface', 'round', 'best_of', 'tourney_level']


def load_raw(require_ranks=True):
    frames = []
    for p in sorted((DATA / 'initial').glob('atp_matches_*.csv')):
        d = pd.read_csv(p, low_memory=False)
        d['match_id'] = [f'{p.stem}:{i}' for i in range(len(d))]
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    raw['date'] = pd.to_datetime(raw.tourney_date.astype(str), format='%Y%m%d', errors='coerce')
    required = ['date', 'winner_id', 'loser_id'] + (['winner_rank', 'loser_rank'] if require_ranks else [])
    raw = raw.dropna(subset=required)
    return raw.sort_values(['date', 'match_id']).reset_index(drop=True)


def serve_counts(row, prefix):
    cols = ['svpt', '1stIn', '1stWon', '2ndWon']
    vals = np.array([getattr(row, f'{prefix}_{s}', np.nan) for s in cols], dtype=float)
    if not np.isfinite(vals).all() or vals[0] <= 0:
        return None
    if not (0 <= vals[2] <= vals[1] <= vals[0] and 0 <= vals[3] <= vals[0]-vals[1]):
        return None
    return vals


def rates(history):
    if not history:
        return np.array([np.nan] * 3)
    a = np.sum([x[1] for x in history], axis=0)
    return np.array([a[1]/a[0], a[2]/a[1] if a[1] else np.nan,
                     a[3]/(a[0]-a[1]) if a[0] > a[1] else np.nan])


def build(raw):
    elo = defaultdict(lambda: 1500.)
    form = defaultdict(lambda: deque(maxlen=20))
    history = defaultdict(lambda: deque(maxlen=50))
    h2h = defaultdict(int)
    latest = {}
    rows = []
    for date, group in raw.groupby('date', sort=True):
        for r in group.itertuples(index=False):
            w, l = r.winner_id, r.loser_id
            total = h2h[w, l] + h2h[l, w]
            h = h2h[w, l] / total if total else .5
            serve = rates(history[w]) - rates(history[l])
            sources = [latest[p] for p in (w, l) if p in latest]
            # All data-derived historical signals have this upper source-date bound.
            source_max = max(sources) if sources else pd.NaT
            for flip in (False, True):
                s = -1 if flip else 1
                rows.append(dict(match_id=r.match_id, date=date, year=date.year,
                    player1=r.loser_name if flip else r.winner_name,
                    player2=r.winner_name if flip else r.loser_name,
                    perspective=int(flip), label=int(not flip),
                    rank_diff=s*(r.winner_rank-r.loser_rank),
                    log_rank_diff=s*(np.log1p(r.winner_rank)-np.log1p(r.loser_rank)),
                    pts_diff=s*(r.winner_rank_points-r.loser_rank_points),
                    age_diff=s*(r.winner_age-r.loser_age), elo_diff=s*(elo[w]-elo[l]),
                    h2h_rate=1-h if flip else h,
                    form_diff=s*((np.mean(form[w]) if form[w] else .5)-
                                 (np.mean(form[l]) if form[l] else .5)),
                    first_in_diff=s*serve[0], first_win_diff=s*serve[1], second_win_diff=s*serve[2],
                    surface=str(r.surface).lower(), round=str(r.round).lower(),
                    best_of=str(r.best_of), tourney_level=str(r.tourney_level).lower(),
                    source_max_date=source_max))
        # Outcomes become available only after every feature in the date group is built.
        for r in group.itertuples(index=False):
            w, l = r.winner_id, r.loser_id
            expected = 1/(1+10**((elo[l]-elo[w])/400))
            delta = 32*(1-expected)
            elo[w] += delta
            elo[l] -= delta
            h2h[w, l] += 1
            form[w].append(1.)
            form[l].append(0.)
            for p, prefix in ((w, 'w'), (l, 'l')):
                counts = serve_counts(r, prefix)
                if counts is not None:
                    history[p].append((date, counts))
                latest[p] = date
    return pd.DataFrame(rows)
