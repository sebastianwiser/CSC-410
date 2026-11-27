import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from extensions.point_in_time.features import build, BASE, SERVE, CAT
from portfolio.common import ROOT, RESULTS
from portfolio.models import make
from portfolio.predictions import canonical
from portfolio.metrics import metrics, paired
from extensions.tuning.search import folds


@pytest.fixture
def raw():
    rows = []
    for i, day in enumerate(['20170101', '20180101', '20180101', '20190101', '20200101', '20210101']):
        row = dict(match_id=f'match-{i}', date=pd.Timestamp(day),
                   winner_id=1, loser_id=2, winner_name='Alpha', loser_name='Beta',
                   winner_rank=10, loser_rank=30, winner_rank_points=2000, loser_rank_points=900,
                   winner_age=25., loser_age=27., surface='Hard', round='R32', best_of=3, tourney_level='A')
        row.update(w_svpt=100, w_1stIn=60, w_1stWon=40, w_2ndWon=20,
                   l_svpt=100, l_1stIn=50, l_1stWon=30, l_2ndWon=20)
        rows.append(row)
    return pd.DataFrame(rows)


def test_source_dates_strictly_precede_matches(raw):
    frame = build(raw)
    assert (frame.source_max_date.isna() | (frame.source_max_date < frame.date)).all()


def test_same_date_and_future_outcomes_cannot_change_features(raw):
    before = build(raw)
    altered = raw.copy()
    # A second match shares the target date. Its stats/outcome cannot affect target features.
    altered.loc[2:, ['w_1stWon', 'l_1stWon']] = [0, 50]
    altered.loc[2:, ['winner_id', 'loser_id']] = [2, 1]
    after = build(altered)
    target = before.match_id.eq('match-1')
    pd.testing.assert_frame_equal(before.loc[target, BASE+SERVE+CAT], after.loc[target, BASE+SERVE+CAT])


def test_truncating_future_preserves_all_prior_features(raw):
    full = build(raw)
    cut = raw.date.iloc[3]
    earlier = build(raw[raw.date < cut])
    pd.testing.assert_frame_equal(full[full.date < cut].reset_index(drop=True), earlier.reset_index(drop=True))


def test_walk_forward_never_touches_holdout(raw):
    train = build(raw[raw.date.dt.year <= 2020])
    for year, tr, va in folds(train):
        assert tr.date.max() < va.date.min()
        assert va.year.max() <= 2020
        assert set(tr.match_id).isdisjoint(va.match_id)


@pytest.mark.parametrize('kind', ['xgb', 'mlp'])
def test_same_seed_identical_predictions_and_metrics(raw, kind):
    data = build(raw)
    a, b = make(kind), make(kind)
    a.fit(data, data.label)
    b.fit(data, data.label)
    pa = canonical(data, a.predict_proba(data)[:, 1])
    pb = canonical(data, b.predict_proba(data)[:, 1])
    np.testing.assert_array_equal(pa.probability, pb.probability)
    assert metrics(pa.label.to_numpy(), pa.probability.to_numpy()) == metrics(pb.label.to_numpy(), pb.probability.to_numpy())


def test_one_prediction_per_match(raw):
    d = build(raw)
    p = canonical(d, np.where(d.label == 1, .7, .3))
    assert len(p) == raw.match_id.nunique()
    assert p.match_id.is_unique
    assert np.isclose(p.probability.where(p.label.eq(1), 1-p.probability), .7).all()


def test_self_comparison(raw):
    p = canonical(build(raw), np.tile([.7, .3], len(raw)))
    result = paired(p, p)
    assert result['accuracy_difference'] == 0
    assert result['ci95'] == [0, 0]
    assert result['mcnemar_p'] == 1


def test_api_smoke_and_validation():
    from extensions.api.app import app
    req = json.loads((ROOT/'extensions/api/example.json').read_text())
    client = TestClient(app)
    health = client.get('/health')
    assert health.status_code == 200, health.text
    response = client.post('/predict', json=req)
    assert response.status_code == 200, response.text
    p = response.json()['player1_win_probability']
    assert 0 <= p <= 1
    reversed_req = dict(req, player1=req['player2'], player2=req['player1'],
                        h2h_player1_win_rate=1-req['h2h_player1_win_rate'])
    q = client.post('/predict', json=reversed_req).json()['player1_win_probability']
    assert p+q == pytest.approx(1)
    req['history_through'] = req['match_date']
    assert client.post('/predict', json=req).status_code == 422
    req['history_through'] = '2024-01-09'
    req['player1']['rank'] = -1
    assert client.post('/predict', json=req).status_code == 422


def test_committed_predictions_unique_and_bounded():
    for path in (RESULTS/'predictions').glob('*.csv'):
        p = pd.read_csv(path)
        assert p.match_id.is_unique
        assert p.probability.between(0, 1).all()
        assert p.label.isin([0, 1]).all()
