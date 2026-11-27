"""Validated profile-to-matchup demo. Caller supplies verifiable pre-match profiles."""
from datetime import date
from functools import lru_cache
import json
import os
from typing import Literal
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from portfolio.common import ROOT


class Player(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    name: str = Field(min_length=1, max_length=120)
    rank: int = Field(ge=1, le=5000)
    points: float = Field(ge=0, le=50000)
    age: float = Field(ge=14, le=65)
    elo: float = Field(ge=0, le=4000)
    recent_win_rate: float = Field(ge=0, le=1)
    first_in_rate: float | None = Field(default=None, ge=0, le=1)
    first_win_rate: float | None = Field(default=None, ge=0, le=1)
    second_win_rate: float | None = Field(default=None, ge=0, le=1)


class Matchup(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    player1: Player
    player2: Player
    match_date: date
    history_through: date
    h2h_player1_win_rate: float = Field(ge=0, le=1)
    surface: Literal['hard', 'clay', 'grass', 'carpet']
    round: Literal['r128', 'r64', 'r32', 'r16', 'qf', 'sf', 'f', 'rr', 'br']
    best_of: Literal['3', '5']
    tourney_level: Literal['A', 'G', 'M', 'D', 'F', 'C', 'S']

    @model_validator(mode='after')
    def chronology(self):
        if self.history_through >= self.match_date:
            raise ValueError('history_through must be strictly before match_date')
        if self.player1.name.strip().casefold() == self.player2.name.strip().casefold():
            raise ValueError('Players must differ')
        return self


def features(req):
    a, b = req.player1, req.player2
    row = dict(rank_diff=a.rank-b.rank, log_rank_diff=np.log1p(a.rank)-np.log1p(b.rank),
               pts_diff=a.points-b.points, age_diff=a.age-b.age, elo_diff=a.elo-b.elo,
               h2h_rate=req.h2h_player1_win_rate, form_diff=a.recent_win_rate-b.recent_win_rate,
               surface=req.surface, round=req.round, best_of=req.best_of,
               tourney_level=req.tourney_level.lower())
    for target, source in [('first_in_diff', 'first_in_rate'), ('first_win_diff', 'first_win_rate'),
                           ('second_win_diff', 'second_win_rate')]:
        x, y = getattr(a, source), getattr(b, source)
        row[target] = x-y if x is not None and y is not None else np.nan
    reverse = row.copy()
    for key in list(row):
        if key.endswith('_diff'):
            reverse[key] = -row[key]
    reverse['h2h_rate'] = 1-row['h2h_rate']
    return pd.DataFrame([row, reverse])


@lru_cache
def load_bundle():
    selection = json.loads((ROOT/'results/selected_model.json').read_text())
    path = os.environ.get('ATP_MODEL_PATH', str(ROOT/selection['artifact']))
    return joblib.load(path)


app = FastAPI(title='ATP matchup portfolio demo', description='Caller-supplied pre-match profiles; no live player data retrieval.')


@app.get('/health')
def health():
    try:
        bundle = load_bundle()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise HTTPException(503, 'Train the selected model first') from exc
    return {'status': 'ready', 'model': bundle['name']}


@app.post('/predict')
def predict(req: Matchup):
    bundle = load_bundle()
    frame = features(req)
    if any(c in bundle['features'] for c in ('first_in_diff', 'first_win_diff', 'second_win_diff')):
        if frame[['first_in_diff', 'first_win_diff', 'second_win_diff']].isna().any().any():
            raise HTTPException(422, 'This model requires historical serve rates for both players')
    probs = bundle['model'].predict_proba(frame[bundle['features']])[:, 1]
    p = float(.5*(probs[0]+1-probs[1]))
    return {'model': bundle['name'], 'player1': req.player1.name, 'player2': req.player2.name,
            'player1_win_probability': p, 'player2_win_probability': 1-p,
            'input_provenance': 'caller-supplied pre-match profiles; dates are assertions, not independently verified'}
