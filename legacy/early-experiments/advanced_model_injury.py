#!/usr/bin/env python3
"""
advanced_model_injury.py

Extends the advanced XGBoost model (Elo + rolling stats) with player injury
data from injury_reports/*.json.

Trains two models side-by-side on the same train/test split:
  Baseline  — XGBoost + Elo + rolling stats (no injury features)
  Injury    — same features PLUS 8 pre-match injury diff columns

Produces 8 comparison charts in plots_injury/ and a metric summary table.

New injury features (all expressed as p1 − p2 diffs for symmetry):
  inj_has_diff              — 1 if player had a recent injury, 0 otherwise
  inj_days_diff             — days since most recent injury (large = healthy)
  inj_count_diff            — number of injuries in lookback window
  inj_joint_diff            — any joint injury in window (knee/elbow/wrist/ankle)
  inj_chronic_diff          — any chronic (ongoing) injury in window
  inj_retired_diff          — any match-retirement-due-to-injury in window
  inj_post_return_days_diff — days since player returned from most recent injury
                              (low = just came back = still in post-injury dip)
  inj_recently_returned_diff— 1 if player returned from injury within last 90 days
                              captures hesitancy/under-match-fitness after return

Key insight: players often show a long-term performance dip AFTER returning from
injury — not just while injured. These last two features capture that effect by
parsing estimated_recovery_time strings to compute when a player likely returned.

Usage:
    python advanced_model_injury.py
    python advanced_model_injury.py --injury-dir injury_reports --lookback-days 365
    python advanced_model_injury.py --use-odds --tune

Requirements: scikit-learn, xgboost, pandas, numpy, matplotlib, joblib
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────

ELO_BASE   = 1500.0
ELO_K      = 32.0
ELO_K_SURF = 24.0
FORM_SHORT = 5
FORM_LONG  = 20

STATIC_NUMERIC = ["rank_diff", "pts_diff", "rank_ratio", "log_rank_diff"]

DYNAMIC_NUMERIC = [
    "elo_diff",
    "elo_surf_diff",
    "form_short_diff",
    "form_long_diff",
    "surf_form_diff",
    "h2h_rate_p1",
    "h2h_n",
    "days_rest_diff",
    "experience_diff",
]

NUMERIC_COLS     = STATIC_NUMERIC + DYNAMIC_NUMERIC
CATEGORICAL_COLS = ["surface", "series", "court", "round", "best_of"]
ODDS_COLS        = ["odds_diff", "odds_ratio", "implied_prob_diff"]

INJURY_DIFF_COLS = [
    "inj_has_diff",
    "inj_days_diff",
    "inj_count_diff",
    "inj_joint_diff",
    "inj_chronic_diff",
    "inj_retired_diff",
    "inj_post_return_days_diff",    # days since return (low = just came back)
    "inj_recently_returned_diff",   # 1 if returned within last 90 days
]

# How many days counts as "recently returned" (post-injury vulnerability window)
POST_RETURN_WINDOW_DAYS = 90


@dataclass
class Config:
    data_path: str
    injury_dir: str
    lookback_days: int
    train_year_start: int
    train_year_end: int
    test_year_start: int
    test_year_end: int
    use_odds: bool
    tune: bool
    plots_dir: str


# ─── Injury database ──────────────────────────────────────────────────────────

def _parse_injury_date(s: str | None) -> pd.Timestamp | None:
    if not s or str(s).strip().lower() == "unknown":
        return None
    try:
        return pd.to_datetime(s)
    except Exception:
        return None


def _parse_recovery_days(s: str | None) -> float | None:
    """
    Convert estimated_recovery_time strings to days (midpoint of range).
    Returns None if unknown or unparseable.

    Examples:
      '1-2 weeks'           → 10.5
      '4-6 weeks'           → 35.0
      '5-7 months'          → 180.0
      'days to 2 weeks'     → 7.0
      'days to weeks'       → 7.0
      'same day to a few days' → 2.0
    """
    import re as _re
    if not s or str(s).strip().lower() in ("unknown", ""):
        return None
    s = str(s).strip().lower().replace("about ", "").strip()

    def _to_days(val: float, unit: str) -> float:
        if "month" in unit:
            return val * 30.0
        if "week" in unit:
            return val * 7.0
        return val  # days

    # "X-Y unit" or "X to Y unit"  (e.g. "1-2 weeks", "2 to 6 weeks")
    m = _re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(days?|weeks?|months?)", s)
    if m:
        lo, hi, unit = float(m.group(1)), float(m.group(2)), m.group(3)
        return _to_days((lo + hi) / 2.0, unit)

    # "X unit" — single value
    m = _re.search(r"(\d+(?:\.\d+)?)\s*(days?|weeks?|months?)", s)
    if m:
        return _to_days(float(m.group(1)), m.group(2))

    # Prose fallbacks (no numbers)
    if "same day" in s:
        return 2.0      # "same day to a few / several days"
    if "day" in s and "week" in s:
        return 7.0      # "days to weeks"
    if "day" in s:
        return 2.0
    if "week" in s:
        return 14.0

    return None


def _norm_name(name: str) -> str:
    """Normalize by removing punctuation/case — works for both CSV and JSON formats."""
    return str(name).strip().lower().replace(".", "").replace("-", " ").replace("  ", " ").strip()


def _csv_key(full_name: str) -> str | None:
    """
    Convert 'Firstname Lastname' (JSON format) to the normalized CSV-format key.
    The ATP CSV stores names as 'Lastname F.' so 'Grigor Dimitrov' → 'dimitrov g'.
    This lets us match injury DB entries against CSV player names.
    """
    parts = full_name.strip().split()
    if len(parts) < 2:
        return None
    initial  = parts[0][0].lower()
    lastname = " ".join(parts[1:]).lower().replace("-", " ")
    return f"{lastname} {initial}"


def load_injury_db(injury_dir: str) -> dict:
    """
    Load all *.json files in injury_dir.
    Returns dict: normalized_player_name → list of injury dicts.
    Indexes each player under BOTH the full-name key ('grigor dimitrov')
    and the CSV-format key ('dimitrov g') so that lookups from either
    naming convention work transparently.
    """
    db: dict[str, list] = {}
    files = sorted(_glob.glob(os.path.join(injury_dir, "*.json")))
    if not files:
        print(f"  Warning: no JSON files found in '{injury_dir}'")
        return db

    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            raw = json.load(f)
        for player_name, pdata in raw.get("players", {}).items():
            key = _norm_name(player_name)
            if key not in db:
                db[key] = []
            for inj in pdata.get("injuries", []):
                d = _parse_injury_date(inj.get("injury_date"))
                if d is None:
                    continue
                retired_raw = str(inj.get("retired_during_match_due_to_injury", "no"))
                db[key].append({
                    "date":          d,
                    "recovery_days": _parse_recovery_days(
                        inj.get("estimated_recovery_time")
                    ),
                    "joint":   bool(inj.get("joint_injury", False)),
                    "chronic": inj.get("acute_or_chronic") == "chronic",
                    "acute":   inj.get("acute_or_chronic") == "acute",
                    "retired": retired_raw.strip().lower() in ("yes", "true", "1"),
                })
            # Also store under CSV-format key so 'Dimitrov G.' → finds 'Grigor Dimitrov'
            ck = _csv_key(player_name)
            if ck and ck not in db:
                db[ck] = db[key]   # same list reference — same player

    # Sort each player's list by date for efficient lookups
    for key in db:
        db[key].sort(key=lambda x: x["date"])

    total = sum(len(v) for v in db.values())
    print(f"  Injury DB: {len(db):,} players, {total:,} injury records from {len(files)} files")
    return db


def _player_injury_feats(
    player: str,
    match_dt: pd.Timestamp,
    db: dict,
    lookback: int,
) -> dict:
    """
    Pre-match injury features for one player.

    Splits injuries in the lookback window into two groups:
      active   — player is still within their estimated recovery period
                 (or recovery time is unknown — conservative assumption)
      returned — player's estimated return date has passed, but they are
                 still in the post-injury vulnerability window

    Returns 8 features (6 existing + 2 post-recovery):
      has, days, count, joint, chronic, retired  — based on active injuries
      post_return_days    — days since most recent return (low = just came back)
      recently_returned   — 1 if any return within POST_RETURN_WINDOW_DAYS days
    """
    key  = _norm_name(player)
    injs = db.get(key, [])

    no_inj = dict(
        has=0.0, days=float(lookback), count=0.0,
        joint=0.0, chronic=0.0, retired=0.0,
        post_return_days=float(lookback), recently_returned=0.0,
    )

    if not injs:
        return no_inj

    cutoff = match_dt - pd.Timedelta(days=lookback)
    window = [i for i in injs if cutoff <= i["date"] < match_dt]

    if not window:
        return no_inj

    active   = []  # still in recovery period at match_date
    returned = []  # estimated return has passed

    for inj in window:
        rec = inj["recovery_days"]
        if rec is not None:
            est_return = inj["date"] + pd.Timedelta(days=rec)
            if match_dt < est_return:
                active.append(inj)          # still recovering
            else:
                days_since = (match_dt - est_return).days
                returned.append(days_since)
        else:
            active.append(inj)              # unknown recovery — treat as still active

    # ── Features from active (currently injured) injuries ────────────────────
    if active:
        most_recent = active[-1]["date"]    # list is date-sorted, last = newest
        has_feat     = 1.0
        days_feat    = float((match_dt - most_recent).days)
        count_feat   = float(len(active))
        joint_feat   = float(any(i["joint"]   for i in active))
        chronic_feat = float(any(i["chronic"] for i in active))
        retired_feat = float(any(i["retired"] for i in active))
    else:
        has_feat = days_feat = count_feat = joint_feat = chronic_feat = retired_feat = 0.0
        if not returned:
            return no_inj   # nothing at all in window

    # ── Post-return features ──────────────────────────────────────────────────
    if returned:
        min_days_since_return  = float(min(returned))
        recently_returned_feat = float(min_days_since_return <= POST_RETURN_WINDOW_DAYS)
        post_return_feat       = min_days_since_return
    else:
        recently_returned_feat = 0.0
        post_return_feat       = float(lookback)   # no recent return → neutral

    return dict(
        has              = has_feat,
        days             = days_feat if active else float(lookback),
        count            = count_feat,
        joint            = joint_feat,
        chronic          = chronic_feat,
        retired          = retired_feat,
        post_return_days = post_return_feat,
        recently_returned= recently_returned_feat,
    )


def add_injury_features(data: pd.DataFrame, injury_db: dict, lookback: int) -> pd.DataFrame:
    """
    For every row in data, compute pre-match injury features for p1 and p2,
    then add INJURY_DIFF_COLS (p1 − p2 differences) to data.

    Requires columns: p1_name, p2_name, match_dt.
    """
    if not injury_db:
        for col in INJURY_DIFF_COLS:
            data[col] = 0.0
        # still add individual columns needed for coverage chart
        for prefix in ("p1_inj_has", "p2_inj_has"):
            data[prefix] = 0.0
        return data

    # Cache to avoid recomputing the same (player, date) pair twice
    cache: dict[tuple, dict] = {}

    def _lookup(player, dt):
        key = (_norm_name(player), dt)
        if key not in cache:
            cache[key] = _player_injury_feats(player, dt, injury_db, lookback)
        return cache[key]

    feat_keys = ["has", "days", "count", "joint", "chronic", "retired",
                 "post_return_days", "recently_returned"]
    f1 = {k: [] for k in feat_keys}
    f2 = {k: [] for k in feat_keys}

    for row in data.itertuples(index=False):
        dt = pd.Timestamp(row.match_dt)
        r1 = _lookup(row.p1_name, dt)
        r2 = _lookup(row.p2_name, dt)
        for k in feat_keys:
            f1[k].append(r1[k])
            f2[k].append(r2[k])

    col_map  = {k: k for k in feat_keys}   # internal key == stored column suffix
    diff_map = dict(zip(feat_keys, INJURY_DIFF_COLS))

    for k in feat_keys:
        arr1 = np.array(f1[k], dtype=float)
        arr2 = np.array(f2[k], dtype=float)
        data[f"p1_inj_{col_map[k]}"] = arr1
        data[f"p2_inj_{col_map[k]}"] = arr2
        data[diff_map[k]]             = arr1 - arr2

    covered      = int((data["p1_inj_has"] > 0).sum() + (data["p2_inj_has"] > 0).sum())
    post_covered = int(
        (data["p1_inj_recently_returned"] > 0).sum()
        + (data["p2_inj_recently_returned"] > 0).sum()
    )
    print(f"  Injury features attached — {covered:,}/{2*len(data):,} "
          f"player-rows with active injury in {lookback}-day window")
    print(f"  Post-return features    — {post_covered:,}/{2*len(data):,} "
          f"player-rows in {POST_RETURN_WINDOW_DAYS}-day post-return window")
    return data


# ─── Dynamic feature computation (Elo, form, H2H) ────────────────────────────

def _elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _form(dq: deque, n: int | None = None) -> float:
    lst = list(dq)[-n:] if n else list(dq)
    return float(np.mean(lst)) if lst else 0.5


def compute_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk matches chronologically; attach pre-match Elo, form, H2H, rest, experience.
    No look-ahead leakage — features for match i use only matches 0..i-1.
    """
    df = df.sort_values("Date").reset_index(drop=True)

    elo_all  = defaultdict(lambda: ELO_BASE)
    elo_surf = defaultdict(lambda: defaultdict(lambda: ELO_BASE))
    res_all  = defaultdict(lambda: deque(maxlen=FORM_LONG))
    res_surf = defaultdict(lambda: defaultdict(lambda: deque(maxlen=FORM_LONG)))
    last_dt  = {}
    exp_cnt  = defaultdict(int)
    h2h_wins = defaultdict(int)

    feat_rows = []
    for row in df.itertuples(index=False):
        p1   = getattr(row, "Player_1", None)
        p2   = getattr(row, "Player_2", None)
        if not isinstance(p1, str) or not isinstance(p2, str):
            feat_rows.append({})
            continue

        win  = getattr(row, "Winner", None)
        surf = str(getattr(row, "Surface", "unknown") or "unknown").strip().lower()
        date = getattr(row, "Date", None)
        p1_won = (win == p1)

        # ── pre-match snapshots ───────────────────────────────────────────
        p1_elo  = elo_all[p1];        p2_elo  = elo_all[p2]
        p1_eloS = elo_surf[p1][surf]; p2_eloS = elo_surf[p2][surf]
        p1_fS = _form(res_all[p1], FORM_SHORT); p2_fS = _form(res_all[p2], FORM_SHORT)
        p1_fL = _form(res_all[p1]);             p2_fL = _form(res_all[p2])
        p1_sf = _form(res_surf[p1][surf]);      p2_sf = _form(res_surf[p2][surf])

        h2h_p1    = h2h_wins[(p1, p2)]
        h2h_p2    = h2h_wins[(p2, p1)]
        h2h_total = h2h_p1 + h2h_p2
        h2h_rate  = h2h_p1 / h2h_total if h2h_total > 0 else 0.5

        p1_days = (date - last_dt[p1]).days if p1 in last_dt else 30
        p2_days = (date - last_dt[p2]).days if p2 in last_dt else 30

        feat_rows.append({
            "p1_elo_pre":      p1_elo,
            "p2_elo_pre":      p2_elo,
            "p1_eloS_pre":     p1_eloS,
            "p2_eloS_pre":     p2_eloS,
            "p1_fS_pre":       p1_fS,
            "p2_fS_pre":       p2_fS,
            "p1_fL_pre":       p1_fL,
            "p2_fL_pre":       p2_fL,
            "p1_sf_pre":       p1_sf,
            "p2_sf_pre":       p2_sf,
            "p1_h2h_rate_pre": h2h_rate,
            "h2h_n_pre":       float(h2h_total),
            "p1_days_pre":     float(p1_days),
            "p2_days_pre":     float(p2_days),
            "p1_exp_pre":      float(exp_cnt[p1]),
            "p2_exp_pre":      float(exp_cnt[p2]),
        })

        # ── post-match updates ────────────────────────────────────────────
        S  = 1.0 if p1_won else 0.0
        E  = _elo_expected(p1_elo, p2_elo)
        elo_all[p1] += ELO_K * (S - E)
        elo_all[p2] += ELO_K * ((1 - S) - (1 - E))
        ES = _elo_expected(p1_eloS, p2_eloS)
        elo_surf[p1][surf] += ELO_K_SURF * (S - ES)
        elo_surf[p2][surf] += ELO_K_SURF * ((1 - S) - (1 - ES))
        res_all[p1].append(1 if p1_won else 0)
        res_all[p2].append(0 if p1_won else 1)
        res_surf[p1][surf].append(1 if p1_won else 0)
        res_surf[p2][surf].append(0 if p1_won else 1)
        h2h_wins[(p1, p2) if p1_won else (p2, p1)] += 1
        if date is not None:
            last_dt[p1] = date
            last_dt[p2] = date
        exp_cnt[p1] += 1
        exp_cnt[p2] += 1

    feat_df = pd.DataFrame(feat_rows, index=df.index)
    return pd.concat([df, feat_df], axis=1)


# ─── Dataset construction ─────────────────────────────────────────────────────

def _make_rows(df: pd.DataFrame, flip: bool) -> pd.DataFrame:
    """Build one side of the symmetric dataset; adds p1_name, p2_name, match_dt."""
    def _v(col_a, col_b):
        col = col_b if flip else col_a
        return df[col].values if col in df.columns else np.full(len(df), np.nan)

    label    = (1 - df["p1_wins"]).values if flip else df["p1_wins"].values
    h2h_rate = (
        1.0 - df["p1_h2h_rate_pre"].values if flip else df["p1_h2h_rate_pre"].values
    ) if "p1_h2h_rate_pre" in df.columns else np.full(len(df), 0.5)

    return pd.DataFrame({
        "tourney_date": df["tourney_date"].values,
        "match_dt":     df["Date"].values,                           # for injury lookups
        "p1_name":      df["Player_2" if flip else "Player_1"].values,
        "p2_name":      df["Player_1" if flip else "Player_2"].values,
        # static
        "p1_rank": _v("Rank_1", "Rank_2"),
        "p2_rank": _v("Rank_2", "Rank_1"),
        "p1_pts":  _v("Pts_1",  "Pts_2"),
        "p2_pts":  _v("Pts_2",  "Pts_1"),
        "p1_odds": _v("Odd_1",  "Odd_2"),
        "p2_odds": _v("Odd_2",  "Odd_1"),
        # dynamic — swap p1/p2 when flipped
        "p1_elo":  _v("p1_elo_pre",  "p2_elo_pre"),
        "p2_elo":  _v("p2_elo_pre",  "p1_elo_pre"),
        "p1_eloS": _v("p1_eloS_pre", "p2_eloS_pre"),
        "p2_eloS": _v("p2_eloS_pre", "p1_eloS_pre"),
        "p1_fS":   _v("p1_fS_pre",   "p2_fS_pre"),
        "p2_fS":   _v("p2_fS_pre",   "p1_fS_pre"),
        "p1_fL":   _v("p1_fL_pre",   "p2_fL_pre"),
        "p2_fL":   _v("p2_fL_pre",   "p1_fL_pre"),
        "p1_sf":   _v("p1_sf_pre",   "p2_sf_pre"),
        "p2_sf":   _v("p2_sf_pre",   "p1_sf_pre"),
        "h2h_rate":  h2h_rate,
        "h2h_n_raw": df["h2h_n_pre"].values if "h2h_n_pre" in df.columns else np.zeros(len(df)),
        "p1_days": _v("p1_days_pre", "p2_days_pre"),
        "p2_days": _v("p2_days_pre", "p1_days_pre"),
        "p1_exp":  _v("p1_exp_pre",  "p2_exp_pre"),
        "p2_exp":  _v("p2_exp_pre",  "p1_exp_pre"),
        # categorical
        "surface": df["Surface"].values,
        "series":  df["Series"].values if "Series" in df.columns else np.full(len(df), "unknown"),
        "court":   df["Court"].values  if "Court"  in df.columns else np.full(len(df), "unknown"),
        "round":   df["Round"].values,
        "best_of": df["best_of"].values,
        "label":   label,
    })


def build_dataset(df: pd.DataFrame, injury_db: dict, cfg: Config) -> pd.DataFrame:
    df = df.rename(columns={"Best of": "best_of"}).copy()

    for col in ["Rank_1", "Rank_2", "Pts_1", "Pts_2", "Odd_1", "Odd_2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(-1, np.nan)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    df["tourney_date"] = df["Date"].dt.strftime("%Y%m%d").astype(int)
    df["p1_wins"]      = (df["Player_1"] == df["Winner"]).astype(int)
    df = df.dropna(subset=["Rank_1", "Rank_2", "Surface", "Round"]).reset_index(drop=True)

    for col in ["Surface", "Series", "Court", "Round"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower()
    df["best_of"] = df["best_of"].astype(str).str.strip()

    print("  Computing dynamic features (Elo, form, H2H) ...")
    df = compute_dynamic_features(df)

    data = pd.concat([_make_rows(df, False), _make_rows(df, True)], ignore_index=True)

    # Static diffs
    data["rank_diff"]     = data["p1_rank"] - data["p2_rank"]
    data["pts_diff"]      = data["p1_pts"].fillna(0) - data["p2_pts"].fillna(0)
    data["rank_ratio"]    = data["p1_rank"] / data["p2_rank"].clip(lower=1)
    data["log_rank_diff"] = np.log1p(data["p1_rank"]) - np.log1p(data["p2_rank"])

    # Dynamic diffs
    data["elo_diff"]        = data["p1_elo"]  - data["p2_elo"]
    data["elo_surf_diff"]   = data["p1_eloS"] - data["p2_eloS"]
    data["form_short_diff"] = data["p1_fS"]   - data["p2_fS"]
    data["form_long_diff"]  = data["p1_fL"]   - data["p2_fL"]
    data["surf_form_diff"]  = data["p1_sf"]   - data["p2_sf"]
    data["h2h_rate_p1"]     = data["h2h_rate"]
    data["h2h_n"]           = data["h2h_n_raw"]
    data["days_rest_diff"]  = data["p1_days"] - data["p2_days"]
    data["experience_diff"] = data["p1_exp"]  - data["p2_exp"]

    if cfg.use_odds:
        data["odds_diff"]         = data["p1_odds"] - data["p2_odds"]
        data["odds_ratio"]        = data["p1_odds"] / data["p2_odds"].clip(lower=0.01)
        data["implied_prob_diff"] = (
            1.0 / data["p1_odds"].clip(lower=0.01)
            - 1.0 / data["p2_odds"].clip(lower=0.01)
        )

    # Injury features
    print("  Computing injury features ...")
    data = add_injury_features(data, injury_db, cfg.lookback_days)
    return data


def _year(series: pd.Series) -> pd.Series:
    return series.astype(str).str[:4].astype(int)


def make_splits(data: pd.DataFrame, cfg: Config):
    years   = _year(data["tourney_date"])
    tr_mask = (years >= cfg.train_year_start) & (years <= cfg.train_year_end)
    te_mask = (years >= cfg.test_year_start)  & (years <= cfg.test_year_end)

    train, test = data[tr_mask], data[te_mask]
    if train.empty or test.empty:
        raise ValueError("Empty train/test split — check year ranges.")

    base_cols = NUMERIC_COLS + (ODDS_COLS if cfg.use_odds else []) + CATEGORICAL_COLS
    inj_cols  = NUMERIC_COLS + (ODDS_COLS if cfg.use_odds else []) + INJURY_DIFF_COLS + CATEGORICAL_COLS

    y_tr = train["label"]
    y_te = test["label"]

    meta = test[["tourney_date", "surface", "series", "round",
                 "p1_inj_has", "p2_inj_has"]].copy()
    meta.index = y_te.index

    return (
        train[base_cols], test[base_cols],
        train[inj_cols],  test[inj_cols],
        y_tr, y_te, meta,
    )


# ─── Model ────────────────────────────────────────────────────────────────────

def _make_preprocessor(num_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imp",    SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, CATEGORICAL_COLS),
    ])


def _build_xgb_params():
    return dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def train_model(X_tr, y_tr, num_cols: list[str], tune: bool):
    pre  = _make_preprocessor(num_cols)
    xgb  = XGBClassifier(**_build_xgb_params())
    pipe = Pipeline([("pre", pre), ("clf", xgb)])

    if tune:
        param_dist = {
            "clf__n_estimators":     [200, 400, 600],
            "clf__max_depth":        [3, 4, 5],
            "clf__learning_rate":    [0.01, 0.05, 0.1],
            "clf__subsample":        [0.7, 0.8, 0.9],
            "clf__colsample_bytree": [0.7, 0.8, 0.9],
            "clf__min_child_weight": [10, 20, 30],
        }
        tscv   = TimeSeriesSplit(n_splits=4)
        search = RandomizedSearchCV(
            pipe, param_dist, n_iter=20, cv=tscv,
            scoring="roc_auc", n_jobs=-1, random_state=42, verbose=1,
        )
        search.fit(X_tr, y_tr)
        best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        xgb_best  = XGBClassifier(**{**_build_xgb_params(), **best_params})
        best_pipe = Pipeline([("pre", pre), ("clf", xgb_best)])
        model = CalibratedClassifierCV(best_pipe, cv=3, method="isotonic")
    else:
        model = CalibratedClassifierCV(pipe, cv=3, method="isotonic")

    model.fit(X_tr, y_tr)
    return model


def evaluate(model, X_te, y_te):
    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return (
        accuracy_score(y_te, preds),
        roc_auc_score(y_te, probs),
        brier_score_loss(y_te, probs),
        probs,
    )


def _extract_importances(model):
    try:
        if hasattr(model, "calibrated_classifiers_"):
            pipes = [cc.estimator for cc in model.calibrated_classifiers_]
        elif hasattr(model, "estimator"):
            pipes = [model.estimator]
        else:
            pipes = [model]
        imps  = [p.named_steps["clf"].feature_importances_
                 for p in pipes if hasattr(p, "named_steps")]
        if not imps:
            return None
        avg   = np.mean(imps, axis=0)
        names = list(pipes[0].named_steps["pre"].get_feature_names_out())
        return names, avg
    except Exception as e:
        print(f"  Warning: feature importance extraction failed: {e}")
        return None


# ─── Shared plot helper ───────────────────────────────────────────────────────

def _savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ─── Comparison charts (8 total) ──────────────────────────────────────────────

def plot_comparison_roc(y_te, probs_b, probs_i, out_dir):
    """Chart 1 — ROC curves for both models on the same axes."""
    fpr_b, tpr_b, _ = roc_curve(y_te, probs_b)
    fpr_i, tpr_i, _ = roc_curve(y_te, probs_i)
    auc_b = roc_auc_score(y_te, probs_b)
    auc_i = roc_auc_score(y_te, probs_i)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_b, tpr_b, lw=2, color="steelblue",
            label=f"Baseline      (AUC = {auc_b:.4f})")
    ax.plot(fpr_i, tpr_i, lw=2, color="darkorange", linestyle="--",
            label=f"+ Injury data (AUC = {auc_i:.4f})")
    ax.plot([0, 1], [0, 1], ":", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Baseline vs Injury Model")
    ax.legend()
    _savefig(os.path.join(out_dir, "comparison_roc.png"))


def plot_comparison_calibration(y_te, probs_b, probs_i, out_dir):
    """Chart 2 — Calibration curves for both models."""
    fop_b, mpv_b = calibration_curve(y_te, probs_b, n_bins=10)
    fop_i, mpv_i = calibration_curve(y_te, probs_i, n_bins=10)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(mpv_b, fop_b, "o-", color="steelblue",   label="Baseline")
    ax.plot(mpv_i, fop_i, "s-", color="darkorange",  label="+ Injury data", linestyle="--")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve — Baseline vs Injury Model")
    ax.legend()
    _savefig(os.path.join(out_dir, "comparison_calibration.png"))


def plot_comparison_yearly(data, model_b, model_i, cfg, out_dir):
    """Chart 3 — Yearly accuracy for both models + rank baseline."""
    use_odds  = cfg.use_odds
    base_cols = NUMERIC_COLS + (ODDS_COLS if use_odds else []) + CATEGORICAL_COLS
    inj_cols  = NUMERIC_COLS + (ODDS_COLS if use_odds else []) + INJURY_DIFF_COLS + CATEGORICAL_COLS

    years = _year(data["tourney_date"])
    rows  = []
    for yr in sorted(years.unique()):
        mask   = years == yr
        X_b    = data[mask][base_cols]
        X_i    = data[mask][inj_cols]
        y_yr   = data[mask]["label"]
        if len(y_yr) < 50:
            continue
        acc_b  = accuracy_score(y_yr, (model_b.predict_proba(X_b)[:, 1] >= 0.5).astype(int))
        acc_i  = accuracy_score(y_yr, (model_i.predict_proba(X_i)[:, 1] >= 0.5).astype(int))
        base   = accuracy_score(y_yr, (X_b["rank_diff"] < 0).astype(int))
        rows.append(dict(year=yr, base_model=acc_b, inj_model=acc_i,
                         rank_baseline=base,
                         split="test" if yr >= cfg.test_year_start else "train"))

    res = pd.DataFrame(rows)
    tr  = res[res.split == "train"]
    te  = res[res.split == "test"]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(tr["year"], tr["base_model"],  "o-",  color="steelblue",  alpha=0.55,
            label="Baseline (train)")
    ax.plot(te["year"], te["base_model"],  "o-",  color="steelblue",
            label="Baseline (test)", lw=2)
    ax.plot(tr["year"], tr["inj_model"],   "s--", color="darkorange", alpha=0.55,
            label="+ Injury (train)")
    ax.plot(te["year"], te["inj_model"],   "s--", color="darkorange",
            label="+ Injury (test)", lw=2)
    ax.plot(res["year"], res["rank_baseline"], ":", color="gray", lw=1.5,
            label="Rank baseline")
    ax.axvline(cfg.train_year_end + 0.5, color="black", linestyle=":", lw=1,
               label="Train / Test split")
    ax.set_xlabel("Year");  ax.set_ylabel("Accuracy")
    ax.set_title("Yearly Accuracy — Baseline vs Injury Model")
    ax.legend(ncol=2, fontsize=9)
    ax.set_ylim(0.45, 0.82)
    _savefig(os.path.join(out_dir, "comparison_yearly.png"))


def plot_comparison_confidence(y_te, probs_b, probs_i, out_dir):
    """Chart 4 — Accuracy by confidence bucket, side-by-side bars."""
    bins   = np.linspace(0, 0.5, 11)
    labels = [f"{int(bins[k]*100)}-{int(bins[k+1]*100)}%" for k in range(len(bins)-1)]

    def _bucket_acc(probs):
        df = pd.DataFrame({"prob": probs, "label": y_te.values})
        df["conf"]    = np.abs(df["prob"] - 0.5)
        df["correct"] = ((df["prob"] >= 0.5).astype(int) == df["label"]).astype(int)
        df["bin"]     = pd.cut(df["conf"], bins=bins, labels=labels, include_lowest=True)
        return df.groupby("bin", observed=True)["correct"].mean()

    grp_b = _bucket_acc(probs_b)
    grp_i = _bucket_acc(probs_i)
    x     = np.arange(len(labels))
    w     = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, grp_b, w, color="steelblue",  alpha=0.85, label="Baseline")
    ax.bar(x + w/2, grp_i, w, color="darkorange", alpha=0.85, label="+ Injury data")
    ax.axhline(0.5, color="gray", linestyle="--")
    ax.set_xticks(x);  ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("Confidence Bucket (|P − 0.5|)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Confidence — Baseline vs Injury Model")
    ax.legend()
    _savefig(os.path.join(out_dir, "comparison_confidence.png"))


def plot_injury_feature_importance(model_i, out_dir):
    """Chart 5 — Top 30 feature importances with injury features highlighted."""
    result = _extract_importances(model_i)
    if result is None:
        return
    names, importances = result
    df   = pd.DataFrame({"feature": names, "importance": importances})
    df   = df.sort_values("importance", ascending=False)
    df.to_csv(os.path.join(out_dir, "feature_importance_injury.csv"), index=False)

    top    = df.head(30).sort_values("importance", ascending=True)
    colors = ["tomato" if "inj_" in f else "steelblue" for f in top["feature"]]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(top["feature"], top["importance"], color=colors)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("Top 30 Feature Importances — Injury Model")
    legend_handles = [
        mpatches.Patch(facecolor="tomato",    label="Injury diff features"),
        mpatches.Patch(facecolor="steelblue", label="Baseline features"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")
    _savefig(os.path.join(out_dir, "feature_importance_injury_model.png"))


def plot_injury_features_only(model_i, out_dir):
    """Chart 6 — Importance of each injury diff feature in isolation."""
    result = _extract_importances(model_i)
    if result is None:
        return
    names, importances = result
    df  = pd.DataFrame({"feature": names, "importance": importances})
    inj = df[df["feature"].str.contains("inj_")].sort_values("importance", ascending=True)
    if inj.empty:
        print("  No injury features found — skipping chart 6")
        return

    # Clean up feature names for readability
    pretty = {
        "num__inj_has_diff":              "Has injury diff",
        "num__inj_days_diff":             "Days since injury diff",
        "num__inj_count_diff":            "Injury count diff",
        "num__inj_joint_diff":            "Joint injury diff",
        "num__inj_chronic_diff":          "Chronic injury diff",
        "num__inj_retired_diff":          "Retired-in-match diff",
        "num__inj_post_return_days_diff": "Days since return diff",
        "num__inj_recently_returned_diff":"Recently returned diff (≤90d)",
    }
    inj["label"] = inj["feature"].map(pretty).fillna(inj["feature"])

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(inj["label"], inj["importance"], color="tomato")
    # annotate percentage of total importance
    total = importances.sum()
    for bar, imp in zip(bars, inj["importance"]):
        ax.text(bar.get_width() + total * 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{imp/total*100:.2f}%", va="center", fontsize=9)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("Injury Feature Importances\n(as % of total model gain)")
    _savefig(os.path.join(out_dir, "injury_features_impact.png"))


def plot_accuracy_injured_vs_healthy(y_te, probs_b, probs_i, meta, out_dir):
    """Chart 7 — Accuracy split by whether either player had a recent injury."""
    df = meta.copy()
    df["label"]       = y_te.values
    df["correct_b"]   = ((probs_b >= 0.5).astype(int) == df["label"]).astype(int)
    df["correct_i"]   = ((probs_i >= 0.5).astype(int) == df["label"]).astype(int)

    c1 = df["p1_inj_has"] > 0
    c2 = df["p2_inj_has"] > 0
    df["match_type"] = "Both Healthy"
    df.loc[ c1 & ~c2, "match_type"] = "P1 Injured Only"
    df.loc[~c1 &  c2, "match_type"] = "P2 Injured Only"
    df.loc[ c1 &  c2, "match_type"] = "Both Injured"

    order = ["Both Healthy", "P1 Injured Only", "P2 Injured Only", "Both Injured"]
    order = [c for c in order if c in df["match_type"].values]

    grp = (df.groupby("match_type")[["correct_b", "correct_i"]]
             .agg(["mean", "count"])
             .loc[order])

    x     = np.arange(len(order))
    w     = 0.35
    acc_b = [grp.loc[c, ("correct_b", "mean")]  for c in order]
    acc_i = [grp.loc[c, ("correct_i", "mean")]  for c in order]
    cnt   = [int(grp.loc[c, ("correct_b", "count")]) for c in order]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, acc_b, w, color="steelblue",  alpha=0.85, label="Baseline")
    ax.bar(x + w/2, acc_i, w, color="darkorange", alpha=0.85, label="+ Injury data")
    ax.axhline(0.5, color="gray", linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={cnt[j]:,})" for j, c in enumerate(order)], fontsize=10)
    ax.set_ylabel("Accuracy");  ax.set_ylim(0.45, 0.80)
    ax.set_title("Accuracy by Match Injury Status — Baseline vs Injury Model")
    ax.legend()
    _savefig(os.path.join(out_dir, "accuracy_injured_vs_healthy.png"))


def plot_injury_coverage(data: pd.DataFrame, lookback_days: int, out_dir: str):
    """Chart 8 — Fraction of match rows that have injury data, by year."""
    if "p1_inj_has" not in data.columns:
        return

    years = _year(data["tourney_date"])
    rows  = []
    for yr in sorted(years.unique()):
        sub  = data[years == yr]
        n    = len(sub)
        any_ = ((sub["p1_inj_has"] > 0) | (sub["p2_inj_has"] > 0)).sum()
        p1_  = (sub["p1_inj_has"] > 0).sum()
        p2_  = (sub["p2_inj_has"] > 0).sum()
        rows.append(dict(year=yr,
                         any_pct =  any_ / n * 100,
                         p1_pct  =  p1_  / n * 100,
                         p2_pct  =  p2_  / n * 100,
                         n=n))

    res = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(res["year"], res["any_pct"], alpha=0.2, color="tomato")
    ax.plot(res["year"], res["any_pct"], "o-", color="tomato",    lw=2,
            label="At least one player injured")
    ax.plot(res["year"], res["p1_pct"],  "s--", color="steelblue", lw=1.5,
            label="P1 injured")
    ax.plot(res["year"], res["p2_pct"],  "^--", color="darkorange", lw=1.5,
            label="P2 injured")
    ax.set_xlabel("Year");  ax.set_ylabel(f"% of rows with injury in {lookback_days}-day window")
    ax.set_title("Injury Data Coverage by Year")
    ax.legend()
    _savefig(os.path.join(out_dir, "injury_coverage.png"))


def plot_all(model_b, model_i, X_te_b, X_te_i, y_te,
             probs_b, probs_i, meta, data, cfg):
    out = cfg.plots_dir
    os.makedirs(out, exist_ok=True)

    tasks = [
        ("comparison_roc.png",             lambda: plot_comparison_roc(y_te, probs_b, probs_i, out)),
        ("comparison_calibration.png",     lambda: plot_comparison_calibration(y_te, probs_b, probs_i, out)),
        ("comparison_yearly.png",          lambda: plot_comparison_yearly(data, model_b, model_i, cfg, out)),
        ("comparison_confidence.png",      lambda: plot_comparison_confidence(y_te, probs_b, probs_i, out)),
        ("feature_importance_injury_model.png", lambda: plot_injury_feature_importance(model_i, out)),
        ("injury_features_impact.png",     lambda: plot_injury_features_only(model_i, out)),
        ("accuracy_injured_vs_healthy.png",lambda: plot_accuracy_injured_vs_healthy(y_te, probs_b, probs_i, meta, out)),
        ("injury_coverage.png",            lambda: plot_injury_coverage(data, cfg.lookback_days, out)),
    ]
    for name, fn in tasks:
        print(f"  {name}")
        fn()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Advanced XGBoost ATP model — baseline vs injury-enhanced comparison."
    )
    p.add_argument("--data-path",        default="atp_tennis.csv")
    p.add_argument("--injury-dir",            default="injury_reports")
    p.add_argument("--lookback-days",         type=int, default=365)
    p.add_argument("--post-return-window",    type=int, default=90,
                   help="Days after estimated return still counted as 'recently returned'")
    p.add_argument("--train-year-start", type=int, default=2000)
    p.add_argument("--train-year-end",   type=int, default=2020)
    p.add_argument("--test-year-start",  type=int, default=2021)
    p.add_argument("--test-year-end",    type=int, default=2026)
    p.add_argument("--use-odds",         action="store_true")
    p.add_argument("--tune",             action="store_true")
    p.add_argument("--plots-dir",        default="plots_injury")
    a = p.parse_args()
    return Config(
        data_path=a.data_path,
        injury_dir=a.injury_dir,
        lookback_days=a.lookback_days,
        train_year_start=a.train_year_start,
        train_year_end=a.train_year_end,
        test_year_start=a.test_year_start,
        test_year_end=a.test_year_end,
        use_odds=a.use_odds,
        tune=a.tune,
        plots_dir=a.plots_dir,
    )


def main():
    cfg = parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading {cfg.data_path} ...")
    raw = pd.read_csv(cfg.data_path, low_memory=False)
    print(f"  {len(raw):,} matches loaded  ({raw['Date'].min()} → {raw['Date'].max()})")

    print(f"\nLoading injury database from '{cfg.injury_dir}/' ...")
    injury_db = load_injury_db(cfg.injury_dir)

    # ── Build symmetric dataset ───────────────────────────────────────────────
    print("\nBuilding dataset (2× symmetric) ...")
    data = build_dataset(raw, injury_db, cfg)
    print(f"  {len(data):,} rows")

    # ── Train / test split ────────────────────────────────────────────────────
    X_tr_b, X_te_b, X_tr_i, X_te_i, y_tr, y_te, meta = make_splits(data, cfg)
    print(f"\n  Train: {len(X_tr_b):,}  |  Test: {len(X_te_b):,}")

    # ── Train baseline model (no injury features) ─────────────────────────────
    base_num_cols = NUMERIC_COLS + (ODDS_COLS if cfg.use_odds else [])
    tune_note     = " (with tuning)" if cfg.tune else ""
    print(f"\nTraining BASELINE model{tune_note} ...")
    model_b = train_model(X_tr_b, y_tr, base_num_cols, cfg.tune)

    # ── Train injury-enhanced model ───────────────────────────────────────────
    inj_num_cols = NUMERIC_COLS + (ODDS_COLS if cfg.use_odds else []) + INJURY_DIFF_COLS
    print(f"Training INJURY model{tune_note} ...")
    model_i = train_model(X_tr_i, y_tr, inj_num_cols, cfg.tune)

    # ── Evaluate both ─────────────────────────────────────────────────────────
    acc_b, auc_b, brier_b, probs_b = evaluate(model_b, X_te_b, y_te)
    acc_i, auc_i, brier_i, probs_i = evaluate(model_i, X_te_i, y_te)
    rank_baseline = accuracy_score(y_te, (X_te_b["rank_diff"] < 0).astype(int))

    print(f"\n{'═'*60}")
    print(f"{'Metric':<20}{'Baseline':>12}{'+ Injury':>12}{'Δ':>10}")
    print(f"{'─'*60}")
    print(f"{'Accuracy':<20}{acc_b:>12.4f}{acc_i:>12.4f}{acc_i-acc_b:>+10.4f}")
    print(f"{'ROC AUC':<20}{auc_b:>12.4f}{auc_i:>12.4f}{auc_i-auc_b:>+10.4f}")
    print(f"{'Brier Score':<20}{brier_b:>12.4f}{brier_i:>12.4f}{brier_i-brier_b:>+10.4f}  (↓ better)")
    print(f"{'Rank baseline':<20}{rank_baseline:>12.4f}{'—':>12}{'—':>10}")
    print(f"{'═'*60}\n")

    # ── Save models ───────────────────────────────────────────────────────────
    joblib.dump(model_b, "model_injury_baseline.joblib")
    joblib.dump(model_i, "model_injury_enhanced.joblib")
    print("Saved: model_injury_baseline.joblib  |  model_injury_enhanced.joblib")

    # ── Generate 8 comparison charts ──────────────────────────────────────────
    print(f"\nGenerating plots in '{cfg.plots_dir}/' ...")
    plot_all(model_b, model_i, X_te_b, X_te_i, y_te,
             probs_b, probs_i, meta, data, cfg)
    print("Done.")


if __name__ == "__main__":
    main()
