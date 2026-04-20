#!/usr/bin/env python3
"""
model_2.py  —  XGBoost with Elo, Rolling Form, and Surface History
===================================================================
Improved version of model_1.py (Baseline Logistic Regression).

WHAT CHANGED vs MODEL 1          WHY IT HELPS
────────────────────────────────────────────────────────────────────────────
1. Algorithm: XGBoost             Captures non-linear interactions (e.g. Elo
   + isotonic probability         advantage matters more on grass than clay).
   calibration                    Calibration makes win probabilities reliable.

2. Extended history for Elo       67 k matches (2000-2026) from atp_tennis.csv
   warm-up                        give Elo ratings 14 years to converge before
                                  the training window. Removes cold-start noise.

3. Elo ratings (overall +         Real-time skill estimate updated after every
   surface-specific)              match. Tracks momentum that weekly ATP rank
                                  misses. Surface-Elo captures specialists.

4. Rolling form                   Short-term (last 5) captures hot/cold streaks.
   (short + long + surface)       Long-term (last 20) adds stability.

5. Form trend                     form_short - form_long: positive = improving.
   (NEW feature)                  Explicit momentum signal XGBoost uses directly.

6. Surface specialist gap         elo_surf - elo_all: how much more (or less)
   (NEW feature)                  suited is each player to THIS surface vs their
                                  general ability. Captures clay kings, grass
                                  servers who struggle on clay, etc.

7. Bayesian-smoothed H2H          (wins+1)/(total+2) instead of wins/total.
   (improved from v1)             Shrinks toward 0.5 with few meetings. Avoids
                                  overconfident 1.0 after a single H2H win.

8. H2H match count                h2h_n tells us how much to trust the H2H rate.

9. Rest & career experience       Days since last match, career match count.

10. Career surface win rate       From player_surface_results.csv. Long-run
    and match count per surface   surface affinity at a different time horizon
                                  than rolling surface Elo.

11. Player height + handedness    Joined from initial match files. Height
                                  matters on serve-heavy grass. Left-handers
                                  have documented serve advantages.

12. Tuned XGBoost hyperparams     More trees (600), deeper (5), lower learning
    (improved from v1)            rate (0.03), L1 regularisation — better
                                  bias-variance trade-off on tabular sports data.

Data sources:
  data/processed/atp_tennis.csv               — 67 k matches 2000-2026
  data/initial/atp_matches_YYYY.csv  (×11)    — player attrs: age, ht, hand
  data/initial/player_surface_results.csv     — career surface win rates

Feature groups (19 numeric + 7 categorical):
  A  Rank/skill:          rank_diff, pts_diff, rank_ratio, log_rank_diff
  B  Elo:                 elo_diff, elo_surf_diff, surface_specialist_diff
  C  Rolling form:        form_short_diff, form_long_diff, surf_form_diff, form_trend_diff
  D  H2H:                 h2h_rate, h2h_n
  E  Rest / experience:   days_rest_diff, experience_diff
  F  Player attributes:   age_diff, ht_diff
  G  Surface history:     surf_win_rate_diff, surf_exp_diff
  H  Categorical:         surface, series, court, round, best_of, p1_hand, p2_hand

Validation (same as model_1 for direct comparison):
  1. Expanding walk-forward  — test years 2016-2024 (reports Model 2 AND logistic baseline)
  2. Single holdout          — train 2014-2020, test 2021-2024

Outputs (models/model_2/plots/ — 14 charts):
  roc_curve.png                 walk_forward.png            accuracy_by_surface.png
  calibration.png               vs_baseline_roc.png         accuracy_by_confidence.png
  confusion_matrix.png          yearly_comparison.png       prob_histogram.png
  feature_importance.png        feature_groups.png
  elo_validation.png            accuracy_by_round.png
  rank_gap_accuracy.png

Usage:
  cd CSC410Pres/models/model_2
  python model_2.py
"""

from __future__ import annotations

import glob
import os
import re
import unicodedata
import warnings
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix,
                              roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(ROOT, "..", "..", "data")
INIT_DIR  = os.path.join(DATA_DIR, "initial")
PROC_DIR  = os.path.join(DATA_DIR, "processed")
PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Elo / form constants ───────────────────────────────────────────────────────
ELO_BASE   = 1500.0
ELO_K      = 32.0
ELO_K_SURF = 24.0
FORM_SHORT = 5
FORM_LONG  = 20

# ── Feature column lists ───────────────────────────────────────────────────────
STATIC_NUMERIC = ["rank_diff", "pts_diff", "rank_ratio", "log_rank_diff"]

DYNAMIC_NUMERIC = [
    # Elo (Group B)
    "elo_diff", "elo_surf_diff", "surface_specialist_diff",
    # Form (Group C)
    "form_short_diff", "form_long_diff", "surf_form_diff", "form_trend_diff",
    # H2H (Group D)
    "h2h_rate", "h2h_n",
    # Rest / experience (Group E)
    "days_rest_diff", "experience_diff",
]

ENRICHED_NUMERIC = [
    "age_diff", "ht_diff",            # Group F — player attributes
    "surf_win_rate_diff", "surf_exp_diff",  # Group G — career surface history
]

NUMERIC_COLS = STATIC_NUMERIC + DYNAMIC_NUMERIC + ENRICHED_NUMERIC
CAT_COLS     = ["surface", "series", "court", "round", "best_of", "p1_hand", "p2_hand"]

# Logistic baseline features (mirrors model_1 using atp_tennis.csv columns)
BASELINE_NUM = ["rank_diff", "log_rank_diff", "rank_ratio", "pts_diff", "age_diff", "h2h_rate"]
BASELINE_CAT = ["surface", "series", "round", "best_of"]

# ── Group colour map for plots ─────────────────────────────────────────────────
GROUP_COLORS = [
    ("num__elo",                   "darkorange"),
    ("num__surf_form",             "steelblue"),
    ("num__surface_specialist",    "darkorange"),
    ("num__form",                  "steelblue"),
    ("num__h2h",                   "seagreen"),
    ("num__rank",                  "mediumpurple"),
    ("num__pts",                   "mediumpurple"),
    ("num__log",                   "mediumpurple"),
    ("num__days",                  "dimgray"),
    ("num__exp",                   "dimgray"),
    ("num__age",                   "orchid"),
    ("num__ht",                    "orchid"),
    ("num__surf_win",              "teal"),
    ("num__surf_exp",              "teal"),
]

GROUP_LABELS = {
    "A — Rank/skill":         ["rank_diff", "pts_diff", "rank_ratio", "log_rank_diff"],
    "B — Elo":                ["elo_diff", "elo_surf_diff", "surface_specialist_diff"],
    "C — Rolling form":       ["form_short_diff", "form_long_diff",
                               "surf_form_diff", "form_trend_diff"],
    "D — H2H":                ["h2h_rate", "h2h_n"],
    "E — Rest/experience":    ["days_rest_diff", "experience_diff"],
    "F — Player attributes":  ["age_diff", "ht_diff"],
    "G — Surface history":    ["surf_win_rate_diff", "surf_exp_diff"],
}
GROUP_COLORS_BAR = {
    "A — Rank/skill":         "mediumpurple",
    "B — Elo":                "darkorange",
    "C — Rolling form":       "steelblue",
    "D — H2H":                "seagreen",
    "E — Rest/experience":    "dimgray",
    "F — Player attributes":  "orchid",
    "G — Surface history":    "teal",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _norm(name: str) -> str:
    s = str(name).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _form(dq: deque, n=None) -> float:
    lst = list(dq)[-n:] if n else list(dq)
    return float(np.mean(lst)) if lst else 0.5


def _feat_color(feat: str) -> str:
    for prefix, c in GROUP_COLORS:
        if feat.startswith(prefix):
            return c
    return "silver"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_processed(proc_dir: str) -> pd.DataFrame:
    path = os.path.join(proc_dir, "atp_tennis.csv")
    df = pd.read_csv(path, low_memory=False)
    if "Best of" in df.columns:
        df = df.rename(columns={"Best of": "best_of"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    df["year"] = df["Date"].dt.year
    for col in ["Rank_1", "Rank_2", "Pts_1", "Pts_2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(-1, np.nan)
    for col in ["Surface", "Series", "Court", "Round"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    df["best_of"] = df["best_of"].astype(str).str.strip()
    df = df.dropna(subset=["Rank_1", "Rank_2", "Surface", "Round"]).reset_index(drop=True)
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"  Processed: {len(df):,} matches  ({int(df['year'].min())}–{int(df['year'].max())})")
    return df


def load_initial_matches(init_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(init_dir, "atp_matches_*.csv")))
    if not files:
        raise FileNotFoundError(f"No atp_matches_YYYY.csv in {init_dir}")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df["match_date"] = pd.to_datetime(
        df["tourney_date"].astype(str).str[:8], format="%Y%m%d", errors="coerce"
    )
    df["year"] = df["match_date"].dt.year
    for col in ["winner_rank", "loser_rank", "winner_age", "loser_age", "winner_ht", "loser_ht"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"  Initial:   {len(df):,} matches  ({int(df['year'].min())}–{int(df['year'].max())})")
    return df


def load_surface_results(init_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(init_dir, "player_surface_results.csv"))
    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["surface"]     = df["surface"].astype(str).str.strip().str.lower()
    for col in ["wins", "losses", "matches"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if "matches" not in df.columns:
        df["matches"] = df.get("wins", 0) + df.get("losses", 0)
    df["win_rate"] = np.where(
        df["matches"] > 0,
        df.get("wins", df["matches"] * 0.5) / df["matches"], 0.5,
    )
    df["name_key"] = df["player_name"].map(_norm)
    print(f"  Surface results: {len(df):,} rows  ({df['name_key'].nunique():,} unique players)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  STATIC LOOKUPS
# ══════════════════════════════════════════════════════════════════════════════

def build_player_attrs(matches: pd.DataFrame) -> dict:
    winner = pd.DataFrame({
        "name_key":   matches["winner_name"].map(_norm),
        "hand":       matches.get("winner_hand", pd.Series("U", index=matches.index)),
        "ht":         matches.get("winner_ht",   pd.Series(np.nan, index=matches.index)),
        "birth_year": matches["year"] - matches["winner_age"],
    })
    loser = pd.DataFrame({
        "name_key":   matches["loser_name"].map(_norm),
        "hand":       matches.get("loser_hand", pd.Series("U", index=matches.index)),
        "ht":         matches.get("loser_ht",   pd.Series(np.nan, index=matches.index)),
        "birth_year": matches["year"] - matches["loser_age"],
    })
    df = pd.concat([winner, loser], ignore_index=True)
    df = df[df["name_key"] != ""]

    def _mode(s):
        m = s.dropna().mode()
        return str(m.iloc[0]) if len(m) > 0 else "U"

    result: dict[str, dict] = {}
    for key, grp in df.groupby("name_key"):
        result[key] = {
            "hand":       _mode(grp["hand"]),
            "ht":         float(grp["ht"].median())         if grp["ht"].notna().any()         else np.nan,
            "birth_year": float(grp["birth_year"].median()) if grp["birth_year"].notna().any() else np.nan,
        }
    print(f"  Player attrs: {len(result):,} players indexed")
    return result


def build_surface_lookup(surf_df: pd.DataFrame) -> dict:
    lookup: dict[tuple, dict] = {}
    for _, row in surf_df.iterrows():
        lookup[(row["name_key"], row["surface"])] = {
            "win_rate":  float(row["win_rate"]),
            "n_matches": int(row["matches"]),
        }
    print(f"  Surface lookup: {len(lookup):,} (player, surface) entries")
    return lookup


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DYNAMIC FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def compute_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk matches chronologically. Attaches pre-match Elo, form, H2H, rest, exp.
    Uses Bayesian-smoothed H2H: (wins+1)/(total+2) — avoids 0/1 extremes.
    No look-ahead: features for match i use only matches 0..i-1.
    """
    elo_all  = defaultdict(lambda: ELO_BASE)
    elo_surf = defaultdict(lambda: defaultdict(lambda: ELO_BASE))
    res_all  = defaultdict(lambda: deque(maxlen=FORM_LONG))
    res_surf = defaultdict(lambda: defaultdict(lambda: deque(maxlen=FORM_LONG)))
    last_dt: dict = {}
    exp_cnt  = defaultdict(int)
    h2h_wins = defaultdict(int)

    feat_rows = []
    for row in df.itertuples(index=False):
        p1 = getattr(row, "Player_1", None)
        p2 = getattr(row, "Player_2", None)
        if not isinstance(p1, str) or not isinstance(p2, str):
            feat_rows.append({})
            continue

        win    = getattr(row, "Winner", None)
        surf   = str(getattr(row, "Surface", "unknown") or "unknown").lower()
        date   = getattr(row, "Date", None)
        p1_won = (win == p1)

        p1_elo  = elo_all[p1];          p2_elo  = elo_all[p2]
        p1_eloS = elo_surf[p1][surf];   p2_eloS = elo_surf[p2][surf]

        p1_fS = _form(res_all[p1], FORM_SHORT);  p2_fS = _form(res_all[p2], FORM_SHORT)
        p1_fL = _form(res_all[p1]);               p2_fL = _form(res_all[p2])
        p1_sf = _form(res_surf[p1][surf]);         p2_sf = _form(res_surf[p2][surf])

        h2h_p1    = h2h_wins[(p1, p2)]
        h2h_p2    = h2h_wins[(p2, p1)]
        h2h_total = h2h_p1 + h2h_p2
        # Bayesian smoothing: prior = 0.5, pseudo-count = 2
        h2h_rate  = (h2h_p1 + 1.0) / (h2h_total + 2.0)

        p1_days = float((date - last_dt[p1]).days) if p1 in last_dt else 30.0
        p2_days = float((date - last_dt[p2]).days) if p2 in last_dt else 30.0

        feat_rows.append({
            "p1_elo_pre":      p1_elo,  "p2_elo_pre":      p2_elo,
            "p1_eloS_pre":     p1_eloS, "p2_eloS_pre":     p2_eloS,
            "p1_fS_pre":       p1_fS,   "p2_fS_pre":       p2_fS,
            "p1_fL_pre":       p1_fL,   "p2_fL_pre":       p2_fL,
            "p1_sf_pre":       p1_sf,   "p2_sf_pre":       p2_sf,
            "p1_h2h_rate_pre": h2h_rate,
            "h2h_n_pre":       float(h2h_total),
            "p1_days_pre":     p1_days, "p2_days_pre":     p2_days,
            "p1_exp_pre":      float(exp_cnt[p1]),
            "p2_exp_pre":      float(exp_cnt[p2]),
        })

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

    return pd.concat([df, pd.DataFrame(feat_rows, index=df.index)], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DATASET CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def _make_rows(df, flip, player_attrs, surf_lookup):
    def _v(a, b):
        col = b if flip else a
        return df[col].values if col in df.columns else np.full(len(df), np.nan)

    p1_names = df["Player_2" if flip else "Player_1"].values
    p2_names = df["Player_1" if flip else "Player_2"].values
    label    = (1 - df["p1_wins"]).values if flip else df["p1_wins"].values
    h2h_rate = (
        1.0 - df["p1_h2h_rate_pre"].values if flip
        else df["p1_h2h_rate_pre"].values
    ) if "p1_h2h_rate_pre" in df.columns else np.full(len(df), 0.5)

    surf_vals   = df["Surface"].values
    match_years = df["year"].values.astype(float)

    def _flt(names, key):
        return np.array([float(player_attrs.get(_norm(n), {}).get(key, np.nan))
                         for n in names], dtype=float)

    p1_by   = _flt(p1_names, "birth_year");  p2_by   = _flt(p2_names, "birth_year")
    p1_ht   = _flt(p1_names, "ht");          p2_ht   = _flt(p2_names, "ht")
    p1_hand = np.array([player_attrs.get(_norm(n), {}).get("hand", "U") for n in p1_names])
    p2_hand = np.array([player_attrs.get(_norm(n), {}).get("hand", "U") for n in p2_names])

    def _srf(names, key, default):
        return np.array([
            float(surf_lookup.get((_norm(n), str(s or "").lower()), {}).get(key, default))
            for n, s in zip(names, surf_vals)
        ], dtype=float)

    return pd.DataFrame({
        "year":    df["year"].values,
        "p1_rank": _v("Rank_1", "Rank_2"), "p2_rank": _v("Rank_2", "Rank_1"),
        "p1_pts":  _v("Pts_1",  "Pts_2"),  "p2_pts":  _v("Pts_2",  "Pts_1"),
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
        "h2h_rate": h2h_rate,
        "h2h_n":    df["h2h_n_pre"].values if "h2h_n_pre" in df.columns else np.zeros(len(df)),
        "p1_days": _v("p1_days_pre", "p2_days_pre"),
        "p2_days": _v("p2_days_pre", "p1_days_pre"),
        "p1_exp":  _v("p1_exp_pre",  "p2_exp_pre"),
        "p2_exp":  _v("p2_exp_pre",  "p1_exp_pre"),
        "p1_age":  match_years - p1_by,  "p2_age":  match_years - p2_by,
        "p1_ht":   p1_ht,                "p2_ht":   p2_ht,
        "p1_hand": p1_hand,              "p2_hand": p2_hand,
        "p1_swr":  _srf(p1_names, "win_rate",  np.nan),
        "p2_swr":  _srf(p2_names, "win_rate",  np.nan),
        "p1_sn":   _srf(p1_names, "n_matches", 0),
        "p2_sn":   _srf(p2_names, "n_matches", 0),
        "surface": df["Surface"].values,
        "series":  df["Series"].values if "Series" in df.columns else np.full(len(df), "unknown"),
        "court":   df["Court"].values  if "Court"  in df.columns else np.full(len(df), "unknown"),
        "round":   df["Round"].values,
        "best_of": df["best_of"].values,
        "label":   label,
    })


def build_dataset(df, player_attrs, surf_lookup):
    df = df.copy()
    df["p1_wins"] = (df["Player_1"] == df["Winner"]).astype(int)
    data = pd.concat(
        [_make_rows(df, False, player_attrs, surf_lookup),
         _make_rows(df, True,  player_attrs, surf_lookup)],
        ignore_index=True,
    )

    data["rank_diff"]              = data["p1_rank"]  - data["p2_rank"]
    data["pts_diff"]               = data["p1_pts"].fillna(0) - data["p2_pts"].fillna(0)
    data["rank_ratio"]             = data["p1_rank"] / data["p2_rank"].clip(lower=1)
    data["log_rank_diff"]          = np.log1p(data["p1_rank"]) - np.log1p(data["p2_rank"])

    data["elo_diff"]               = data["p1_elo"]  - data["p2_elo"]
    data["elo_surf_diff"]          = data["p1_eloS"] - data["p2_eloS"]
    # Surface specialist: how much more suited is p1 vs p2 to this specific surface?
    data["surface_specialist_diff"]= data["elo_surf_diff"] - data["elo_diff"]

    data["form_short_diff"]        = data["p1_fS"]   - data["p2_fS"]
    data["form_long_diff"]         = data["p1_fL"]   - data["p2_fL"]
    data["surf_form_diff"]         = data["p1_sf"]   - data["p2_sf"]
    # Form trend: positive = p1 improving relative to p2
    data["form_trend_diff"]        = data["form_short_diff"] - data["form_long_diff"]

    data["days_rest_diff"]         = data["p1_days"] - data["p2_days"]
    data["experience_diff"]        = data["p1_exp"]  - data["p2_exp"]

    data["age_diff"]               = data["p1_age"]  - data["p2_age"]
    data["ht_diff"]                = data["p1_ht"]   - data["p2_ht"]
    data["surf_win_rate_diff"]     = data["p1_swr"]  - data["p2_swr"]
    data["surf_exp_diff"]          = data["p1_sn"]   - data["p2_sn"]

    print(f"  Dataset: {len(data):,} rows  ({int(data['year'].min())}–{int(data['year'].max())})")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MODEL PIPELINES
# ══════════════════════════════════════════════════════════════════════════════

def _num_cat_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc",  StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, num_cols),
                               ("cat", cat_pipe, cat_cols)])


def make_pipeline() -> CalibratedClassifierCV:
    """Full Model 2: XGBoost with isotonic calibration."""
    pre = _num_cat_preprocessor(NUMERIC_COLS, CAT_COLS)
    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=15,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    return CalibratedClassifierCV(Pipeline([("pre", pre), ("clf", xgb)]),
                                  cv=3, method="isotonic")


def make_baseline_pipeline() -> Pipeline:
    """Simple logistic regression on 6 rank/form features — mirrors model_1."""
    pre = _num_cat_preprocessor(BASELINE_NUM, BASELINE_CAT)
    lr  = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    return Pipeline([("pre", pre), ("clf", lr)])


def fit_eval(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return accuracy_score(y_te, preds), roc_auc_score(y_te, probs), \
           brier_score_loss(y_te, probs), probs


# ══════════════════════════════════════════════════════════════════════════════
# 6.  VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    """
    Expanding walk-forward 2016-2024.
    Trains BOTH the full Model 2 AND a logistic baseline each year so the
    yearly comparison chart shows apples-to-apples improvement.
    """
    rows = []
    for test_yr in sorted(data["year"].unique()):
        if test_yr < 2016 or test_yr > 2024:
            continue
        tr = data[data["year"] < test_yr]
        te = data[data["year"] == test_yr]
        if len(tr) < 100 or len(te) < 50:
            continue

        y_tr = tr["label"];  y_te = te["label"]

        # Full Model 2
        m2 = make_pipeline()
        acc2, auc2, _, _ = fit_eval(m2,
                                    tr[NUMERIC_COLS + CAT_COLS], y_tr,
                                    te[NUMERIC_COLS + CAT_COLS], y_te)
        # Logistic baseline (same data window)
        mb = make_baseline_pipeline()
        acc_b, auc_b, _, _ = fit_eval(mb,
                                      tr[BASELINE_NUM + BASELINE_CAT], y_tr,
                                      te[BASELINE_NUM + BASELINE_CAT], y_te)

        base = accuracy_score(y_te, (te["rank_diff"] < 0).astype(int))

        rows.append(dict(test_year=test_yr,
                         acc_m2=acc2,  auc_m2=auc2,
                         acc_lr=acc_b, auc_lr=auc_b,
                         rank_baseline=base, n=len(te)))
        print(f"    {test_yr}: M2={acc2:.4f}  LR={acc_b:.4f}  "
              f"rank={base:.4f}  (n={len(te):,})")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  FEATURE IMPORTANCE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_importances(model):
    try:
        pipes = ([cc.estimator for cc in model.calibrated_classifiers_]
                 if hasattr(model, "calibrated_classifiers_") else [model])
        imps  = [p.named_steps["clf"].feature_importances_
                 for p in pipes if hasattr(p, "named_steps")]
        if not imps:
            return None
        avg   = np.mean(imps, axis=0)
        names = list(pipes[0].named_steps["pre"].get_feature_names_out())
        return names, avg
    except Exception as exc:
        print(f"  Warning: importance extraction failed: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 8.  PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _save(fname: str) -> None:
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ── Chart 1 — ROC curve ───────────────────────────────────────────────────────
def plot_roc_curve(y_te, probs, acc, auc):
    fpr, tpr, _ = roc_curve(y_te, probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color="darkorange", label=f"Model 2 XGBoost (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], ":", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — Model 2\n(Test 2021-2024  |  Accuracy={acc:.4f})")
    ax.legend()
    _save("roc_curve.png")


# ── Chart 2 — Calibration ─────────────────────────────────────────────────────
def plot_calibration(y_te, probs):
    fop, mpv = calibration_curve(y_te, probs, n_bins=10)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(mpv, fop, "o-", color="darkorange", lw=2, label="Model 2")
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve — Model 2\n"
                 "(points close to diagonal = well-calibrated probabilities)")
    ax.legend()
    _save("calibration.png")


# ── Chart 3 — Confusion matrix ────────────────────────────────────────────────
def plot_confusion_matrix(y_te, probs):
    cm = confusion_matrix(y_te, (probs >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Oranges")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Loss", "Predicted Win"], fontsize=11)
    ax.set_yticklabels(["Actual Loss",    "Actual Win"],    fontsize=11)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=16,
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    ax.set_title("Confusion Matrix — Model 2 (Test 2021-2024)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    _save("confusion_matrix.png")


# ── Chart 4 — Feature importance (per feature) ────────────────────────────────
def plot_feature_importance(model):
    result = _extract_importances(model)
    if result is None:
        return
    names, importances = result
    df = pd.DataFrame({"feature": names, "importance": importances})
    df = df.sort_values("importance", ascending=False)
    df.to_csv(os.path.join(PLOTS_DIR, "feature_importance.csv"), index=False)
    top = df.head(25).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(top["feature"], top["importance"], color=[_feat_color(f) for f in top["feature"]])
    ax.set_xlabel("Feature Importance (XGBoost Gain)")
    ax.set_title("Top 25 Feature Importances — Model 2  (colour = feature group)")
    ax.legend(handles=[
        mpatches.Patch(facecolor="darkorange",   label="Elo (Groups B)"),
        mpatches.Patch(facecolor="steelblue",    label="Rolling form (Group C)"),
        mpatches.Patch(facecolor="seagreen",     label="H2H (Group D)"),
        mpatches.Patch(facecolor="mediumpurple", label="Rank/points (Group A)"),
        mpatches.Patch(facecolor="teal",         label="Surface history (Group G)"),
        mpatches.Patch(facecolor="dimgray",      label="Rest/experience (Group E)"),
        mpatches.Patch(facecolor="orchid",       label="Player attributes (Group F)"),
    ], loc="lower right", fontsize=8)
    _save("feature_importance.png")


# ── Chart 5 — Walk-forward accuracy + AUC ─────────────────────────────────────
def plot_walk_forward(wf: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    ax1.plot(wf.test_year, wf.acc_m2,        "o-",  color="darkorange", lw=2,
             label="Model 2 — XGBoost + Elo")
    ax1.plot(wf.test_year, wf.rank_baseline, "^:",  color="gray",       lw=1.5,
             label="Rank baseline")
    ax1.set_ylabel("Accuracy"); ax1.set_title("Walk-Forward Validation — Accuracy (Model 2)")
    ax1.legend(); ax1.set_ylim(0.55, 0.82); ax1.grid(axis="y", alpha=0.3)
    ax2.plot(wf.test_year, wf.auc_m2, "o-", color="darkorange", lw=2)
    ax2.set_ylabel("ROC AUC"); ax2.set_xlabel("Test Year")
    ax2.set_title("Walk-Forward Validation — ROC AUC (Model 2)")
    ax2.set_ylim(0.60, 0.88); ax2.grid(axis="y", alpha=0.3)
    for _, r in wf.iterrows():
        ax1.annotate(f"n={int(r.n):,}", xy=(r.test_year, r.acc_m2 - 0.015),
                     ha="center", fontsize=7, color="gray")
    _save("walk_forward.png")


# ── Chart 6 — Accuracy by surface ─────────────────────────────────────────────
def plot_accuracy_by_surface(test, model):
    valid, accs, bases, ns = [], [], [], []
    for surf in sorted(s for s in test["surface"].unique()
                       if s not in ("unknown", "nan") and pd.notna(s)):
        sub = test[test["surface"] == surf]
        if len(sub) < 100:
            continue
        p = model.predict_proba(sub[NUMERIC_COLS + CAT_COLS])[:, 1]
        valid.append(surf)
        accs.append(accuracy_score(sub["label"], (p >= 0.5).astype(int)))
        bases.append(accuracy_score(sub["label"], (sub["rank_diff"] < 0).astype(int)))
        ns.append(len(sub))
    x = np.arange(len(valid)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, accs,  w, color="darkorange", alpha=0.85, label="Model 2 (XGBoost)")
    ax.bar(x + w/2, bases, w, color="lightgray",  alpha=0.85, label="Rank baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s.title()}\n(n={n:,})" for s, n in zip(valid, ns)])
    ax.set_ylabel("Accuracy"); ax.set_ylim(0.55, 0.82)
    ax.set_title("Accuracy by Surface — Model 2 vs Rank Baseline (Test 2021-2024)")
    ax.axhline(0.5, color="gray", linestyle=":", lw=1); ax.legend()
    _save("accuracy_by_surface.png")


# ── Chart 7 — Accuracy by confidence ──────────────────────────────────────────
def plot_accuracy_by_confidence(y_te, probs):
    df = pd.DataFrame({"prob": probs, "label": y_te.values})
    df["conf"]    = np.abs(df["prob"] - 0.5)
    df["correct"] = ((df["prob"] >= 0.5).astype(int) == df["label"]).astype(int)
    bins   = np.linspace(0, 0.5, 11)
    labels = [f"{int(bins[i]*100)}-{int(bins[i+1]*100)}%" for i in range(len(bins)-1)]
    df["bin"] = pd.cut(df["conf"], bins=bins, labels=labels, include_lowest=True)
    grp = df.groupby("bin", observed=True)["correct"].agg(acc="mean", n="count").reset_index()
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax1.bar(grp["bin"], grp["acc"], alpha=0.7, color="darkorange", label="Accuracy")
    ax2.plot(grp["bin"], grp["n"], "o-", color="steelblue", label="Count")
    ax1.axhline(0.5, color="gray", linestyle="--")
    ax1.set_xlabel("Confidence Bucket (|P − 0.5|)")
    ax1.set_ylabel("Accuracy"); ax2.set_ylabel("Sample Count")
    ax1.set_title("Accuracy vs Prediction Confidence — Model 2\n"
                  "(higher confidence should correlate with higher accuracy)")
    plt.xticks(rotation=30, ha="right")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="lower right")
    _save("accuracy_by_confidence.png")


# ── Chart 8 — Probability histogram ───────────────────────────────────────────
def plot_prob_histogram(y_te, probs):
    df = pd.DataFrame({"prob": probs, "label": y_te.values})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[df.label==1]["prob"], bins=30, alpha=0.6, color="darkorange", label="Wins (label=1)")
    ax.hist(df[df.label==0]["prob"], bins=30, alpha=0.6, color="steelblue",  label="Losses (label=0)")
    ax.axvline(0.5, color="black", linestyle="--")
    ax.set_xlabel("Predicted Win Probability"); ax.set_ylabel("Count")
    ax.set_title("Predicted Probability Distribution — Model 2\n"
                 "(good separation between wins and losses = strong model)")
    ax.legend()
    _save("prob_histogram.png")


# ── Chart 9 — Model 2 vs logistic baseline ROC (same axes) ───────────────────
def plot_vs_baseline_roc(y_te, probs_m2, probs_lr):
    fpr2, tpr2, _ = roc_curve(y_te, probs_m2)
    fpr_b, tpr_b, _ = roc_curve(y_te, probs_lr)
    auc2 = roc_auc_score(y_te, probs_m2)
    auc_b = roc_auc_score(y_te, probs_lr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr2,  tpr2,  lw=2, color="darkorange",
            label=f"Model 2 — XGBoost + Elo  (AUC={auc2:.4f})")
    ax.plot(fpr_b, tpr_b, lw=2, color="steelblue", linestyle="--",
            label=f"Logistic baseline (AUC={auc_b:.4f})")
    ax.plot([0,1], [0,1], ":", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Comparison — Model 2 vs Logistic Baseline\n"
                 "(both trained on same 2014-2020 data, tested 2021-2024)")
    ax.fill_between(fpr2, tpr_b[:len(fpr2)] if len(fpr_b) >= len(fpr2)
                    else np.interp(fpr2, fpr_b, tpr_b), tpr2,
                    alpha=0.08, color="darkorange", label="Improvement region")
    ax.legend()
    _save("vs_baseline_roc.png")


# ── Chart 10 — Yearly accuracy: M2 vs logistic vs rank baseline ───────────────
def plot_yearly_comparison(wf: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wf.test_year, wf.acc_m2,        "o-",  color="darkorange", lw=2,
            label="Model 2 — XGBoost + Elo + Form")
    ax.plot(wf.test_year, wf.acc_lr,        "s--", color="steelblue",  lw=2,
            label="Logistic baseline (rank + H2H only)")
    ax.plot(wf.test_year, wf.rank_baseline, "^:",  color="gray",       lw=1.5,
            label="Rank baseline (always pick lower-ranked)")
    ax.fill_between(wf.test_year, wf.acc_lr, wf.acc_m2, alpha=0.1, color="darkorange",
                    label="Model 2 gain over logistic")
    ax.set_xlabel("Test Year"); ax.set_ylabel("Accuracy")
    ax.set_title("Yearly Accuracy Comparison\nModel 2 vs Logistic Baseline vs Rank Baseline")
    ax.legend(); ax.set_ylim(0.55, 0.80); ax.grid(axis="y", alpha=0.3)
    for _, r in wf.iterrows():
        delta = r.acc_m2 - r.acc_lr
        ax.annotate(f"+{delta:.3f}", xy=(r.test_year, (r.acc_m2+r.acc_lr)/2),
                    ha="center", fontsize=7.5, color="darkorange", fontweight="bold")
    _save("yearly_comparison.png")


# ── Chart 11 — Feature importance by group ────────────────────────────────────
def plot_feature_groups(model):
    result = _extract_importances(model)
    if result is None:
        return
    names, importances = result
    imp_df = pd.DataFrame({"feature": names, "importance": importances})

    # Map each raw feature name back to its group
    def _feature_to_base(feat: str) -> str:
        feat = feat.replace("num__", "").replace("cat__", "")
        return feat.split("_")[0] if "_" in feat else feat

    group_totals = {}
    for grp, feats in GROUP_LABELS.items():
        total = 0.0
        for f in feats:
            match = imp_df[imp_df["feature"].str.contains(f, regex=False)]
            total += match["importance"].sum()
        group_totals[grp] = total

    # Normalise to % of total
    total_imp = sum(group_totals.values()) or 1.0
    groups = list(group_totals.keys())
    pcts   = [group_totals[g] / total_imp * 100 for g in groups]
    colors = [GROUP_COLORS_BAR[g] for g in groups]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(groups, pcts, color=colors, edgecolor="white")
    ax.set_xlabel("% of Total Model Gain")
    ax.set_title("Feature Importance by Group — Model 2\n"
                 "(shows which category of features contributes most)")
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{pct:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(pcts) * 1.2)
    _save("feature_groups.png")


# ── Chart 12 — Elo diff vs actual win rate ────────────────────────────────────
def plot_elo_validation(test: pd.DataFrame):
    """
    Validates that Elo is a real predictive signal.
    Bins test rows by elo_diff and compares actual win rate vs
    the theoretical Elo win probability at the bin centre.
    """
    df = test[["elo_diff", "label"]].dropna().copy()
    bin_edges = [-600, -300, -150, -75, -25, 25, 75, 150, 300, 600]
    bin_labels = [f"{(a+b)//2:+d}" for a, b in zip(bin_edges[:-1], bin_edges[1:])]
    df["bin"] = pd.cut(df["elo_diff"], bins=bin_edges, labels=bin_labels)
    grp = (df.groupby("bin", observed=True)["label"]
             .agg(win_rate="mean", n="count")
             .reset_index()
             .dropna())

    # Theoretical Elo probability at each bin midpoint
    midpoints = [(a + b) / 2 for a, b in zip(bin_edges[:-1], bin_edges[1:])]
    mid_labels = [f"{int(m):+d}" for m in midpoints]
    elo_theory = [_elo_expected(1500 + m, 1500) for m in midpoints]
    theory_df  = pd.DataFrame({"bin": bin_labels, "elo_prob": elo_theory})
    grp = grp.merge(theory_df, on="bin", how="left")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(grp["bin"], grp["win_rate"], color="darkorange", alpha=0.75,
           label="Actual win rate (test data)")
    ax.plot(grp["bin"], grp["elo_prob"], "o--", color="steelblue", lw=2,
            label="Theoretical Elo win probability")
    ax.axhline(0.5, color="gray", linestyle=":", lw=1)
    ax.set_xlabel("Elo Difference (p1 − p2)")
    ax.set_ylabel("Win Rate")
    ax.set_title("Elo Signal Validation — Actual Win Rate vs Elo Theory\n"
                 "(bars close to line = Elo predicts match outcomes well)")
    for _, r in grp.iterrows():
        ax.annotate(f"n={int(r.n):,}", xy=(r.bin, r.win_rate + 0.01),
                    ha="center", fontsize=7.5, color="gray")
    ax.legend(); ax.set_ylim(0.25, 0.80)
    _save("elo_validation.png")


# ── Chart 13 — Accuracy by round ──────────────────────────────────────────────
def plot_accuracy_by_round(test: pd.DataFrame, model):
    order  = ["r128", "r64", "r32", "r16", "qf", "sf", "f", "rr", "br"]
    rounds = [r for r in order if r in test["round"].unique()]
    rounds += [r for r in test["round"].unique()
               if r not in order and r not in ("unknown", "nan") and pd.notna(r)]

    accs, ns, valid = [], [], []
    for rnd in rounds:
        sub = test[test["round"] == rnd]
        if len(sub) < 50:
            continue
        p = model.predict_proba(sub[NUMERIC_COLS + CAT_COLS])[:, 1]
        accs.append(accuracy_score(sub["label"], (p >= 0.5).astype(int)))
        ns.append(len(sub)); valid.append(rnd.upper())

    x = np.arange(len(valid))
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(x, accs, color="darkorange", alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n(n={n:,})" for r, n in zip(valid, ns)], fontsize=9)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0.55, 0.88)
    ax.axhline(0.5, color="gray", linestyle=":", lw=1)
    ax.set_title("Accuracy by Round — Model 2 (Test 2021-2024)\n"
                 "(early rounds are more predictable — bigger rank gaps)")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{acc:.3f}", ha="center", fontsize=9)
    _save("accuracy_by_round.png")


# ── Chart 14 — Accuracy by rank gap ───────────────────────────────────────────
def plot_rank_gap_accuracy(test: pd.DataFrame, probs):
    df = test[["rank_diff", "label"]].copy()
    df["prob"]    = probs
    df["correct"] = ((df["prob"] >= 0.5).astype(int) == df["label"]).astype(int)
    df["abs_gap"] = df["rank_diff"].abs()

    bins   = [0, 10, 25, 50, 100, 200, 500, 2000]
    labels = ["0-10", "10-25", "25-50", "50-100", "100-200", "200-500", "500+"]
    df["gap_bin"] = pd.cut(df["abs_gap"], bins=bins, labels=labels)
    grp = (df.groupby("gap_bin", observed=True)
             .agg(m2=("correct", "mean"),
                  base=("abs_gap", lambda x: ((df.loc[x.index, "rank_diff"] < 0) ==
                                               df.loc[x.index, "label"]).mean()),
                  n=("correct", "count"))
             .reset_index())

    x = np.arange(len(grp)); w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, grp["m2"],   w, color="darkorange", alpha=0.85, label="Model 2 (XGBoost)")
    ax.bar(x + w/2, grp["base"], w, color="lightgray",  alpha=0.85, label="Rank baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{str(r['gap_bin'])}\n(n={int(r.n):,})"
                        for _, r in grp.iterrows()], fontsize=9)
    ax.axhline(0.5, color="gray", linestyle=":", lw=1)
    ax.set_xlabel("|Rank gap| between players")
    ax.set_ylabel("Accuracy"); ax.set_ylim(0.40, 0.95)
    ax.set_title("Accuracy by Rank Gap — Model 2 vs Rank Baseline\n"
                 "(larger rank gap = more predictable; Model 2 gains most in close matches)")
    ax.legend()
    _save("rank_gap_accuracy.png")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 65)
    print("  Model 2 — XGBoost + Elo + Rolling Form + Surface History")
    print("═" * 65)

    print("\nLoading data ...")
    processed = load_processed(PROC_DIR)
    matches   = load_initial_matches(INIT_DIR)
    surf_df   = load_surface_results(INIT_DIR)

    print("\nBuilding static lookups ...")
    player_attrs = build_player_attrs(matches)
    surf_lookup  = build_surface_lookup(surf_df)

    print("\nComputing dynamic features (Elo, form, H2H) ...")
    processed = compute_dynamic_features(processed)

    print("\nBuilding symmetric dataset ...")
    data = build_dataset(processed, player_attrs, surf_lookup)

    data_comp = data[(data["year"] >= 2014) & (data["year"] <= 2024)].copy()

    # ── Walk-forward (reports both Model 2 AND logistic baseline per year) ────
    print("\nRunning walk-forward validation (2016–2024) ...")
    wf = walk_forward(data_comp)

    # ── Final holdout ─────────────────────────────────────────────────────────
    print("\nTraining final models (train 2014-2020 / test 2021-2024) ...")
    train = data_comp[data_comp["year"] <= 2020]
    test  = data_comp[data_comp["year"] >= 2021]

    # Full Model 2
    final_model = make_pipeline()
    acc, auc, brier, probs = fit_eval(
        final_model,
        train[NUMERIC_COLS + CAT_COLS], train["label"],
        test[NUMERIC_COLS + CAT_COLS],  test["label"],
    )

    # Logistic baseline on same split (for comparison charts)
    lr_model = make_baseline_pipeline()
    acc_lr, auc_lr, brier_lr, probs_lr = fit_eval(
        lr_model,
        train[BASELINE_NUM + BASELINE_CAT], train["label"],
        test[BASELINE_NUM + BASELINE_CAT],  test["label"],
    )

    rank_base = accuracy_score(test["label"], (test["rank_diff"] < 0).astype(int))

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  Results — Model 2 vs Logistic Baseline vs Rank Baseline")
    print(f"{'═'*70}")
    print(f"{'Metric':<28} {'Model 2':>12}  {'Logistic LR':>13}  {'Rank base':>10}")
    print(f"{'─'*70}")
    print(f"{'Accuracy':<28} {acc:>12.4f}  {acc_lr:>13.4f}  {rank_base:>10.4f}")
    print(f"{'ROC AUC':<28} {auc:>12.4f}  {auc_lr:>13.4f}  {'—':>10}")
    print(f"{'Brier Score':<28} {brier:>12.4f}  {brier_lr:>13.4f}  {'—':>10}  (↓ better)")
    print(f"{'─'*70}")
    print(f"{'Δ Accuracy vs LR':<28} {acc - acc_lr:>+12.4f}")
    print(f"{'Δ AUC vs LR':<28} {auc - auc_lr:>+12.4f}")
    print(f"{'Δ Brier vs LR':<28} {brier - brier_lr:>+12.4f}  (negative = better)")
    print(f"{'═'*70}\n")

    joblib.dump(final_model, os.path.join(ROOT, "model_2.joblib"))
    print(f"Saved model → {ROOT}/model_2.joblib")

    # ── Generate all 14 plots ─────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_roc_curve(test["label"], probs, acc, auc)
    plot_calibration(test["label"], probs)
    plot_confusion_matrix(test["label"], probs)
    plot_feature_importance(final_model)
    plot_walk_forward(wf)
    plot_accuracy_by_surface(test, final_model)
    plot_accuracy_by_confidence(test["label"], probs)
    plot_prob_histogram(test["label"], probs)
    plot_vs_baseline_roc(test["label"], probs, probs_lr)
    plot_yearly_comparison(wf)
    plot_feature_groups(final_model)
    plot_elo_validation(test)
    plot_accuracy_by_round(test, final_model)
    plot_rank_gap_accuracy(test, probs)

    print(f"\nAll 14 plots saved to: {PLOTS_DIR}")
    print("Done.\n")


if __name__ == "__main__":
    main()
