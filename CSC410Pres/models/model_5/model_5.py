#!/usr/bin/env python3
"""
model_5.py - Model 2 + Playstyle Matchups (Impact Analysis)
===========================================================
Builds directly on Model 2 by adding static player playstyle categories from
CSC410Pres/data/playstyle_reports/. Each player belongs to one of four styles:
Aggressive Baseliner, All-Court Player, Counterpuncher, or Serve-and-Volleyer.

The goal is to test whether playstyle matchup information improves predictions
and to produce tennis-specific charts that tell a clear story about which style
matchups matter and where style helps on different surfaces.

Data sources:
  data/processed/atp_tennis.csv
  data/initial/atp_matches_YYYY.csv
  data/initial/player_surface_results.csv
  data/playstyle_reports/*.json

Validation:
  1. Expanding walk-forward - test years 2016-2024
  2. Single holdout         - train 2014-2020, test 2021-2024

Usage:
  cd CSC410Pres/models/model_5
  python model_5.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import unicodedata
import warnings
from collections import defaultdict, deque

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "..", "..", "data")
INIT_DIR = os.path.join(DATA_DIR, "initial")
PROC_DIR = os.path.join(DATA_DIR, "processed")
PLAYSTYLE_DIR = os.path.join(DATA_DIR, "playstyle_reports")
PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

ELO_BASE = 1500.0
ELO_K = 32.0
ELO_K_SURF = 24.0
FORM_SHORT = 5
FORM_LONG = 20

STYLE_ORDER = [
    "Aggressive Baseliner",
    "All-Court Player",
    "Counterpuncher",
    "Serve-and-Volleyer",
]
STYLE_UNKNOWN = "Unknown"

# Model 2 base features retained for direct comparison.
NUMERIC_COLS_M2 = [
    "rank_diff",
    "pts_diff",
    "rank_ratio",
    "log_rank_diff",
    "elo_diff",
    "elo_surf_diff",
    "surface_specialist_diff",
    "form_short_diff",
    "form_long_diff",
    "surf_form_diff",
    "form_trend_diff",
    "h2h_rate",
    "h2h_n",
    "days_rest_diff",
    "experience_diff",
    "age_diff",
    "ht_diff",
    "surf_win_rate_diff",
    "surf_exp_diff",
]
CAT_COLS_M2 = ["surface", "series", "court", "round", "best_of", "p1_hand", "p2_hand"]
STYLE_CAT_COLS = ["p1_style", "p2_style", "style_matchup", "style_pair"]
STYLE_NUMERIC = ["same_style"]
CAT_COLS_M5 = CAT_COLS_M2 + STYLE_CAT_COLS
NUMERIC_COLS_M5 = NUMERIC_COLS_M2 + STYLE_NUMERIC
NUMERIC_COLS = NUMERIC_COLS_M5

BASELINE_NUM = ["rank_diff", "log_rank_diff", "rank_ratio", "pts_diff", "age_diff", "h2h_rate"]
BASELINE_CAT = ["surface", "series", "round", "best_of"]

GROUP_COLORS = [
    ("num__elo", "darkorange"),
    ("num__surf_form", "steelblue"),
    ("num__surface_specialist", "darkorange"),
    ("num__form", "steelblue"),
    ("num__h2h", "seagreen"),
    ("num__rank", "mediumpurple"),
    ("num__pts", "mediumpurple"),
    ("num__log", "mediumpurple"),
    ("num__days", "dimgray"),
    ("num__exp", "dimgray"),
    ("num__age", "orchid"),
    ("num__ht", "orchid"),
    ("num__surf_win", "teal"),
    ("num__surf_exp", "teal"),
    ("num__same_style", "crimson"),
    ("cat__p1_style", "crimson"),
    ("cat__p2_style", "crimson"),
    ("cat__style_matchup", "crimson"),
    ("cat__style_pair", "crimson"),
]

GROUP_LABELS = {
    "A - Rank/skill": ["rank_diff", "pts_diff", "rank_ratio", "log_rank_diff"],
    "B - Elo": ["elo_diff", "elo_surf_diff", "surface_specialist_diff"],
    "C - Rolling form": ["form_short_diff", "form_long_diff", "surf_form_diff", "form_trend_diff"],
    "D - H2H": ["h2h_rate", "h2h_n"],
    "E - Rest/experience": ["days_rest_diff", "experience_diff"],
    "F - Player attributes": ["age_diff", "ht_diff"],
    "G - Surface history": ["surf_win_rate_diff", "surf_exp_diff"],
    "H - Playstyle": ["same_style", "p1_style", "p2_style", "style_matchup", "style_pair"],
}

GROUP_COLORS_BAR = {
    "A - Rank/skill": "mediumpurple",
    "B - Elo": "darkorange",
    "C - Rolling form": "steelblue",
    "D - H2H": "seagreen",
    "E - Rest/experience": "dimgray",
    "F - Player attributes": "orchid",
    "G - Surface history": "teal",
    "H - Playstyle": "crimson",
}


def _norm(name: str) -> str:
    text = str(name).strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _norm_atp(full_name: str) -> str:
    parts = str(full_name).strip().split()
    if len(parts) >= 2:
        last_name = " ".join(parts[1:])
        first_initial = parts[0][0]
        return _norm(f"{last_name} {first_initial}.")
    return _norm(full_name)


def _elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _form(dq: deque, n: int | None = None) -> float:
    values = list(dq)[-n:] if n else list(dq)
    return float(np.mean(values)) if values else 0.5


def _feat_color(feat: str) -> str:
    for prefix, color in GROUP_COLORS:
        if feat.startswith(prefix):
            return color
    return "silver"


def _save(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=220, bbox_inches="tight")
    plt.close()


def _extract_importances(model) -> tuple[np.ndarray, np.ndarray] | None:
    pipe = None
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        pipe = model.calibrated_classifiers_[0].estimator
    elif hasattr(model, "named_steps"):
        pipe = model
    if pipe is None or not hasattr(pipe, "named_steps"):
        return None
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return None
    return pre.get_feature_names_out(), clf.feature_importances_


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
    print(f"  Processed: {len(df):,} matches ({int(df['year'].min())}-{int(df['year'].max())})")
    return df


def load_initial_matches(init_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(init_dir, "atp_matches_*.csv")))
    if not files:
        raise FileNotFoundError(f"No atp_matches_YYYY.csv in {init_dir}")
    df = pd.concat([pd.read_csv(path, low_memory=False) for path in files], ignore_index=True)
    df["match_date"] = pd.to_datetime(
        df["tourney_date"].astype(str).str[:8], format="%Y%m%d", errors="coerce"
    )
    df["year"] = df["match_date"].dt.year
    for col in ["winner_rank", "loser_rank", "winner_age", "loser_age", "winner_ht", "loser_ht"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"  Initial: {len(df):,} matches ({int(df['year'].min())}-{int(df['year'].max())})")
    return df


def load_surface_results(init_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(init_dir, "player_surface_results.csv"))
    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["surface"] = df["surface"].astype(str).str.strip().str.lower()
    for col in ["wins", "losses", "matches"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if "matches" not in df.columns:
        df["matches"] = df.get("wins", 0) + df.get("losses", 0)
    df["win_rate"] = np.where(
        df["matches"] > 0,
        df.get("wins", df["matches"] * 0.5) / df["matches"],
        0.5,
    )
    df["name_key"] = df["player_name"].map(_norm)
    print(f"  Surface results: {len(df):,} rows ({df['name_key'].nunique():,} unique players)")
    return df


def load_playstyle_data(playstyle_dir: str) -> tuple[dict, dict]:
    files = sorted(glob.glob(os.path.join(playstyle_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No playstyle report JSON files in {playstyle_dir}")

    canonical_styles: dict[str, str] = {}
    aliases: dict[str, set] = defaultdict(set)
    for file_path in files:
        with open(file_path, encoding="utf-8") as handle:
            raw = json.load(handle)
        for row in raw:
            player_name = str(row.get("player_name", "")).strip()
            style = str(row.get("play_style", STYLE_UNKNOWN)).strip() or STYLE_UNKNOWN
            canonical = _norm(player_name)
            canonical_styles[canonical] = style
            aliases[canonical].update([canonical, _norm_atp(player_name)])

    lookup: dict[str, str] = {}
    counts = defaultdict(int)
    for canonical, style in canonical_styles.items():
        counts[style] += 1
        for alias in aliases[canonical]:
            if alias:
                lookup[alias] = style

    print(f"  Playstyles: {len(canonical_styles):,} players from {len(files)} files")
    for style in STYLE_ORDER:
        print(f"    {style}: {counts.get(style, 0):,} players")
    return lookup, dict(counts)


def build_player_attrs(matches: pd.DataFrame) -> dict:
    winner = pd.DataFrame(
        {
            "name_key": matches["winner_name"].map(_norm),
            "hand": matches.get("winner_hand", pd.Series("U", index=matches.index)),
            "ht": matches.get("winner_ht", pd.Series(np.nan, index=matches.index)),
            "birth_year": matches["year"] - matches["winner_age"],
        }
    )
    loser = pd.DataFrame(
        {
            "name_key": matches["loser_name"].map(_norm),
            "hand": matches.get("loser_hand", pd.Series("U", index=matches.index)),
            "ht": matches.get("loser_ht", pd.Series(np.nan, index=matches.index)),
            "birth_year": matches["year"] - matches["loser_age"],
        }
    )
    df = pd.concat([winner, loser], ignore_index=True)
    df = df[df["name_key"] != ""]

    def _mode(series: pd.Series) -> str:
        mode = series.dropna().mode()
        return str(mode.iloc[0]) if len(mode) else "U"

    result: dict[str, dict] = {}
    for key, grp in df.groupby("name_key"):
        result[key] = {
            "hand": _mode(grp["hand"]),
            "ht": float(grp["ht"].median()) if grp["ht"].notna().any() else np.nan,
            "birth_year": float(grp["birth_year"].median()) if grp["birth_year"].notna().any() else np.nan,
        }
    print(f"  Player attrs: {len(result):,} players indexed")
    return result


def build_surface_lookup(surf_df: pd.DataFrame) -> dict:
    lookup: dict[tuple, dict] = {}
    for _, row in surf_df.iterrows():
        lookup[(row["name_key"], row["surface"])] = {
            "win_rate": float(row["win_rate"]),
            "n_matches": int(row["matches"]),
        }
    print(f"  Surface lookup: {len(lookup):,} (player, surface) entries")
    return lookup


def _player_style(player: str, playstyle_lookup: dict) -> str:
    return playstyle_lookup.get(_norm(player), STYLE_UNKNOWN)


def compute_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    elo_all = defaultdict(lambda: ELO_BASE)
    elo_surf = defaultdict(lambda: defaultdict(lambda: ELO_BASE))
    res_all = defaultdict(lambda: deque(maxlen=FORM_LONG))
    res_surf = defaultdict(lambda: defaultdict(lambda: deque(maxlen=FORM_LONG)))
    last_dt: dict = {}
    exp_cnt = defaultdict(int)
    h2h_wins = defaultdict(int)

    feat_rows = []
    for row in df.itertuples(index=False):
        p1 = getattr(row, "Player_1", None)
        p2 = getattr(row, "Player_2", None)
        if not isinstance(p1, str) or not isinstance(p2, str):
            feat_rows.append({})
            continue

        win = getattr(row, "Winner", None)
        surf = str(getattr(row, "Surface", "unknown") or "unknown").lower()
        date = getattr(row, "Date", None)
        p1_won = win == p1

        p1_elo = elo_all[p1]
        p2_elo = elo_all[p2]
        p1_elo_s = elo_surf[p1][surf]
        p2_elo_s = elo_surf[p2][surf]

        p1_fs = _form(res_all[p1], FORM_SHORT)
        p2_fs = _form(res_all[p2], FORM_SHORT)
        p1_fl = _form(res_all[p1])
        p2_fl = _form(res_all[p2])
        p1_sf = _form(res_surf[p1][surf])
        p2_sf = _form(res_surf[p2][surf])

        h2h_p1 = h2h_wins[(p1, p2)]
        h2h_p2 = h2h_wins[(p2, p1)]
        h2h_total = h2h_p1 + h2h_p2
        h2h_rate = (h2h_p1 + 1.0) / (h2h_total + 2.0)

        p1_days = float((date - last_dt[p1]).days) if p1 in last_dt else 30.0
        p2_days = float((date - last_dt[p2]).days) if p2 in last_dt else 30.0

        feat_rows.append(
            {
                "p1_elo_pre": p1_elo,
                "p2_elo_pre": p2_elo,
                "p1_eloS_pre": p1_elo_s,
                "p2_eloS_pre": p2_elo_s,
                "p1_fS_pre": p1_fs,
                "p2_fS_pre": p2_fs,
                "p1_fL_pre": p1_fl,
                "p2_fL_pre": p2_fl,
                "p1_sf_pre": p1_sf,
                "p2_sf_pre": p2_sf,
                "p1_h2h_rate_pre": h2h_rate,
                "h2h_n_pre": float(h2h_total),
                "p1_days_pre": p1_days,
                "p2_days_pre": p2_days,
                "p1_exp_pre": float(exp_cnt[p1]),
                "p2_exp_pre": float(exp_cnt[p2]),
            }
        )

        score = 1.0 if p1_won else 0.0
        expected = _elo_expected(p1_elo, p2_elo)
        elo_all[p1] += ELO_K * (score - expected)
        elo_all[p2] += ELO_K * ((1 - score) - (1 - expected))

        expected_s = _elo_expected(p1_elo_s, p2_elo_s)
        elo_surf[p1][surf] += ELO_K_SURF * (score - expected_s)
        elo_surf[p2][surf] += ELO_K_SURF * ((1 - score) - (1 - expected_s))

        res_all[p1].append(1.0 if p1_won else 0.0)
        res_all[p2].append(0.0 if p1_won else 1.0)
        res_surf[p1][surf].append(1.0 if p1_won else 0.0)
        res_surf[p2][surf].append(0.0 if p1_won else 1.0)

        if p1_won:
            h2h_wins[(p1, p2)] += 1
        else:
            h2h_wins[(p2, p1)] += 1

        last_dt[p1] = date
        last_dt[p2] = date
        exp_cnt[p1] += 1
        exp_cnt[p2] += 1

    return pd.concat([df, pd.DataFrame(feat_rows, index=df.index)], axis=1)


def _make_rows(df: pd.DataFrame, flip: bool, player_attrs: dict, surf_lookup: dict, playstyle_lookup: dict) -> pd.DataFrame:
    def _v(a: str, b: str):
        col = b if flip else a
        return df[col].values if col in df.columns else np.full(len(df), np.nan)

    p1_names = df["Player_2" if flip else "Player_1"].values
    p2_names = df["Player_1" if flip else "Player_2"].values
    label = (1 - df["p1_wins"]).values if flip else df["p1_wins"].values
    h2h_rate = (
        1.0 - df["p1_h2h_rate_pre"].values if flip else df["p1_h2h_rate_pre"].values
    ) if "p1_h2h_rate_pre" in df.columns else np.full(len(df), 0.5)

    surf_vals = df["Surface"].values
    match_years = df["year"].values.astype(float)
    def _flt(names, key):
        return np.array(
            [float(player_attrs.get(_norm(name), {}).get(key, np.nan)) for name in names],
            dtype=float,
        )

    def _srf(names, key, default):
        return np.array(
            [
                float(surf_lookup.get((_norm(name), str(surface or "").lower()), {}).get(key, default))
                for name, surface in zip(names, surf_vals)
            ],
            dtype=float,
        )

    p1_birth_year = _flt(p1_names, "birth_year")
    p2_birth_year = _flt(p2_names, "birth_year")
    p1_height = _flt(p1_names, "ht")
    p2_height = _flt(p2_names, "ht")
    p1_hand = np.array([player_attrs.get(_norm(name), {}).get("hand", "U") for name in p1_names])
    p2_hand = np.array([player_attrs.get(_norm(name), {}).get("hand", "U") for name in p2_names])

    p1_styles = np.array([_player_style(name, playstyle_lookup) for name in p1_names], dtype=object)
    p2_styles = np.array([_player_style(name, playstyle_lookup) for name in p2_names], dtype=object)
    style_matchups = np.array([f"{a} vs {b}" for a, b in zip(p1_styles, p2_styles)], dtype=object)
    style_pairs = np.array([" vs ".join(sorted([a, b])) for a, b in zip(p1_styles, p2_styles)], dtype=object)
    p1_has_style = (p1_styles != STYLE_UNKNOWN).astype(int)
    p2_has_style = (p2_styles != STYLE_UNKNOWN).astype(int)

    return pd.DataFrame(
        {
            "year": df["year"].values,
            "p1_rank": _v("Rank_1", "Rank_2"),
            "p2_rank": _v("Rank_2", "Rank_1"),
            "p1_pts": _v("Pts_1", "Pts_2"),
            "p2_pts": _v("Pts_2", "Pts_1"),
            "p1_elo": _v("p1_elo_pre", "p2_elo_pre"),
            "p2_elo": _v("p2_elo_pre", "p1_elo_pre"),
            "p1_eloS": _v("p1_eloS_pre", "p2_eloS_pre"),
            "p2_eloS": _v("p2_eloS_pre", "p1_eloS_pre"),
            "p1_fS": _v("p1_fS_pre", "p2_fS_pre"),
            "p2_fS": _v("p2_fS_pre", "p1_fS_pre"),
            "p1_fL": _v("p1_fL_pre", "p2_fL_pre"),
            "p2_fL": _v("p2_fL_pre", "p1_fL_pre"),
            "p1_sf": _v("p1_sf_pre", "p2_sf_pre"),
            "p2_sf": _v("p2_sf_pre", "p1_sf_pre"),
            "h2h_rate": h2h_rate,
            "h2h_n": df["h2h_n_pre"].values if "h2h_n_pre" in df.columns else np.zeros(len(df)),
            "p1_days": _v("p1_days_pre", "p2_days_pre"),
            "p2_days": _v("p2_days_pre", "p1_days_pre"),
            "p1_exp": _v("p1_exp_pre", "p2_exp_pre"),
            "p2_exp": _v("p2_exp_pre", "p1_exp_pre"),
            "p1_age": match_years - p1_birth_year,
            "p2_age": match_years - p2_birth_year,
            "p1_ht": p1_height,
            "p2_ht": p2_height,
            "p1_hand": p1_hand,
            "p2_hand": p2_hand,
            "p1_swr": _srf(p1_names, "win_rate", np.nan),
            "p2_swr": _srf(p2_names, "win_rate", np.nan),
            "p1_sn": _srf(p1_names, "n_matches", 0),
            "p2_sn": _srf(p2_names, "n_matches", 0),
            "p1_style": p1_styles,
            "p2_style": p2_styles,
            "style_matchup": style_matchups,
            "style_pair": style_pairs,
            "p1_has_style": p1_has_style,
            "p2_has_style": p2_has_style,
            "surface": df["Surface"].values,
            "series": df["Series"].values if "Series" in df.columns else np.full(len(df), "unknown"),
            "court": df["Court"].values if "Court" in df.columns else np.full(len(df), "unknown"),
            "round": df["Round"].values,
            "best_of": df["best_of"].values,
            "label": label,
        }
    )


def build_dataset(df: pd.DataFrame, player_attrs: dict, surf_lookup: dict, playstyle_lookup: dict) -> pd.DataFrame:
    df = df.copy()
    df["p1_wins"] = (df["Player_1"] == df["Winner"]).astype(int)
    data = pd.concat(
        [
            _make_rows(df, False, player_attrs, surf_lookup, playstyle_lookup),
            _make_rows(df, True, player_attrs, surf_lookup, playstyle_lookup),
        ],
        ignore_index=True,
    )

    data["rank_diff"] = data["p1_rank"] - data["p2_rank"]
    data["pts_diff"] = data["p1_pts"].fillna(0) - data["p2_pts"].fillna(0)
    data["rank_ratio"] = data["p1_rank"] / data["p2_rank"].clip(lower=1)
    data["log_rank_diff"] = np.log1p(data["p1_rank"]) - np.log1p(data["p2_rank"])

    data["elo_diff"] = data["p1_elo"] - data["p2_elo"]
    data["elo_surf_diff"] = data["p1_eloS"] - data["p2_eloS"]
    data["surface_specialist_diff"] = data["elo_surf_diff"] - data["elo_diff"]

    data["form_short_diff"] = data["p1_fS"] - data["p2_fS"]
    data["form_long_diff"] = data["p1_fL"] - data["p2_fL"]
    data["surf_form_diff"] = data["p1_sf"] - data["p2_sf"]
    data["form_trend_diff"] = data["form_short_diff"] - data["form_long_diff"]

    data["days_rest_diff"] = data["p1_days"] - data["p2_days"]
    data["experience_diff"] = data["p1_exp"] - data["p2_exp"]

    data["age_diff"] = data["p1_age"] - data["p2_age"]
    data["ht_diff"] = data["p1_ht"] - data["p2_ht"]

    data["surf_win_rate_diff"] = data["p1_swr"] - data["p2_swr"]
    data["surf_exp_diff"] = data["p1_sn"] - data["p2_sn"]

    data["same_style"] = (data["p1_style"] == data["p2_style"]).astype(int)
    data["playstyle_coverage"] = data["p1_has_style"] + data["p2_has_style"]

    print(f"  Dataset: {len(data):,} rows ({int(data['year'].min())}-{int(data['year'].max())})")
    coverage = data["playstyle_coverage"].value_counts().sort_index()
    for level, count in coverage.items():
        label = {0: "Neither player", 1: "One player", 2: "Both players"}.get(int(level), str(level))
        print(f"    Playstyle coverage - {label}: {count:,} rows ({count / len(data) * 100:.1f}%)")
    return data


def _num_cat_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


def make_m5_pipeline() -> CalibratedClassifierCV:
    pre = _num_cat_preprocessor(NUMERIC_COLS_M5, CAT_COLS_M5)
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
    return CalibratedClassifierCV(Pipeline([("pre", pre), ("clf", xgb)]), cv=3, method="isotonic")


def make_m2_pipeline() -> CalibratedClassifierCV:
    pre = _num_cat_preprocessor(NUMERIC_COLS_M2, CAT_COLS_M2)
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
    return CalibratedClassifierCV(Pipeline([("pre", pre), ("clf", xgb)]), cv=3, method="isotonic")


def make_baseline_pipeline() -> Pipeline:
    pre = _num_cat_preprocessor(BASELINE_NUM, BASELINE_CAT)
    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    return Pipeline([("pre", pre), ("clf", lr)])


def fit_eval(model, x_tr, y_tr, x_te, y_te):
    model.fit(x_tr, y_tr)
    probs = model.predict_proba(x_te)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return (
        accuracy_score(y_te, preds),
        roc_auc_score(y_te, probs),
        brier_score_loss(y_te, probs),
        probs,
    )


def walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for test_year in sorted(data["year"].unique()):
        if test_year < 2016 or test_year > 2024:
            continue
        train = data[data["year"] < test_year]
        test = data[data["year"] == test_year]
        if len(train) < 100 or len(test) < 50:
            continue

        y_train = train["label"]
        y_test = test["label"]

        m5 = make_m5_pipeline()
        acc_m5, auc_m5, _, _ = fit_eval(
            m5,
            train[NUMERIC_COLS_M5 + CAT_COLS_M5],
            y_train,
            test[NUMERIC_COLS_M5 + CAT_COLS_M5],
            y_test,
        )

        lr = make_baseline_pipeline()
        acc_lr, auc_lr, _, _ = fit_eval(
            lr,
            train[BASELINE_NUM + BASELINE_CAT],
            y_train,
            test[BASELINE_NUM + BASELINE_CAT],
            y_test,
        )

        rank_base = accuracy_score(y_test, (test["rank_diff"] < 0).astype(int))
        rows.append(
            {
                "test_year": test_year,
                "acc_m5": acc_m5,
                "auc_m5": auc_m5,
                "acc_lr": acc_lr,
                "auc_lr": auc_lr,
                "rank_baseline": rank_base,
            }
        )

    wf = pd.DataFrame(rows)
    if not wf.empty:
        print("\nWalk-forward (2016-2024)")
        print(wf.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return wf


def plot_roc_curve(y_true, probs, acc: float, auc: float) -> None:
    fpr, tpr, _ = roc_curve(y_true, probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color="darkorange", label=f"Model 5 XGBoost+Playstyle (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], ":", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - Model 5\n(Test 2021-2024 | Accuracy={acc:.4f})")
    ax.legend()
    _save("roc_curve.png")


def plot_calibration(y_true, probs) -> None:
    fop, mpv = calibration_curve(y_true, probs, n_bins=10)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(mpv, fop, "o-", color="darkorange", lw=2, label="Model 5")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve - Model 5")
    ax.legend()
    _save("calibration.png")


def plot_confusion_matrix_chart(y_true, probs) -> None:
    cm = confusion_matrix(y_true, (probs >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Oranges")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Loss", "Predicted Win"])
    ax.set_yticklabels(["Actual Loss", "Actual Win"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.set_title("Confusion Matrix - Model 5 (Test 2021-2024)")
    plt.colorbar(im, ax=ax)
    _save("confusion_matrix.png")


def plot_feature_importance(model) -> None:
    result = _extract_importances(model)
    if result is None:
        return
    names, importances = result
    df = pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False)
    df.to_csv(os.path.join(PLOTS_DIR, "feature_importance.csv"), index=False)
    top = df.head(25).sort_values("importance", ascending=True)
    colors = [_feat_color(feature) for feature in top["feature"]]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(top["feature"], top["importance"], color=colors)
    ax.set_xlabel("Feature Importance (XGBoost Gain)")
    ax.set_title("Top 25 Feature Importances - Model 5 (crimson = playstyle features)")
    ax.legend(
        handles=[
            mpatches.Patch(facecolor="crimson", label="Injury reports (Group H)"),
            mpatches.Patch(facecolor="darkorange", label="Elo"),
            mpatches.Patch(facecolor="steelblue", label="Rolling form"),
            mpatches.Patch(facecolor="teal", label="Surface history"),
            mpatches.Patch(facecolor="mediumpurple", label="Rank/skill"),
        ],
        fontsize=9,
    )
    _save("feature_importance.png")


def plot_walk_forward(wf: pd.DataFrame) -> None:
    if wf.empty:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    ax1.plot(wf.test_year, wf.acc_m5, "o-", color="darkorange", lw=2, label="Model 5")
    ax1.plot(wf.test_year, wf.rank_baseline, "^:", color="gray", lw=1.5, label="Rank baseline")
    ax1.plot(wf.test_year, wf.acc_lr, "s--", color="steelblue", lw=1.5, label="Logistic baseline")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Walk-Forward Validation - Accuracy (Model 5)")
    ax1.legend()
    ax1.set_ylim(0.55, 0.82)
    ax1.grid(axis="y", alpha=0.3)

    ax2.plot(wf.test_year, wf.auc_m5, "o-", color="darkorange", lw=2, label="Model 5")
    ax2.plot(wf.test_year, wf.auc_lr, "s--", color="steelblue", lw=1.5, label="Logistic baseline")
    ax2.set_xlabel("Test Year")
    ax2.set_ylabel("ROC AUC")
    ax2.set_title("Walk-Forward Validation - ROC AUC (Model 5)")
    ax2.set_ylim(0.60, 0.85)
    ax2.grid(axis="y", alpha=0.3)
    _save("walk_forward.png")


def plot_accuracy_by_surface(test: pd.DataFrame, model) -> None:
    valid, accs, bases, ns = [], [], [], []
    for surf in sorted(s for s in test["surface"].unique() if s not in ("unknown", "nan") and pd.notna(s)):
        subset = test[test["surface"] == surf]
        if len(subset) < 100:
            continue
        probs = model.predict_proba(subset[NUMERIC_COLS_M5 + CAT_COLS_M5])[:, 1]
        accs.append(accuracy_score(subset["label"], (probs >= 0.5).astype(int)))
        bases.append(accuracy_score(subset["label"], (subset["rank_diff"] < 0).astype(int)))
        ns.append(len(subset))
        valid.append(surf)

    if not valid:
        return

    x = np.arange(len(valid))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, accs, width, color="darkorange", alpha=0.85, label="Model 5")
    ax.bar(x + width / 2, bases, width, color="gray", alpha=0.75, label="Rank baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{surf}\n(n={n:,})" for surf, n in zip(valid, ns)])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 0.9)
    ax.set_title("Accuracy by Surface - Model 5 vs Rank Baseline (Test 2021-2024)")
    ax.legend()
    _save("accuracy_by_surface.png")


def plot_accuracy_by_confidence(y_true, probs) -> None:
    df = pd.DataFrame({"prob": probs, "label": y_true.values})
    df["conf"] = np.abs(df["prob"] - 0.5)
    df["correct"] = ((df["prob"] >= 0.5).astype(int) == df["label"]).astype(int)
    bins = np.linspace(0, 0.5, 11)
    labels = [f"{int(bins[i] * 100)}-{int(bins[i + 1] * 100)}%" for i in range(len(bins) - 1)]
    df["bin"] = pd.cut(df["conf"], bins=bins, labels=labels, include_lowest=True)
    grp = df.groupby("bin", observed=True)["correct"].agg(acc="mean", n="count").reset_index()
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(grp["bin"].astype(str), grp["acc"], "o-", color="darkorange", lw=2)
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0.45, 1.0)
    ax1.set_title("Accuracy vs Prediction Confidence - Model 5")
    ax1.tick_params(axis="x", rotation=45)
    ax2 = ax1.twinx()
    ax2.bar(grp["bin"].astype(str), grp["n"], color="steelblue", alpha=0.2)
    ax2.set_ylabel("Count")
    _save("accuracy_by_confidence.png")


def plot_prob_histogram(y_true, probs) -> None:
    df = pd.DataFrame({"prob": probs, "label": y_true.values})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[df.label == 1]["prob"], bins=30, alpha=0.6, color="darkorange", label="Wins")
    ax.hist(df[df.label == 0]["prob"], bins=30, alpha=0.6, color="steelblue", label="Losses")
    ax.axvline(0.5, color="black", linestyle="--")
    ax.set_xlabel("Predicted Win Probability")
    ax.set_ylabel("Count")
    ax.set_title("Predicted Probability Distribution - Model 5")
    ax.legend()
    _save("prob_histogram.png")


def plot_vs_baseline_roc(y_true, probs_m5, probs_lr) -> None:
    fpr_m5, tpr_m5, _ = roc_curve(y_true, probs_m5)
    fpr_lr, tpr_lr, _ = roc_curve(y_true, probs_lr)
    auc_m5 = roc_auc_score(y_true, probs_m5)
    auc_lr = roc_auc_score(y_true, probs_lr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_m5, tpr_m5, lw=2, color="darkorange", label=f"Model 5 (AUC={auc_m5:.4f})")
    ax.plot(fpr_lr, tpr_lr, lw=2, color="steelblue", linestyle="--", label=f"Logistic baseline (AUC={auc_lr:.4f})")
    ax.plot([0, 1], [0, 1], ":", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Comparison - Model 5 vs Logistic Baseline (Test 2021-2024)")
    ax.legend()
    _save("vs_baseline_roc.png")


def plot_yearly_comparison(wf: pd.DataFrame) -> None:
    if wf.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wf.test_year, wf.acc_m5, "o-", color="darkorange", lw=2, label="Model 5")
    ax.plot(wf.test_year, wf.acc_lr, "s--", color="steelblue", lw=2, label="Logistic baseline")
    ax.plot(wf.test_year, wf.rank_baseline, "^:", color="gray", lw=1.5, label="Rank baseline")
    ax.fill_between(wf.test_year, wf.acc_lr, wf.acc_m5, alpha=0.10, color="darkorange", label="Model 5 gain over logistic")
    ax.set_xlabel("Test Year")
    ax.set_ylabel("Accuracy")
    ax.set_title("Yearly Accuracy - Model 5 vs Logistic vs Rank Baseline")
    ax.legend()
    ax.set_ylim(0.55, 0.82)
    ax.grid(axis="y", alpha=0.3)
    _save("yearly_comparison.png")


def plot_feature_groups(model) -> None:
    result = _extract_importances(model)
    if result is None:
        return
    names, importances = result
    imp_df = pd.DataFrame({"feature": names, "importance": importances})
    totals = {}
    for group, features in GROUP_LABELS.items():
        total = 0.0
        for feature in features:
            total += imp_df.loc[imp_df["feature"].str.contains(feature, regex=False), "importance"].sum()
        totals[group] = total
    grp = pd.DataFrame({"group": list(totals.keys()), "importance": list(totals.values())})
    grp = grp.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(grp["group"], grp["importance"], color=[GROUP_COLORS_BAR[group] for group in grp["group"]])
    ax.set_xlabel("Total Feature Importance")
    ax.set_title("Feature Importance by Group - Model 5")
    _save("feature_groups.png")


def plot_elo_validation(test: pd.DataFrame) -> None:
    df = test[["elo_diff", "label"]].dropna().copy()
    bin_edges = [-600, -300, -150, -75, -25, 25, 75, 150, 300, 600]
    bin_labels = [f"{(a + b) // 2:+d}" for a, b in zip(bin_edges[:-1], bin_edges[1:])]
    df["bin"] = pd.cut(df["elo_diff"], bins=bin_edges, labels=bin_labels)
    grp = df.groupby("bin", observed=True)["label"].agg(win_rate="mean", n="count").reset_index().dropna()
    theory = []
    for label in grp["bin"]:
        elo_gap = int(str(label))
        theory.append(_elo_expected(elo_gap, 0))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grp["bin"].astype(str), grp["win_rate"], "o-", color="darkorange", label="Actual")
    ax.plot(grp["bin"].astype(str), theory, "s--", color="steelblue", label="Elo theory")
    ax.set_xlabel("Elo Difference Bin")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Elo Signal Validation - Actual Win Rate vs Elo Theory (Model 5)")
    ax.legend()
    _save("elo_validation.png")


def plot_accuracy_by_round(test: pd.DataFrame, model) -> None:
    order = ["r128", "r64", "r32", "r16", "qf", "sf", "f", "rr", "br"]
    rounds = [rnd for rnd in order if rnd in test["round"].unique()]
    rounds += [
        rnd
        for rnd in test["round"].unique()
        if rnd not in rounds and rnd not in ("unknown", "nan") and pd.notna(rnd)
    ]
    valid, accs, ns = [], [], []
    for rnd in rounds:
        subset = test[test["round"] == rnd]
        if len(subset) < 50:
            continue
        probs = model.predict_proba(subset[NUMERIC_COLS_M5 + CAT_COLS_M5])[:, 1]
        accs.append(accuracy_score(subset["label"], (probs >= 0.5).astype(int)))
        ns.append(len(subset))
        valid.append(rnd)
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(valid, accs, color="darkorange", alpha=0.85)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 0.85)
    ax.set_title("Accuracy by Round - Model 5 (Test 2021-2024)")
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"n={n:,}", ha="center", fontsize=8)
    _save("accuracy_by_round.png")


def plot_rank_gap_accuracy(test: pd.DataFrame, probs) -> None:
    df = test[["rank_diff", "label"]].copy()
    df["prob"] = probs
    df["correct"] = ((df["prob"] >= 0.5).astype(int) == df["label"]).astype(int)
    df["abs_gap"] = df["rank_diff"].abs()
    bins = [0, 10, 25, 50, 100, 200, 500, 2000]
    labels = ["0-10", "10-25", "25-50", "50-100", "100-200", "200-500", "500+"]
    df["gap_bin"] = pd.cut(df["abs_gap"], bins=bins, labels=labels)
    grp = (
        df.groupby("gap_bin", observed=True)
        .agg(m5=("correct", "mean"), n=("correct", "count"))
        .reset_index()
        .dropna()
    )
    rank_base = []
    for label in grp["gap_bin"]:
        subset = df[df["gap_bin"] == label]
        rank_base.append(accuracy_score(subset["label"], (subset["rank_diff"] < 0).astype(int)))
    grp["rank_base"] = rank_base

    x = np.arange(len(grp))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, grp["m5"], width, color="darkorange", alpha=0.85, label="Model 5")
    ax.bar(x + width / 2, grp["rank_base"], width, color="gray", alpha=0.75, label="Rank baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{label}\n(n={n:,})" for label, n in zip(grp["gap_bin"], grp["n"])])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 0.95)
    ax.set_title("Accuracy by Rank Gap - Model 5 vs Rank Baseline (Test 2021-2024)")
    ax.legend()
    _save("rank_gap_accuracy.png")

def plot_playstyle_coverage(data: pd.DataFrame) -> None:
    coverage = data["playstyle_coverage"].value_counts().sort_index()
    labels_map = {0: "Neither player\nhas style", 1: "One player\nhas style", 2: "Both players\nhave styles"}
    labels = [labels_map.get(int(level), str(level)) for level in coverage.index]
    counts = coverage.values
    pct = counts / counts.sum() * 100.0
    colors = ["#d62728", "#ff7f0e", "#2ca02c"][: len(counts)]

    style_counts = data.loc[data["p1_style"] != STYLE_UNKNOWN, "p1_style"].value_counts()
    ordered_styles = [style for style in STYLE_ORDER if style in style_counts.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.pie(pct, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90, textprops={"fontsize": 10})
    ax1.set_title("Playstyle Coverage\n(% of dataset rows)")

    bars = ax2.bar(
        ordered_styles,
        [style_counts[style] for style in ordered_styles],
        color=["crimson", "darkorange", "steelblue", "seagreen"][: len(ordered_styles)],
        edgecolor="white",
    )
    ax2.set_title("Playstyle Distribution in Match Rows")
    ax2.set_ylabel("Count")
    ax2.tick_params(axis="x", rotation=20)
    for bar, style in zip(bars, ordered_styles):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 150, f"{style_counts[style]:,}", ha="center", fontsize=8)
    fig.suptitle("How Complete Is the Playstyle Layer?", fontsize=13, fontweight="bold", y=1.02)
    _save("playstyle_coverage.png")


def plot_playstyle_feature_importance(model) -> None:
    result = _extract_importances(model)
    if result is None:
        return
    names, importances = result
    df = pd.DataFrame({"feature": names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(30).sort_values("importance", ascending=True)
    style_keywords = ["same_style", "p1_style", "p2_style", "style_matchup", "style_pair"]
    colors = ["crimson" if any(keyword in feature for keyword in style_keywords) else "#aaaaaa" for feature in df["feature"]]

    fig, ax = plt.subplots(figsize=(11, 10))
    bars = ax.barh(df["feature"], df["importance"], color=colors)
    ax.set_xlabel("Feature Importance (XGBoost Gain)")
    ax.set_title("Feature Importance - Model 5\n(crimson bars = playstyle features; grey = inherited Model 2 features)")
    ax.legend(
        handles=[
            mpatches.Patch(facecolor="crimson", label="Playstyle features"),
            mpatches.Patch(facecolor="#aaaaaa", label="Model 2 features"),
        ],
        loc="lower right",
        fontsize=10,
    )
    for bar, feature, importance in zip(bars, df["feature"], df["importance"]):
        if any(keyword in feature for keyword in style_keywords):
            ax.text(importance + 0.0001, bar.get_y() + bar.get_height() / 2, f"  {importance:.4f}", va="center", fontsize=8.5)
    _save("playstyle_feature_importance.png")


def plot_m5_vs_m2_comparison(acc5, auc5, brier5, acc2, auc2, brier2, acc_lr, auc_lr, brier_lr) -> None:
    models = ["Logistic\nBaseline", "Model 2\n(no playstyle)", "Model 5\n(+playstyle)"]
    accs = [acc_lr, acc2, acc5]
    aucs = [auc_lr, auc2, auc5]
    briers = [brier_lr, brier2, brier5]
    colors = ["steelblue", "darkorange", "crimson"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    settings = [
        (axes[0], accs, "Accuracy\n(up is better)", "Accuracy", False),
        (axes[1], aucs, "ROC AUC\n(up is better)", "ROC AUC", False),
        (axes[2], briers, "Brier Score\n(down is better)", "Brier Score", True),
    ]
    for ax, values, title, ylabel, invert in settings:
        bars = ax.bar(models, values, color=colors, edgecolor="white", width=0.5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        vmin = min(values) * 0.985
        vmax = max(values) * 1.015
        ax.set_ylim(vmin, vmax)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (vmax - vmin) * 0.01, f"{value:.4f}", ha="center")
        delta = values[2] - values[1]
        good = (delta > 0 and not invert) or (delta < 0 and invert)
        sign = "+" if delta > 0 else ""
        ax.annotate(
            f"M5 vs M2: {sign}{delta:.4f}",
            xy=(2, values[2]),
            xytext=(1.5, vmax * 0.997),
            fontsize=9,
            color="green" if good else "red",
            fontweight="bold",
            ha="center",
        )

    fig.suptitle("Model 5 vs Model 2 vs Logistic Baseline\n(Test 2021-2024 - same split)", fontsize=13, fontweight="bold")
    _save("m5_vs_m2_comparison.png")


def plot_m5_vs_m2_yearly(wf: pd.DataFrame, acc_m2_holdout: float) -> None:
    if wf.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wf.test_year, wf.acc_m5, "o-", color="crimson", lw=2, label="Model 5 - XGBoost + playstyle")
    ax.plot(wf.test_year, wf.acc_lr, "s--", color="steelblue", lw=1.5, label="Logistic baseline")
    ax.plot(wf.test_year, wf.rank_baseline, "^:", color="gray", lw=1.2, label="Rank baseline")
    ax.axhline(
        acc_m2_holdout,
        color="darkorange",
        linestyle="-.",
        lw=1.5,
        alpha=0.8,
        label=f"Model 2 holdout accuracy ({acc_m2_holdout:.4f})",
    )
    ax.fill_between(wf.test_year, wf.acc_lr, wf.acc_m5, alpha=0.12, color="crimson", label="M5 playstyle gain over logistic")
    ax.set_xlabel("Test Year")
    ax.set_ylabel("Accuracy")
    ax.set_title("Walk-Forward Accuracy - Model 5 vs Model 2 reference\n(orange line = Model 2 final holdout accuracy)")
    ax.legend(fontsize=9)
    ax.set_ylim(0.55, 0.82)
    ax.grid(axis="y", alpha=0.3)
    _save("m5_vs_m2_yearly.png")


def plot_playstyle_matchup_heatmap(test: pd.DataFrame) -> None:
    df = test[(test["p1_style"] != STYLE_UNKNOWN) & (test["p2_style"] != STYLE_UNKNOWN)].copy()
    if df.empty:
        return
    win_rate = df.pivot_table(index="p1_style", columns="p2_style", values="label", aggfunc="mean")
    counts = df.pivot_table(index="p1_style", columns="p2_style", values="label", aggfunc="count")
    win_rate = win_rate.reindex(index=STYLE_ORDER, columns=STYLE_ORDER)
    counts = counts.reindex(index=STYLE_ORDER, columns=STYLE_ORDER)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(win_rate.values, cmap="coolwarm", vmin=0.25, vmax=0.75)
    ax.set_xticks(range(len(STYLE_ORDER)))
    ax.set_yticks(range(len(STYLE_ORDER)))
    ax.set_xticklabels(STYLE_ORDER, rotation=25, ha="right")
    ax.set_yticklabels(STYLE_ORDER)
    ax.set_xlabel("Opponent playstyle")
    ax.set_ylabel("Player playstyle")
    ax.set_title("Who tends to win each playstyle matchup?\n(cell = P1 win rate, annotation includes n)")
    for i, row_style in enumerate(STYLE_ORDER):
        for j, col_style in enumerate(STYLE_ORDER):
            wr = win_rate.loc[row_style, col_style]
            n = counts.loc[row_style, col_style]
            if pd.notna(wr):
                ax.text(j, i, f"{wr:.2f}\n(n={int(n):,})", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save("playstyle_matchup_heatmap.png")


def plot_playstyle_surface_matchups(test: pd.DataFrame) -> None:
    surfaces = [surface for surface in ["hard", "clay", "grass"] if surface in set(test["surface"])]
    if not surfaces:
        return
    fig, axes = plt.subplots(1, len(surfaces), figsize=(6 * len(surfaces), 6), squeeze=False)
    for ax, surface in zip(axes[0], surfaces):
        subset = test[
            (test["surface"] == surface)
            & (test["p1_style"] != STYLE_UNKNOWN)
            & (test["p2_style"] != STYLE_UNKNOWN)
        ].copy()
        if subset.empty:
            ax.axis("off")
            continue
        win_rate = subset.pivot_table(index="p1_style", columns="p2_style", values="label", aggfunc="mean")
        counts = subset.pivot_table(index="p1_style", columns="p2_style", values="label", aggfunc="count")
        win_rate = win_rate.reindex(index=STYLE_ORDER, columns=STYLE_ORDER)
        counts = counts.reindex(index=STYLE_ORDER, columns=STYLE_ORDER)
        im = ax.imshow(win_rate.values, cmap="coolwarm", vmin=0.25, vmax=0.75)
        ax.set_xticks(range(len(STYLE_ORDER)))
        ax.set_yticks(range(len(STYLE_ORDER)))
        ax.set_xticklabels(STYLE_ORDER, rotation=25, ha="right", fontsize=8)
        ax.set_yticklabels(STYLE_ORDER, fontsize=8)
        ax.set_title(surface.title())
        for i, row_style in enumerate(STYLE_ORDER):
            for j, col_style in enumerate(STYLE_ORDER):
                n = counts.loc[row_style, col_style]
                if pd.notna(n) and n >= 20:
                    ax.text(j, i, f"{win_rate.loc[row_style, col_style]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Do playstyle matchups change by surface?", fontsize=13, fontweight="bold", y=1.02)
    _save("playstyle_surface_matchups.png")


def plot_playstyle_matchup_accuracy(test: pd.DataFrame, probs5: np.ndarray, probs2: np.ndarray) -> None:
    df = test.copy()
    df["correct_m5"] = ((probs5 >= 0.5).astype(int) == df["label"]).astype(int)
    df["correct_m2"] = ((probs2 >= 0.5).astype(int) == df["label"]).astype(int)
    df = df[(df["style_pair"].notna()) & (df["style_pair"] != f"{STYLE_UNKNOWN} vs {STYLE_UNKNOWN}")]
    grp = (
        df.groupby("style_pair", observed=True)
        .agg(acc_m5=("correct_m5", "mean"), acc_m2=("correct_m2", "mean"), n=("label", "count"))
        .reset_index()
        .sort_values("n", ascending=False)
        .head(8)
        .sort_values("acc_m5", ascending=True)
    )
    if grp.empty:
        return

    y = np.arange(len(grp))
    height = 0.36
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(y - height / 2, grp["acc_m2"], height, color="darkorange", alpha=0.75, label="Model 2")
    ax.barh(y + height / 2, grp["acc_m5"], height, color="crimson", alpha=0.85, label="Model 5")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{pair}\n(n={n:,})" for pair, n in zip(grp["style_pair"], grp["n"])])
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0.5, 0.85)
    ax.set_title("Where does playstyle help most?\n(top style pairings in the 2021-2024 holdout)")
    ax.legend()
    _save("playstyle_matchup_accuracy.png")


def main() -> None:
    print("\n" + "=" * 68)
    print("  Model 5 - XGBoost + Elo + Rolling Form + Playstyle Matchups")
    print("=" * 68)

    print("\nLoading data ...")
    processed = load_processed(PROC_DIR)
    matches = load_initial_matches(INIT_DIR)
    surf_df = load_surface_results(INIT_DIR)
    playstyle_lookup, _ = load_playstyle_data(PLAYSTYLE_DIR)

    print("\nBuilding static lookups ...")
    player_attrs = build_player_attrs(matches)
    surf_lookup = build_surface_lookup(surf_df)

    print("\nComputing dynamic features (Elo, form, H2H) ...")
    processed = compute_dynamic_features(processed)

    print("\nBuilding symmetric dataset ...")
    data = build_dataset(processed, player_attrs, surf_lookup, playstyle_lookup)
    data_comp = data[(data["year"] >= 2014) & (data["year"] <= 2024)].copy()

    print("\nRunning walk-forward validation (2016-2024) ...")
    wf = walk_forward(data_comp)

    print("\nTraining final models (train 2014-2020 / test 2021-2024) ...")
    train = data_comp[data_comp["year"] <= 2020]
    test = data_comp[data_comp["year"] >= 2021]

    model_m5 = make_m5_pipeline()
    acc5, auc5, brier5, probs5 = fit_eval(
        model_m5,
        train[NUMERIC_COLS_M5 + CAT_COLS_M5],
        train["label"],
        test[NUMERIC_COLS_M5 + CAT_COLS_M5],
        test["label"],
    )

    print("  Training M2-equivalent (no playstyle) for comparison ...")
    model_m2 = make_m2_pipeline()
    acc2, auc2, brier2, probs2 = fit_eval(
        model_m2,
        train[NUMERIC_COLS_M2 + CAT_COLS_M2],
        train["label"],
        test[NUMERIC_COLS_M2 + CAT_COLS_M2],
        test["label"],
    )

    lr_model = make_baseline_pipeline()
    acc_lr, auc_lr, brier_lr, probs_lr = fit_eval(
        lr_model,
        train[BASELINE_NUM + BASELINE_CAT],
        train["label"],
        test[BASELINE_NUM + BASELINE_CAT],
        test["label"],
    )

    rank_base = accuracy_score(test["label"], (test["rank_diff"] < 0).astype(int))

    print(f"\n{'=' * 74}")
    print("  Results - Model 5 vs Model 2 vs Logistic Baseline vs Rank Baseline")
    print(f"{'=' * 74}")
    print(f"{'Metric':<28} {'Model 5':>12}  {'Model 2':>10}  {'Logistic LR':>12}  {'Rank base':>10}")
    print(f"{'-' * 74}")
    print(f"{'Accuracy':<28} {acc5:>12.4f}  {acc2:>10.4f}  {acc_lr:>12.4f}  {rank_base:>10.4f}")
    print(f"{'ROC AUC':<28} {auc5:>12.4f}  {auc2:>10.4f}  {auc_lr:>12.4f}  {'-':>10}")
    print(f"{'Brier Score':<28} {brier5:>12.4f}  {brier2:>10.4f}  {brier_lr:>12.4f}  {'-':>10}  (down is better)")
    print(f"{'-' * 74}")
    print(f"{'Delta Accuracy  M5 vs M2':<28} {acc5 - acc2:>+12.4f}")
    print(f"{'Delta AUC       M5 vs M2':<28} {auc5 - auc2:>+12.4f}")
    print(f"{'Delta Brier     M5 vs M2':<28} {brier5 - brier2:>+12.4f}  (negative = better)")
    print(f"{'-' * 74}")
    print(f"{'Delta Accuracy  M5 vs LR':<28} {acc5 - acc_lr:>+12.4f}")
    print(f"{'Delta AUC       M5 vs LR':<28} {auc5 - auc_lr:>+12.4f}")
    print(f"{'=' * 74}\n")

    joblib.dump(model_m5, os.path.join(ROOT, "model_5.joblib"))
    print(f"Saved model -> {os.path.join(ROOT, 'model_5.joblib')}")

    print("\nGenerating plots ...")
    plot_roc_curve(test["label"], probs5, acc5, auc5)
    plot_calibration(test["label"], probs5)
    plot_confusion_matrix_chart(test["label"], probs5)
    plot_feature_importance(model_m5)
    plot_walk_forward(wf)
    plot_accuracy_by_surface(test, model_m5)
    plot_accuracy_by_confidence(test["label"], probs5)
    plot_prob_histogram(test["label"], probs5)
    plot_vs_baseline_roc(test["label"], probs5, probs_lr)
    plot_yearly_comparison(wf)
    plot_feature_groups(model_m5)
    plot_elo_validation(test)
    plot_accuracy_by_round(test, model_m5)
    plot_rank_gap_accuracy(test, probs5)

    plot_playstyle_coverage(data_comp)
    plot_playstyle_feature_importance(model_m5)
    plot_m5_vs_m2_comparison(acc5, auc5, brier5, acc2, auc2, brier2, acc_lr, auc_lr, brier_lr)
    plot_m5_vs_m2_yearly(wf, acc2)
    plot_playstyle_matchup_heatmap(test)
    plot_playstyle_surface_matchups(test)
    plot_playstyle_matchup_accuracy(test, probs5, probs2)

    print(f"\nAll plots saved to: {PLOTS_DIR}")
    print("Done.\n")


if __name__ == "__main__":
    main()
